import pandas as pd
import threading
import asyncio
import os
import time
from datetime import datetime, timedelta, date
from fastapi import WebSocket
from kiteconnect import KiteTicker
from utils.common import kite, logger, get_active_token, supabase, get_ist_time
import math
import random
import requests

ACTIVE_OPTION_TRADES = {}
last_mock_prices = {}
# Global cache for NFO instruments to avoid heavy API calls
NFO_INSTRUMENTS_CACHE = None

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🟢 React UI Connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("🔴 React UI Disconnected.")

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for connection in dead:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
                logger.info("🧹 Removed stale WebSocket connection during broadcast.")

ws_manager = ConnectionManager()

ticker = None
token_to_symbol = {}
fastapi_loop = None
last_check_minute = -1

def get_nifty_weekly_options():
    global NFO_INSTRUMENTS_CACHE
    try:
        nifty_quote = kite.quote("NSE:NIFTY 50")
        spot_price = nifty_quote["NSE:NIFTY 50"]["last_price"]
        atm_strike = round(spot_price / 100) * 100

        # Cache instruments if not already done
        if NFO_INSTRUMENTS_CACHE is None or date.today() != getattr(NFO_INSTRUMENTS_CACHE, '_cache_date', None):
            NFO_INSTRUMENTS_CACHE = pd.DataFrame(kite.instruments("NFO"))
            NFO_INSTRUMENTS_CACHE._cache_date = date.today()
        
        df = NFO_INSTRUMENTS_CACHE
        df_nifty = df[(df["name"] == "NIFTY") & (df["segment"] == "NFO-OPT")].copy()
        df_nifty["expiry"] = pd.to_datetime(df_nifty["expiry"]).dt.date
        
        today_val = date.today()
        future_expiries = df_nifty[df_nifty["expiry"] >= today_val]["expiry"].unique()
        if len(future_expiries) == 0: return {}, []
            
        current_expiry = min(future_expiries)
        df_curr = df_nifty[df_nifty["expiry"] == current_expiry]

        atm_strike = round(spot_price / 50) * 50
        strikes = [atm_strike + (i * 50) for i in range(-20, 21)]
        options_tokens = []
        token_map = {}
        
        for strike in strikes:
            for opt_type in ["CE", "PE"]:
                row = df_curr[(df_curr["strike"] == strike) & (df_curr["instrument_type"] == opt_type)]
                if not row.empty:
                    tkn = int(row.iloc[0]["instrument_token"])
                    options_tokens.append(tkn)
                    token_map[tkn] = {"symbol": row.iloc[0]["tradingsymbol"], "strike": int(strike), "type": opt_type}
                
        nifty_spot_token = 256265 
        options_tokens.append(nifty_spot_token)
        token_map[nifty_spot_token] = {"symbol": "NIFTY 50", "strike": "SPOT", "type": "SPOT"}

        # 🟢 FIX: Ensure we never drop active trades during a WebSocket reconnect
        for sym in list(ACTIVE_OPTION_TRADES.keys()):
            trade_row = df[df["tradingsymbol"] == sym]
            if not trade_row.empty:
                tkn = int(trade_row.iloc[0]["instrument_token"])
                if tkn not in token_map:
                    options_tokens.append(tkn)
                    token_map[tkn] = {"symbol": sym, "strike": int(trade_row.iloc[0]["strike"]), "type": trade_row.iloc[0]["instrument_type"]}

        return token_map, options_tokens
    except Exception as e:
        logger.error(f"❌ Error fetching Nifty options: {e}")
        return {}, []

def is_candle_green(interval_minutes):
    try:
        nifty_token = 256265
        # 🟢 FIX 1: Use IST time to prevent cloud server UTC bugs
        to_dt = get_ist_time()
        from_dt = to_dt - timedelta(days=2)
        
        interval_map = {5: "5minute", 15: "15minute", 60: "60minute"}
        
        # 1. Force string format so Kite doesn't default to midnight (yesterday)
        to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")
        from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        data = kite.historical_data(nifty_token, from_str, to_str, interval_map[interval_minutes])
        
        if not data or len(data) < 1: return False
        
        live_candle = data[-1]
        
        # 2. Kite's historical 'close' is delayed for forming candles. 
        # We MUST fetch the true live LTP to compare against the interval's open.
        quote = kite.quote("NSE:NIFTY 50")
        live_ltp = quote["NSE:NIFTY 50"]["last_price"]
        
        return live_ltp > live_candle['open']
    except Exception as e:
        logger.error(f"Error checking {interval_minutes}min candle: {e}")
        return False

def get_5_percent_otm_ce():
    global NFO_INSTRUMENTS_CACHE
    try:
        nifty_quote = kite.quote("NSE:NIFTY 50")
        spot_price = nifty_quote["NSE:NIFTY 50"]["last_price"]
        target_strike = round((spot_price * 1.05) / 100) * 100
        
        if NFO_INSTRUMENTS_CACHE is None or date.today() != getattr(NFO_INSTRUMENTS_CACHE, '_cache_date', None):
            NFO_INSTRUMENTS_CACHE = pd.DataFrame(kite.instruments("NFO"))
            NFO_INSTRUMENTS_CACHE._cache_date = date.today()
            
        df = NFO_INSTRUMENTS_CACHE
        df_nifty = df[(df["name"] == "NIFTY") & (df["strike"] == target_strike) & (df["instrument_type"] == "CE")].copy()
        df_nifty["expiry"] = pd.to_datetime(df_nifty["expiry"]).dt.date
        
        future_expiries = df_nifty[df_nifty["expiry"] >= date.today()]["expiry"].unique()
        if len(future_expiries) > 0:
            current_expiry = min(future_expiries)
            final_option = df_nifty[df_nifty["expiry"] == current_expiry].iloc[0]
            return {"symbol": final_option["tradingsymbol"], "token": int(final_option["instrument_token"])}
    except Exception as e:
        logger.error(f"Error finding 5% OTM CE: {e}")
    return None

def run_triple_green_strategy():
    now = get_ist_time()
    if now.hour > 15 or (now.hour == 15 and now.minute >= 30) or now.hour < 9:
        return
    logger.info("🔍 Checking Triple Green Strategy...")
    if is_candle_green(5) and is_candle_green(15) and is_candle_green(60):
        logger.info("🚀 TRIPLE GREEN DETECTED!")
        target_ce = get_5_percent_otm_ce()
        if target_ce:
            sym = target_ce["symbol"]
            quote = kite.quote(f"NFO:{sym}")
            ltp = quote.get(f"NFO:{sym}", {}).get("last_price", 0)
            
            if ltp > 0 and sym not in ACTIVE_OPTION_TRADES:
                ACTIVE_OPTION_TRADES[sym] = {
                    "entry": ltp,
                    "sl": round(ltp * 0.99, 2),
                    "target": round(ltp * 1.02, 2)
                }
                logger.info(f"✅ Auto-Buy Signal: {sym} at {ltp}")
    else:
        logger.info("⏭️ Strategy conditions not met.")

def on_ticks(ws, ticks):
    global fastapi_loop, last_check_minute
    if not fastapi_loop: return

    curr_min = datetime.now().minute
    if curr_min % 5 == 0 and curr_min != last_check_minute:
        last_check_minute = curr_min
        threading.Thread(target=run_triple_green_strategy).start()

    formatted_ticks = []

    for tick in ticks:
        token = tick['instrument_token']
        info = token_to_symbol.get(token, {})
        ltp = tick.get('last_price', 0)
        
        # Zerodha provides 'ohlc' data in 'MODE_FULL'
        ohlc = tick.get('ohlc', {})
        open_price = ohlc.get('open', ltp) # Fallback to LTP if market just opened
        
        # 🟢 NEW: PnL % calculation based on Trade Entry Price
        symbol = info.get("symbol")
        change_percent = 0
        
        # Check if we have an active trade for this symbol
        if symbol in ACTIVE_OPTION_TRADES:
            entry_price = ACTIVE_OPTION_TRADES[symbol].get("entry", 0)
            if entry_price > 0:
                # Calculate PnL % relative to your Buy Price
                change_percent = round(((ltp - entry_price) / entry_price) * 100, 2)
        else:
            # Fallback to standard Day Change % if no trade is active
            if open_price > 0:
                change_percent = round(((ltp - open_price) / open_price) * 100, 2)

        formatted_ticks.append({
            "symbol": info.get("symbol"),
            "strike": info.get("strike", 0),
            "type": info.get("type", "UNKNOWN"),
            "ltp": ltp,
            "open": open_price,
            "change_percent": change_percent,
            "volume": tick.get('volume_traded', 0),
            "oi": tick.get('oi', 0)
        })

    for sym, trade_data in list(ACTIVE_OPTION_TRADES.items()):
        # Find the current LTP for this specific traded symbol
        tick_info = next((t for t in formatted_ticks if t['symbol'] == sym), None)
        if not tick_info: continue
        
        ltp = tick_info['ltp']
        side = trade_data['type']
        sl = trade_data.get('sl')
        target = trade_data.get('target')
        trade_id = trade_data.get('trade_id')

        hit_sl = False
        hit_target = False

        # Only check SL/TP if the user has actually set them
        if sl is not None:
            hit_sl = (side == "BUY" and ltp <= sl) or (side == "SELL" and ltp >= sl)
        if target is not None:
            hit_target = (side == "BUY" and ltp >= target) or (side == "SELL" and ltp <= target)

        if hit_sl or hit_target:
            reason = f"AUTO_EXIT: {'SL' if hit_sl else 'Target'} Hit"
            logger.info(f"🚨 {reason} for {sym} at {ltp}")
            
            # 🟢 1. Pop from memory to prevent duplicate firing
            trade_backup = ACTIVE_OPTION_TRADES.pop(sym, None)
            
            # 🟢 2. Direct DB execution (No HTTP requests, no local network failures)
            def process_direct_exit(s, tid, r, e_price, backup):
                try:
                    # Fetch trade directly
                    db_trade = supabase.table("paper_trades").select("*").eq("id", tid).execute()
                    if not db_trade.data:
                        return
                        
                    trade_data = db_trade.data[0]
                    if trade_data.get("status") == "CLOSED":
                        return
                        
                    entry_price = trade_data["entry_price"]
                    quantity = trade_data["quantity"]
                    side = trade_data["side"]
                    
                    # Calculate PnL & Refund
                    pnl = (e_price - entry_price) * quantity if side == "BUY" else (entry_price - e_price) * quantity
                    blocked_margin = trade_data.get("margin_blocked") or (entry_price * quantity)
                    refund_amount = blocked_margin + pnl
                    
                    # Fetch Nifty Spot at Exit
                    try:
                        nifty_quote = kite.quote("NSE:NIFTY 50")
                        nifty_spot_at_exit = nifty_quote["NSE:NIFTY 50"]["last_price"]
                    except Exception:
                        nifty_spot_at_exit = None

                    # Update Balance
                    bal_res = supabase.table("kite_config").select("value").eq("key_name", "paper_balance").execute()
                    current_balance = float(bal_res.data[0]["value"]) if bal_res.data else 100000.0
                    new_balance = round(current_balance + refund_amount, 2)
                    supabase.table("kite_config").update({"value": str(new_balance)}).eq("key_name", "paper_balance").execute()
                    
                    # Close Trade in DB
                    supabase.table("paper_trades").update({
                        "status": "CLOSED",
                        "exit_price": float(e_price),
                        "pnl": float(pnl),
                        "nifty_spot_at_exit": nifty_spot_at_exit,
                        "updated_at": datetime.now().isoformat(),
                        "exit_reasoning": r
                    }).eq("id", tid).execute()
                    
                    logger.info(f"✅ SYSTEM EXIT COMPLETE: {s} squared off perfectly via direct DB sync.")
                except Exception as e:
                    logger.error(f"❌ Critical DB Error during auto-exit for {s}: {e}")
                    # 🟢 3. IF IT FAILS, PUT IT BACK IN MEMORY TO RETRY INSTANTLY
                    if backup: ACTIVE_OPTION_TRADES[s] = backup
            
            threading.Thread(target=process_direct_exit, args=(sym, trade_id, reason, ltp, trade_backup), daemon=True).start()

    asyncio.run_coroutine_threadsafe(ws_manager.broadcast({"type": "live_options", "data": formatted_ticks}), fastapi_loop)

def exit_option_trade(symbol, exit_price, reason):
    if symbol in ACTIVE_OPTION_TRADES:
        logger.info(f"🚩 EXITING OPTION: {symbol} at {exit_price} - {reason}")
        del ACTIVE_OPTION_TRADES[symbol]

def on_connect(ws, response):
    logger.info("🟢 KiteTicker Connected.")
    tokens = list(token_to_symbol.keys())
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)

def on_close(ws, code, reason):
    logger.warning(f"🔴 KiteTicker closed: {code} - {reason}")

def start_kite_ticker():
    def run_loop():
        global ticker, token_to_symbol
        while True:
            try:
                # 🟢 ZOMBIE KILLER: Force stop the previous ticker before starting a new one
                global ticker
                if ticker is not None:
                    logger.info("🛑 Stopping existing ticker instance...")
                    try:
                        ticker.stop()
                    except: pass
                    ticker = None

                logger.info("🔄 Fetching fresh token from Supabase...")
                auth_token = get_active_token()
                if not auth_token:
                    time.sleep(10)
                    continue
                
                # Apply the token to the global REST client before calling functions that use it
                kite.set_access_token(auth_token)
                logger.info(f"✅ Session Synced. Token starts with: {auth_token[:5]}...")
                
                token_to_symbol, tokens_to_sub = get_nifty_weekly_options()
                from utils.common import KITE_API_KEY
                
                ticker = KiteTicker(KITE_API_KEY, auth_token)
                ticker.on_connect = on_connect
                ticker.on_ticks = on_ticks
                # Log the actual reason so the "Ticker Loop Error" is not empty
                ticker.on_close = lambda ws, code, reason: (logger.warning(f"Ticker Closed: {reason}"), ticker.stop())
                ticker.on_error = lambda ws, code, reason: (logger.error(f"Ticker Error: {reason}"), ticker.stop())
                
                logger.info("🔌 Connecting Kite Ticker...")
                ticker.connect(threaded=True)
                
                time.sleep(5)
                
                # Monitor loop: Detect if it drops so we can fetch a fresh token
                while True:
                    if not ticker.is_connected():
                        logger.warning("⚠️ Ticker disconnected. Restarting loop...")
                        break
                    time.sleep(5)
            except Exception as e:
                # 🟢 FIX: Force print the type of error to see why it's silent
                logger.error(f"❌ Ticker Loop Error ({type(e).__name__}): {repr(e)}")
                time.sleep(10)
    
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    # threading.Thread(target=start_internal_ui_test, daemon=True).start()

# 🧪 TEMPORARY TEST LOOP (Remove after testing)
def run_ui_test():
    time.sleep(5) 
    base = 25200.0
    while True:
        mock_tick = [{
            'instrument_token': 256265,
            'last_price': round(base + (datetime.now().second % 10), 2),
            'volume_traded': 5000,
            'oi': 120000
        }]
        on_ticks(None, mock_tick)
        time.sleep(1)

def start_internal_ui_test():
    global token_to_symbol
    logger.info("🧪 Internal Mock Test Starting (Spot +/- 500 Fluctuations)...")
    
    mock_opens = {}
    base_nifty = 25000.0
    current_spot = base_nifty

    while True:
        if not token_to_symbol:
            token_to_symbol = {256265: {"symbol": "NIFTY 50", "strike": "SPOT", "type": "SPOT"}}
            for i in range(-15, 16):
                strike = int(base_nifty + (i * 100))
                token_to_symbol[100000 + strike] = {"symbol": f"NIFTY_MOCK_{strike}_CE", "strike": strike, "type": "CE"}
                token_to_symbol[200000 + strike] = {"symbol": f"NIFTY_MOCK_{strike}_PE", "strike": strike, "type": "PE"}

        fake_tick_packet = []
        
        # 🟢 RANDOM WALK: Replaces the wave with realistic wandering
        # It has a slight pull back to base_nifty to keep it within a testable range
        pull_factor = (base_nifty - current_spot) * 0.01 
        drift = random.uniform(-2.5, 2.5) + pull_factor
        current_spot = round(current_spot + drift, 2)
        
        for token, info in token_to_symbol.items():
            # 1. Handle Spot Index
            if info["type"] == "SPOT":
                price = current_spot
                open_price = 25000.0
            
            # 2. Handle Options (Correlated with changing spot price)
            else:
                strike = info["strike"]
                # 🟢 GREEK ENGINE: Delta + Theta + Vega
                intrinsic = max(0, current_spot - strike) if info["type"] == "CE" else max(0, strike - current_spot)
                
                # Simulate Time Decay (Theta) over a 5-minute session
                session_elapsed = (time.time() % 300) / 300 
                time_decay = 40 * (1 - session_elapsed) 
                
                # Add Volatility (Vega)
                volatility_premium = 20 + random.uniform(-1, 1)
                theoretical_price = round(intrinsic + time_decay + volatility_premium, 2)
                
                # 🟢 BID-ASK SPREAD: Simulates 0.10 to 0.40 flickering
                spread = random.uniform(0.10, 0.40)
                price = round(random.choice([theoretical_price - (spread/2), theoretical_price + (spread/2)]), 2)
                
                # Store a persistent 'Open' price for this session to get proper % PnL
                if token not in mock_opens:
                    mock_opens[token] = price + 5.5
                open_price = mock_opens[token]

                last_mock_prices[info["symbol"]] = price

            # 🟢 PnL %: Strictly calculates based on Trade Entry Price if active
            symbol = info["symbol"]
            if symbol in ACTIVE_OPTION_TRADES:
                entry_price = ACTIVE_OPTION_TRADES[symbol].get("entry", price)
                # Formula: ((Current - Entry) / Entry) * 100
                change_percent = round(((price - entry_price) / entry_price) * 100, 2)
            else:
                # Fallback to Day Change % using session open_price
                change_percent = round(((price - open_price) / open_price) * 100, 2) if open_price > 0 else 0

            fake_tick_packet.append({
                'instrument_token': token,
                'last_price': price,
                'volume_traded': 1500 + (token % 500),
                'oi': 10000 + (token % 1000),
                'ohlc': {'open': open_price}, 
                'change_percent': change_percent 
            })
        
        # Fire mock data into the real WebSocket broadcaster
        on_ticks(None, fake_tick_packet)
        time.sleep(2)

strategy_history_cache = {"date": None, "hits": []}

def subscribe_to_new_trade(symbol: str):
    """Dynamically forces the Kite WebSocket to subscribe to a new trade."""
    global ticker, token_to_symbol, NFO_INSTRUMENTS_CACHE
    try:
        if NFO_INSTRUMENTS_CACHE is None or date.today() != getattr(NFO_INSTRUMENTS_CACHE, '_cache_date', None):
            NFO_INSTRUMENTS_CACHE = pd.DataFrame(kite.instruments("NFO"))
            NFO_INSTRUMENTS_CACHE._cache_date = date.today()
            
        row = NFO_INSTRUMENTS_CACHE[NFO_INSTRUMENTS_CACHE['tradingsymbol'] == symbol]
        if not row.empty:
            token = int(row.iloc[0]['instrument_token'])
            
            # Add to mapping so on_ticks can process it
            if token not in token_to_symbol:
                strike = int(row.iloc[0]['strike'])
                opt_type = row.iloc[0]['instrument_type']
                token_to_symbol[token] = {"symbol": symbol, "strike": strike, "type": opt_type}
                
                # Tell Kite to start streaming it NOW
                if ticker and ticker.is_connected():
                    ticker.subscribe([token])
                    ticker.set_mode(ticker.MODE_FULL, [token])
                    logger.info(f"📡 Dynamically subscribed to live data for new trade: {symbol}")
        else:
            logger.warning(f"Could not find token for {symbol} to subscribe.")
    except Exception as e:
        logger.error(f"Failed to dynamically subscribe to {symbol}: {e}")

async def global_strategy_monitor():
    """Background task to globally monitor Nifty strategy and broadcast live status."""
    global ws_manager, kite
    last_alert_time = 0

    while True:
        try:
            now = get_ist_time()
            if now.hour > 15 or (now.hour == 15 and now.minute >= 30) or now.hour < 9 or (now.hour == 9 and now.minute < 15):
                await asyncio.sleep(300)
                continue

            # 1. Reset history on a new day
            today_str = now.strftime("%Y-%m-%d")
            if strategy_history_cache["date"] != today_str:
                strategy_history_cache["date"] = today_str
                strategy_history_cache["hits"] = []

            # 2. Check individual candle statuses
            is_5m  = is_candle_green(5)
            is_15m = is_candle_green(15)
            is_1h  = is_candle_green(60)
            
            all_green = is_5m and is_15m and is_1h
            current_time_str = now.strftime("%H:%M:%S")

            # 3. Build the live status payload
            payload = {
                "type": "STRATEGY_STATUS",
                "status": {"5m": is_5m, "15m": is_15m, "1h": is_1h},
                "is_met": all_green,
                "history": strategy_history_cache["hits"]
            }
            
            if all_green:
                current_ts = time.time()
                # 4. Only record history and trigger the top banner if 5 mins have passed since the last hit
                if current_ts - last_alert_time > 300:
                    strategy_history_cache["hits"].append(current_time_str)
                    last_alert_time = current_ts
                    
                    try:
                        nifty_quote = kite.quote("NSE:NIFTY 50")
                        nifty_spot = nifty_quote["NSE:NIFTY 50"]["last_price"]
                        itm_strike = round((nifty_spot - 150) / 50) * 50
                        
                        payload["alert"] = {
                            "message": f"🟢 3 Green Candles Met! Recommended: Buy NIFTY {itm_strike} CE",
                            "timestamp": current_time_str
                        }
                    except Exception as e:
                        logger.error(f"Failed to fetch Nifty quote for alert: {e}")
            
            # 5. Attach updated history and broadcast
            payload["history"] = strategy_history_cache["hits"]
            await ws_manager.broadcast(payload)
            
            await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"Global strategy monitor error: {e}")
            await asyncio.sleep(60)