import pandas as pd
import threading
import asyncio
import os
import time
from datetime import datetime, timedelta, date
from fastapi import WebSocket
from kiteconnect import KiteTicker
from utils.common import kite, logger, get_active_token

# 1. Single source of truth for trades
ACTIVE_OPTION_TRADES = {}
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
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

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
        if NFO_INSTRUMENTS_CACHE is None:
            NFO_INSTRUMENTS_CACHE = pd.DataFrame(kite.instruments("NFO"))
        
        df = NFO_INSTRUMENTS_CACHE
        df_nifty = df[(df["name"] == "NIFTY") & (df["segment"] == "NFO-OPT")].copy()
        df_nifty["expiry"] = pd.to_datetime(df_nifty["expiry"]).dt.date
        
        today_val = date.today()
        future_expiries = df_nifty[df_nifty["expiry"] >= today_val]["expiry"].unique()
        if len(future_expiries) == 0: return {}, []
            
        current_expiry = min(future_expiries)
        df_curr = df_nifty[df_nifty["expiry"] == current_expiry]

        strikes = [atm_strike + (i * 100) for i in range(-20, 21)]
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

        return token_map, options_tokens
    except Exception as e:
        logger.error(f"❌ Error fetching Nifty options: {e}")
        return {}, []

def is_candle_green(interval_minutes):
    try:
        nifty_token = 256265
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=2)
        
        interval_map = {5: "5minute", 15: "15minute", 60: "60minute"}
        data = kite.historical_data(nifty_token, from_dt, to_dt, interval_map[interval_minutes])
        
        if not data or len(data) < 2: return False
        
        # Use index -2 to get the last COMPLETED candle
        completed_candle = data[-2]
        return completed_candle['close'] > completed_candle['open']
    except Exception as e:
        logger.error(f"Error checking {interval_minutes}min candle: {e}")
        return False

def get_5_percent_otm_ce():
    global NFO_INSTRUMENTS_CACHE
    try:
        nifty_quote = kite.quote("NSE:NIFTY 50")
        spot_price = nifty_quote["NSE:NIFTY 50"]["last_price"]
        target_strike = round((spot_price * 1.05) / 100) * 100
        
        if NFO_INSTRUMENTS_CACHE is None:
            NFO_INSTRUMENTS_CACHE = pd.DataFrame(kite.instruments("NFO"))
            
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
    # 🟢 Add this at the start of your loop to calculate % moves
    for tick in ticks:
        token = tick['instrument_token']
        info = token_to_symbol.get(token, {})
        ltp = tick.get('last_price', 0)
        
        # Zerodha provides 'ohlc' data in 'MODE_FULL'
        ohlc = tick.get('ohlc', {})
        open_price = ohlc.get('open', ltp) # Fallback to LTP if market just opened
        
        # Calculate % Move
        change_percent = 0
        if open_price > 0:
            change_percent = round(((ltp - open_price) / open_price) * 100, 2)

        formatted_ticks.append({
            "symbol": info.get("symbol"),
            "strike": info.get("strike", 0),
            "type": info.get("type", "UNKNOWN"),
            "ltp": ltp,
            "open": open_price,
            "change_percent": change_percent, # 🟢 NEW
            "volume": tick.get('volume_traded', 0),
            "oi": tick.get('oi', 0)
        })

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
    global ticker, token_to_symbol, fastapi_loop
    fastapi_loop = asyncio.get_event_loop()
    auth_token = get_active_token()
    if not auth_token: return

    token_to_symbol, tokens_to_sub = get_nifty_weekly_options()
    if not tokens_to_sub: return

    api_key = os.getenv("KITE_API_KEY")
    ticker = KiteTicker(api_key, auth_token)
    ticker.on_connect = on_connect
    ticker.on_ticks = on_ticks
    ticker.on_close = on_close
    t = threading.Thread(target=ticker.connect, kwargs={"threaded": True})
    t.daemon = True
    t.start()

    threading.Thread(target=start_internal_ui_test, daemon=True).start()

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
    logger.info("🧪 Internal Mock Test Starting...")
    
    # Static 'Open' prices to keep % calculations consistent
    mock_opens = {}
    base_nifty = 25200.0

    while True:
        if not token_to_symbol:
            time.sleep(1)
            continue
            
        fake_tick_packet = []
        for token, info in token_to_symbol.items():
            # 1. Handle Spot Index
            if info["type"] == "SPOT":
                price = round(base_nifty + (datetime.now().second % 15), 2)
                open_price = 25178.65 # Matching your Zerodha screenshot
            
            # 2. Handle Options
            else:
                strike_diff = abs(25200 - info["strike"]) if isinstance(info["strike"], int) else 500
                # Realistic option pricing logic
                price = round(max(5, 10 + (datetime.now().second % 20)), 2)
                
                # Store a persistent 'Open' price for this session
                if token not in mock_opens:
                    mock_opens[token] = price + 5.5 # Simulate a gap-down start
                open_price = mock_opens[token]

            # 3. Calculate % Change
            change_percent = round(((price - open_price) / open_price) * 100, 2)

            fake_tick_packet.append({
                'instrument_token': token,
                'last_price': price,
                'volume_traded': 1500 + (token % 500),
                'oi': 10000 + (token % 1000),
                'ohlc': {'open': open_price}, # 🟢 Production field
                'change_percent': change_percent # 🟢 Production field
            })
        
        on_ticks(None, fake_tick_packet)
        time.sleep(1)