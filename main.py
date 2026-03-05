import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
import httpx
from pydantic import BaseModel
from live_core_functions.build_monitor_list import create_monitor_list
from live_core_functions.minute_monitor_stocks import start_finding_breakouts
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import gspread
import base64
import json
from utils.common import supabase, logger, get_active_token, kite
from datetime import date, datetime
from utils.common import kite
from fastapi import WebSocket, WebSocketDisconnect
from utils.options_streamer import ws_manager, start_kite_ticker
from utils.options_streamer import ACTIVE_OPTION_TRADES
import asyncio

def get_db_balance():
    """Consolidated helper to fetch paper balance."""
    try:
        res = supabase.table("kite_config").select("value").eq("key_name", "paper_balance").execute()
        if res.data and len(res.data) > 0:
            return float(res.data[0]["value"])
        return 100000.0
    except Exception as e:
        logger.error(f"❌ Error reading balance from Supabase: {e}")
        return 100000.0

creds_b64 = os.getenv("GOOGLE_SHEET_CREDS_B64")

if not creds_b64:
    raise Exception("GOOGLE_SHEET_CREDS_B64 not set")

decoded = base64.b64decode(creds_b64)
info = json.loads(decoded)

gc = gspread.service_account_from_dict(info)
sh = gc.open("Live_Paper_Trading")
sheet = sh.sheet1

app = FastAPI(title="Trademate")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "https://melodious-wisp-e4651e.netlify.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PayloadRequest(BaseModel):
    token: str

@app.get("/api/get-paper-balance")
async def get_paper_balance():
    return {"balance": get_db_balance()}
    
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting FastAPI Server & Syncing Session...")
    
    # 1. Sync Token (from your previous fix)
    from utils.common import get_active_token, kite
    token = get_active_token()
    if token:
        kite.set_access_token(token)
    
    # 2. Start the Options Ticker (The "Ears")
    start_kite_ticker()
    
    # This fires and forgets the monitor into the background immediately
    # asyncio.create_task(asyncio.to_thread(start_finding_breakouts))
    # logger.info("📡 Stock Breakout Monitor started in background.")

@app.post("/api/place-option-order")
async def place_option_order(data: dict):
    symbol = data.get("symbol")
    price = data.get("price")
    side = data.get("side", "BUY")

    # 🟢 DUPLICATE PROTECTION: Check if trade already exists
    existing_check = supabase.table("paper_trades") \
        .select("id") \
        .eq("symbol", symbol) \
        .eq("status", "OPEN") \
        .execute()

    if existing_check.data and len(existing_check.data) > 0:
        logger.warning(f"⚠️ Duplicate Blocked: {symbol} already has an open position.")
        return {"status": "ignored", "message": "Position already open"}
    
    # 🟢 1. FETCH & VERIFY FUNDS (Refactored)
    current_balance = get_db_balance()
    qty = data.get("quantity", 50)
    margin_required = price * qty
    
    if margin_required > current_balance:
        logger.warning(f"❌ Insufficient Funds: Need {margin_required}, have {current_balance}")
        return {"status": "error", "message": "Don't have enough funds, please add funds."}
    
    # 🟢 FIXED: Default to 50 if Kite fails so balance still updates
    qty = 50
    try:
        instruments = kite.instruments("NFO")
        instrument_info = next((i for i in instruments if i['tradingsymbol'] == symbol), None)
        if instrument_info:
            qty = instrument_info['lot_size']
    except Exception as e:
        logger.warning(f"⚠️ Using default lot size 50: {e}")

    # 🟢 2. UNIFIED RISK-REWARD & MARGIN
    sl = round(price * 0.99, 2) if side == "BUY" else round(price * 1.01, 2)
    target = round(price * 1.02, 2) if side == "BUY" else round(price * 0.98, 2)
    margin_required = price * qty

    # 🟢 3. ATOMIC EXECUTION (Memory -> DB -> Balance)
    ACTIVE_OPTION_TRADES[symbol] = {"entry": price, "qty": qty, "type": side, "sl": sl, "target": target}
    
    trade_record = {
        "symbol": symbol, "entry_price": price, "quantity": qty,
        "side": side, "status": "OPEN", "sl_price": sl, "target_price": target,
        "created_at": datetime.now().isoformat()
    }
    
    # Run DB insertions
    supabase.table("paper_trades").insert(trade_record).execute()
    
    # Unified Balance Update
    new_balance = round(current_balance - margin_required, 2)
    update_res = supabase.table("kite_config").update({"value": str(new_balance)}).eq("key_name", "paper_balance").execute()
    
    if not update_res.data:
        logger.error("❌ Failed to update balance in Supabase. Ensure row 'paper_balance' exists.")
    
    return {"status": "success", "verified_quantity": qty, "new_balance": new_balance}

@app.get("/api/get-active-trades")
async def get_active_trades():
    """Provides the UI with active trades from Supabase."""
    try:
        response = supabase.table("paper_trades").select("*").eq("status", "OPEN").execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        return []

@app.post("/api/exit-option-trade")
async def exit_option_trade(data: dict):
    symbol = data.get("symbol")
    trade_id = data.get("trade_id")
    
    if symbol not in ACTIVE_OPTION_TRADES:
        return {"status": "error", "message": "Trade not found in active memory."}

    # 1. Get current trade details
    trade = ACTIVE_OPTION_TRADES[symbol]
    entry_price = trade["entry"]
    quantity = trade.get("qty", 50)
    
    # 2. Get the current Price (Exit Price)
    # Checks Mock Cache first, then asks Kite for Live price
    from utils.options_streamer import last_mock_prices
    exit_price = last_mock_prices.get(symbol)
    
    if not exit_price:
        try:
            q = kite.quote(f"NFO:{symbol}")
            exit_price = q.get(f"NFO:{symbol}", {}).get("last_price", entry_price)
        except:
            exit_price = entry_price

    # 3. Calculate PnL & Refund
    pnl = (exit_price - entry_price) * quantity if trade["type"] == "BUY" else (entry_price - exit_price) * quantity
    refund_amount = (entry_price * quantity) + pnl

    # 4. Update Database
    try:
        current_balance = get_db_balance()
        new_balance = round(current_balance + refund_amount, 2)
        supabase.table("kite_config").update({"value": str(new_balance)}).eq("key_name", "paper_balance").execute()
        supabase.table("paper_trades").update({"status": "CLOSED"}).eq("id", trade_id).execute()
        
        # 5. Clear Memory
        del ACTIVE_OPTION_TRADES[symbol]
        return {"status": "success", "new_balance": new_balance, "pnl": pnl}
    except Exception as e:
        logger.error(f"❌ Exit Error: {e}")
        return {"status": "error", "message": "Database sync failed during exit."}

@app.api_route("/start-finding-breakouts", methods=["GET", "POST"])
async def handle_find_breakouts(background_tasks: BackgroundTasks):
    """
    Triggers the build and monitor flow in the background to prevent 
    HTTP timeouts and handle memory spikes safely.
    """
    def full_algo_flow():
        try:
            today_str = date.today().strftime("%Y-%m-%d")

            # 🔄 RE-FETCH AND FORCE SET TOKEN
            token_response = supabase.table("kite_config") \
                .select("value") \
                .eq("key_name", "access_token") \
                .execute()

            if token_response.data and len(token_response.data) > 0:
                new_token = token_response.data[0]["value"]
                kite.set_access_token(new_token)
                logger.info(f"✅ Token synced from DB (starts with {new_token[:5]}...)")
            else:
                logger.error("❌ CRITICAL: No access_token found in kite_config table.")
                return # Stop the flow if no token exists

            # 1. Check/Build monitor list inside the background task
            existing = supabase.table("monitor_list") \
                .select("symbol", count="exact") \
                .eq("date", today_str) \
                .limit(1) \
                .execute()

            if not existing.count:
                print(f"🔄 {today_str}: No monitor list found. Building now...")
                create_monitor_list()
            else:
                print(f"✅ {today_str}: Monitor list already exists. Skipping build.")

            # 2. Start monitoring immediately after build completes
            start_finding_breakouts()
            
        except Exception as e:
            print(f"❌ Background Algo Flow Error: {e}")

    # Add the combined task to background
    background_tasks.add_task(full_algo_flow)

    return {
        "status": "success", 
        "message": "Monitor list build and breakout monitoring started in background."
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return PlainTextResponse("OK", status_code=200)

@app.get("/api/dashboard")
def get_dashboard(date: str):
    requested_date = date  # e.g. "2026-02-23"

    # === 1. If today → always return LIVE data ===
    today_str = datetime.today().strftime("%Y-%m-%d")
    if requested_date == today_str:
        return build_dashboard_data(requested_date)   # ← we'll create this helper below

    # === 2. For past dates → return frozen snapshot ===
    try:
        snapshot = supabase.table("dashboard_snapshots") \
            .select("data") \
            .eq("date", requested_date) \
            .limit(1) \
            .execute()

        if snapshot.data:
            return snapshot.data[0]["data"]   # Return exactly what was saved at 3:30 PM
        else:
            # No snapshot yet (very old date) → return empty or live as fallback
            return build_dashboard_data(requested_date)
    except:
        return build_dashboard_data(requested_date)
    
def get_count(date_str: str, tier=None):
    query = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str)
    if tier: 
        query = query.eq("monitoring_tier", tier)
    return query.execute().count or 0

def build_dashboard_data(date_str: str):
    """All the existing dashboard logic moved here so we can reuse it."""
    result = {}

    # 🟢 FIXED: Correctly forwarding date_str to every count request
    result["monitor_list_count"] = get_count(date_str)
    result["slow_tier_count"] = get_count(date_str, "slow")
    result["fast_tier_count"] = get_count(date_str, "fast")

    # FOR MOCK DATA TESTING
    # result["monitor_list_count"] = get_count(date_str)
    # result["slow_tier_count"] = get_count(date_str, "slow")
    # result["fast_tier_count"] = get_count(date_str, "fast")
        
    # Fast tier count + symbols
    try:
        r = supabase.table("monitor_list") \
            .select("symbol,has_derivative, max_ma, tick_size", count="exact") \
            .eq("date", date_str) \
            .eq("monitoring_tier", "fast") \
            .execute()

        result["fast_tier_count"] = r.count

        fast_symbols_enriched = []
        for row in (r.data or []):
            max_ma = row.get("max_ma")
            tick_size = row.get("tick_size") or 0.05
            breakout_price = None
            if max_ma:
                from decimal import Decimal, ROUND_CEILING
                dv = Decimal(str(max_ma))
                dt = Decimal(str(tick_size))
                n = (dv / dt).to_integral_value(rounding=ROUND_CEILING)
                candidate = n * dt
                if candidate <= dv:
                    candidate = (n + 1) * dt
                breakout_price = float(candidate)
            
            fast_symbols_enriched.append({
                "symbol": row["symbol"],
                "breakout_price": round(breakout_price, 2) if breakout_price else None,
                "current_price": None,
                "has_derivative": bool(row.get("has_derivative", False))
            })

        result["fast_tier_symbols"] = fast_symbols_enriched

        if fast_symbols_enriched:
            try:
                symbols_list = [s["symbol"] for s in fast_symbols_enriched]
                from utils.common import kite
                all_quotes = {}
                for i in range(0, len(symbols_list), 50):
                    batch = [f"NSE:{s}" for s in symbols_list[i:i+50]]
                    quotes = kite.quote(batch)
                    all_quotes.update(quotes)
                
                for item in result["fast_tier_symbols"]:
                    sym = item["symbol"]
                    q = all_quotes.get(f"NSE:{sym}")
                    if q:
                        item["current_price"] = q.get("last_price")
            except Exception as e:
                pass
    except:
        result["fast_tier_count"] = 0
        result["fast_tier_symbols"] = []

    # 4. Breakout details with Live Price fetching
    try:
        r = supabase.table("live_breakouts") \
            .select("symbol, breakout_price, breakout_time, percent_move, high_price, exit_reason") \
            .eq("breakout_date", date_str) \
            .execute()

        breakouts_data = r.data or []
        
        # --- NEW: Fetch live prices for these breakouts ---
        if breakouts_data:
            breakout_symbols = [f"NSE:{b['symbol']}" for b in breakouts_data]
            try:
                # Fetch quotes from Kite for all breakout stocks
                live_quotes = kite.quote(breakout_symbols)
                
                for b in breakouts_data:
                    sym_key = f"NSE:{b['symbol']}"
                    if sym_key in live_quotes:
                        # Add current_price to the object sent to frontend
                        b["current_price"] = live_quotes[sym_key].get("last_price")
                    else:
                        b["current_price"] = b["breakout_price"] # Fallback
            except Exception as e:
                print(f"⚠️ Failed to fetch live prices for breakouts: {e}")

        result["breakouts"] = breakouts_data
        for b in result["breakouts"]:
            exit_reason = b.get("exit_reason") or ""
            if "Target" in exit_reason:
                b["status"] = "Target Hit"
            elif "SL" in exit_reason:
                b["status"] = "SL Hit"
            elif "Stagnant" in exit_reason:
                b["status"] = "Circuit/Stagnant"
            elif "EOD" in exit_reason:
                b["status"] = "EOD Exit"
            else:
                b["status"] = "In Play"
        result["breakout_count"] = len(result["breakouts"])

    except Exception as e:
        print(f"❌ Error building breakout dashboard: {e}")
        result["breakouts"] = []
        result["breakout_count"] = 0

    # 5. Exit conditions + PnL from Google Sheet
    sl_hit = 0
    target_hit = 0
    eod_exit = 0
    pnl_list = []

    if sheet:
        try:
            all_rows = sheet.get_all_values()
            for row in all_rows[1:]:
                if len(row) < 14:
                    continue
                row_date = row[0][:10] if row[0] else ""
                if row_date != date_str:
                    continue
                exit_reason = row[12].strip() if len(row) > 12 else ""
                if "SL" in exit_reason:
                    sl_hit += 1
                elif "Target" in exit_reason:
                    target_hit += 1
                elif "EOD" in exit_reason:
                    eod_exit += 1
                pnl_str = row[13].replace("%", "").strip() if len(row) > 13 else "0"
                try:
                    pnl_list.append({
                        "symbol": row[1], 
                        "pnl": float(pnl_str),
                        "percent_move": float(pnl_str)
                    })
                except:
                    pass
        except:
            pass

    result["exit_conditions"] = {
        "sl_hit": sl_hit,
        "target_hit": target_hit,
        "eod_exit": eod_exit
    }

    overall_pnl = round(sum(p["pnl"] for p in pnl_list) / len(pnl_list), 2) if pnl_list else 0
    result["pnl"] = {
        "overall": overall_pnl,
        "stocks": pnl_list
    }

    return result

@app.websocket("/ws/options")
async def websocket_options_endpoint(websocket: WebSocket):
    """Endpoint for the React frontend to connect and receive live option prices."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for ping/messages from frontend if any
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.get("/api/get-option-chain-snapshot")
async def get_option_chain_snapshot():
    res = supabase.table("market_snapshots").select("data").order("created_at", desc=True).limit(1).execute()
    return res.data[0]["data"] if res.data else []

@app.get("/api/get-live-option-chain")
async def get_live_option_chain():
    """Fetches real-time LTP from Kite even if the market is closed."""
    from utils.options_streamer import get_nifty_weekly_options
    try:
        # 1. Get the tokens and mapping of what to fetch
        token_map, _ = get_nifty_weekly_options()
        
        # 2. Build the list of Kite symbols (NSE:NIFTY 50 or NFO:SYMBOL)
        symbols = [f"NFO:{v['symbol']}" if v['type'] != 'SPOT' else "NSE:NIFTY 50" for v in token_map.values()]
        
        # 3. Fetch current quotes from Zerodha
        quotes = kite.quote(symbols)
        
        live_data = []
        for info in token_map.values():
            q_key = f"NFO:{info['symbol']}" if info['type'] != 'SPOT' else "NSE:NIFTY 50"
            q = quotes.get(q_key, {})
            
            ltp = q.get('last_price', 0)
            ohlc = q.get('ohlc', {})
            open_price = ohlc.get('open', ltp)
            
            live_data.append({
                "symbol": info["symbol"],
                "strike": info["strike"],
                "type": info["type"],
                "ltp": ltp,
                "open": open_price,
                "change_percent": round(((ltp - open_price) / open_price * 100), 2) if open_price > 0 else 0
            })
        return live_data
    except Exception as e:
        logger.error(f"❌ Kite Fetch Error: {e}")
        return []

@app.post("/api/exit-option-trade")
async def exit_option_trade(data: dict):
    symbol = data.get("symbol")
    trade_id = data.get("trade_id")
    
    if symbol not in ACTIVE_OPTION_TRADES:
        return {"status": "error", "message": "Trade not found"}

    # 1. Get current trade details
    trade = ACTIVE_OPTION_TRADES[symbol]
    entry_price = trade["entry"]
    quantity = 50  # Matches our mock default
    
    # 2. Get the current LTP (Exit Price)
    # In mock mode, we pull the latest price from our streamer
    from utils.options_streamer import last_mock_prices
    exit_price = last_mock_prices.get(symbol, entry_price)

    # 3. Calculate PnL
    # If BUY: (Exit - Entry) * Qty | If SELL: (Entry - Exit) * Qty
    pnl = (exit_price - entry_price) * quantity if trade["side"] == "BUY" else (entry_price - exit_price) * quantity
    
    # 4. Calculate Refund (Original Margin + PnL)
    original_margin = entry_price * quantity
    refund_amount = original_margin + pnl

    # 5. Update Database
    current_balance = get_db_balance()
    new_balance = round(current_balance + refund_amount, 2)
    supabase.table("kite_config").update({"value": str(new_balance)}).eq("key_name", "paper_balance").execute()

    # 6. Remove from active trades
    del ACTIVE_OPTION_TRADES[symbol]
    
    return {"status": "success", "new_balance": new_balance, "pnl": pnl}