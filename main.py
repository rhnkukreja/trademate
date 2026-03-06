import os
import pandas as pd
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
from utils.common import supabase, logger, get_active_token, kite, get_ist_time
from datetime import date, datetime
from fastapi import WebSocket, WebSocketDisconnect
from utils.options_streamer import ws_manager, start_kite_ticker, is_candle_green
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_dashboard_data(requested_date: str):
    """Builds comprehensive dashboard data for a given date (optimized for breakout page)."""
    date_str = requested_date  # ISO from frontend (e.g., "2026-03-05")
    logger.info(f"[Dashboard] Building data for date: {date_str} (IST today: {get_ist_time().strftime('%Y-%m-%d')})")
    
    result = {
        "date": date_str,
        "breakouts": [],  # For "Breakout Price & Time" section
        "pnl": {"overall": 0, "stocks": []},
        "exit_conditions": {"sl_hit": 0, "target_hit": 0, "eod_exit": 0},
        "fast_tier_count": 0,
        "fast_tier_symbols": [],  # Key for "Fast Tier" UI
        "monitor_count": 0  # Key for "Monitor List" UI
    }
    
    def get_count(date_str: str, tier=None):
        try:
            if tier:
                exact_r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str).eq("monitoring_tier", tier).execute()
                count = exact_r.count or 0
                logger.info(f"[get_count] Exact '{tier}' on {date_str}: {count}")
                if count == 0:
                    fallback_r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str).ilike("monitoring_tier", f"%{tier}%").execute()
                    count = fallback_r.count or 0
                    logger.info(f"[get_count] Fallback '%{tier}%' on {date_str}: {count}")
            else:
                r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str).execute()
                count = r.count or 0
                logger.info(f"[get_count] Total on {date_str}: {count} (data_len: {len(r.data or [])})")
            return count
        except Exception as e:
            logger.error(f"[get_count] Error: {e}")
            return 0
    
    # 1. FAST TIER: Query, Enrich, Log (matches your logs: 8 stocks)
    try:
        tier_query = supabase.table("monitor_list").select("*").eq("date", date_str).eq("monitoring_tier", "fast")
        r = tier_query.execute()
        result["fast_tier_count"] = r.count
        logger.info(f"[Dashboard] Fast tier raw: count={r.count}, data_len={len(r.data or [])}")
        
        fast_symbols_enriched = []
        for row in (r.data or []):
            candidate = row.get("breakout_price_candidate") or row.get("breakout_price") or 0
            breakout_price = round(float(candidate), 2) if candidate else None
            fast_symbols_enriched.append({
                "symbol": row["symbol"],  # e.g., 'CARERATING'
                "breakout_price": breakout_price,
                "current_price": None,  # Filled live below
                "has_derivative": bool(row.get("has_derivative", False))
            })
        result["fast_tier_symbols"] = fast_symbols_enriched
        logger.info(f"[Dashboard] Enriched fast symbols: {len(fast_symbols_enriched)} (e.g., {fast_symbols_enriched[0]['symbol'] if fast_symbols_enriched else 'none'})")
    except Exception as e:
        logger.error(f"[Dashboard] Fast tier failed: {e}")
        result["fast_tier_count"] = 0
        result["fast_tier_symbols"] = []
    
    # 2. MONITOR COUNT: Total from DB (matches logs: 1152+)
    result["monitor_count"] = get_count(date_str)
    logger.info(f"[Dashboard] Final monitor_count: {result['monitor_count']}")
    
    # 3. LIVE PRICES: Fetch current for fast symbols (Kite, non-blocking)
    try:
        if result["fast_tier_symbols"]:
            nse_symbols = [f"NSE:{s['symbol']}" for s in result["fast_tier_symbols"]]
            quotes = kite.quote(nse_symbols[:50])  # Batch limit
            for s in result["fast_tier_symbols"]:
                q_key = f"NSE:{s['symbol']}"
                q = quotes.get(q_key, {})
                s["current_price"] = round(q.get('last_price', s['breakout_price'] or 0), 2)
            logger.info(f"[Dashboard] Updated {len(result['fast_tier_symbols'])} live prices")
    except Exception as e:
        logger.error(f"[Dashboard] Live prices failed: {e}")
    
    # 4. BREAKOUTS: From live_breakouts table (empty per logs, but ready)
    try:
        breakouts_r = supabase.table("live_breakouts").select("*").eq("date", date_str).order("breakout_time").execute()
        result["breakouts"] = breakouts_r.data or []
        logger.info(f"[Dashboard] Breakouts: {len(result['breakouts'])}")
    except Exception as e:
        logger.error(f"[Dashboard] Breakouts failed: {e}")
        result["breakouts"] = []
    
    # 5. P&L/EXITS: From Sheet (your existing logic, unchanged)
    sl_hit, target_hit, eod_exit = 0, 0, 0
    pnl_list = []
    try:
        if sheet:
            all_rows = sheet.get_all_values()
            for row in all_rows[1:]:
                if len(row) < 14 or (row[0] and row[0][:10] != date_str):
                    continue
                exit_reason = row[12].strip() if len(row) > 12 else ""
                if "SL" in exit_reason: sl_hit += 1
                elif "Target" in exit_reason: target_hit += 1
                elif "EOD" in exit_reason: eod_exit += 1
                pnl_str = (row[13].replace("%", "").strip() if len(row) > 13 else "0")
                pnl_list.append({"symbol": row[1], "pnl": float(pnl_str), "percent_move": float(pnl_str)})
    except Exception as e:
        logger.error(f"[Dashboard] Sheet P&L failed: {e}")
    
    result["exit_conditions"] = {"sl_hit": sl_hit, "target_hit": target_hit, "eod_exit": eod_exit}
    result["pnl"] = {"overall": round(sum(p["pnl"] for p in pnl_list) / len(pnl_list), 2) if pnl_list else 0, "stocks": pnl_list}
    
    logger.info(f"[Dashboard] COMPLETE: fast={result['fast_tier_count']}, monitor={result['monitor_count']}, breakouts={len(result['breakouts'])}")
    return result

@app.get("/api/dashboard")
async def dashboard_endpoint(date: str = None):
    """Dashboard endpoint for breakout page (defaults to today)."""
    if not date:
        date = get_ist_time().strftime("%Y-%m-%d")
    try:
        data = build_dashboard_data(date)
        logger.info(f"[API/Dashboard] Served for {date}: {data['fast_tier_count']} fast, {data['monitor_count']} monitor")
        return data
    except Exception as e:
        logger.error(f"[API/Dashboard] Error for {date}: {e}")
        return {"error": str(e), "fast_tier_count": 0, "fast_tier_symbols": [], "monitor_count": 0}

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
    asyncio.create_task(asyncio.to_thread(start_finding_breakouts))
    logger.info("📡 Stock Breakout Monitor started in background.")

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
    # Use IST time to match the monitor list date format
    today_str = get_ist_time().strftime("%Y-%m-%d")
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
    """Fetches real-time LTP from Kite — works during AND after market hours."""
    try:
        # Always fetch fresh instruments — never rely on cache here
        instruments = kite.instruments("NFO")
        df = pd.DataFrame(instruments)

        # Filter NIFTY options only
        df_nifty = df[(df["name"] == "NIFTY") & (df["instrument_type"].isin(["CE", "PE"]))].copy()
        df_nifty["expiry"] = pd.to_datetime(df_nifty["expiry"]).dt.date

        # Get nearest future expiry
        future_expiries = sorted([e for e in df_nifty["expiry"].unique() if e >= date.today()])
        if not future_expiries:
            logger.warning("No future expiries found")
            return []

        current_expiry = future_expiries[0]
        df_expiry = df_nifty[df_nifty["expiry"] == current_expiry]

        # Get spot price and ATM ± 10 strikes at 50-pt intervals
        spot_quote = kite.quote("NSE:NIFTY 50")
        spot_price = spot_quote["NSE:NIFTY 50"]["last_price"]
        atm = round(spot_price / 50) * 50
        strikes = [atm + (i * 50) for i in range(-10, 11)]
        df_filtered = df_expiry[df_expiry["strike"].isin(strikes)]

        # Fetch quotes in one batch call
        nfo_symbols = [f"NFO:{row['tradingsymbol']}" for _, row in df_filtered.iterrows()]
        all_symbols = nfo_symbols + ["NSE:NIFTY 50"]
        quotes = kite.quote(all_symbols)

        live_data = [{
            "symbol": "NIFTY 50", "strike": "SPOT", "type": "SPOT",
            "ltp": spot_price, "open": spot_price, "change_percent": 0
        }]

        for _, row in df_filtered.iterrows():
            q_key = f"NFO:{row['tradingsymbol']}"
            q = quotes.get(q_key, {})
            ltp = q.get("last_price", 0)
            ohlc = q.get("ohlc", {})
            # After market close, use close price as LTP fallback
            close_price = ohlc.get("close", 0)
            open_price = ohlc.get("open", 0)
            effective_ltp = ltp if ltp > 0 else close_price

            live_data.append({
                "symbol": row["tradingsymbol"],
                "strike": int(row["strike"]),
                "type": row["instrument_type"],
                "ltp": effective_ltp,
                "open": open_price,
                "change_percent": round(((effective_ltp - open_price) / open_price * 100), 2) if open_price > 0 else 0
            })

        logger.info(f"✅ Option chain: {len(live_data)} entries, expiry={current_expiry}, spot={spot_price}")
        return live_data

    except Exception as e:
        logger.error(f"❌ get_live_option_chain error: {e}")
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
    if quantity == 50:
        db_trade = supabase.table("paper_trades").select("quantity").eq("id", trade_id).execute()
        if db_trade.data:
            quantity = db_trade.data[0]["quantity"]
    
    # 2. Get the current Price (Exit Price)
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
        return {"status": "error", "message": "Database sync failed."}

@app.post("/api/check-strategy")
async def check_strategy(data: dict):
    strategy = data.get("strategy")  # e.g. "THREE_GREEN_CANDLES"
    symbol = data.get("symbol")      # e.g. "NIFTY2631024700CE"
    
    if strategy == "THREE_GREEN_CANDLES":
        is_5m  = is_candle_green(5)
        is_15m = is_candle_green(15)
        is_1h  = is_candle_green(60)
        all_green = is_5m and is_15m and is_1h
        return {
            "strategy": strategy,
            "conditions": {
                "5min_green": is_5m,
                "15min_green": is_15m,
                "1hr_green": is_1h
            },
            "signal": "BUY" if all_green else "NO_SIGNAL",
            "target_pct": 2.0,
            "sl_pct": 1.0,
            "message": "All 3 timeframes green — BUY signal confirmed" if all_green else "Conditions not met"
        }
    
    return {"status": "error", "message": "Unknown strategy"}