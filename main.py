import os
import time
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
import threading

# Cache for NFO instruments — refreshed once per day
_nfo_cache = {"data": None, "date": None}

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

def get_db_count(date_str: str, tier=None):
    """Helper to fetch monitor counts from Supabase."""
    try:
        if tier:
            exact_r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str).eq("monitoring_tier", tier).execute()
            count = exact_r.count or 0
            if count == 0:
                fallback_r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str).ilike("monitoring_tier", f"%{tier}%").execute()
                count = fallback_r.count or 0
        else:
            r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str).execute()
            count = r.count or 0
        return count
    except Exception as e:
        logger.error(f"[get_count] Error: {e}")
        return 0

def build_dashboard_data(requested_date: str):
    """Builds comprehensive dashboard data for a given date (optimized for breakout page)."""
    date_str = requested_date  # ISO from frontend (e.g., "2026-03-05")

    result = {
        "date": date_str,
        "breakouts": [],  # For "Breakout Price & Time" section
        "pnl": {"overall": 0, "stocks": []},
        "exit_conditions": {"sl_hit": 0, "target_hit": 0, "eod_exit": 0},
        "fast_tier_count": 0,
        "fast_tier_symbols": [],  # Key for "Fast Tier" UI
        "monitor_count": 0  # Key for "Monitor List" UI
    }
    
    # 1. FAST TIER: Query, Enrich, Log (matches your logs: 8 stocks)
    try:
        tier_query = supabase.table("monitor_list").select("*", count="exact").eq("date", date_str).eq("monitoring_tier", "fast")
        r = tier_query.execute()
        result["fast_tier_count"] = r.count or len(r.data or [])
        logger.info(f"[Dashboard] Fast tier raw: count={r.count}, data_len={len(r.data or [])}")
        
        fast_symbols_enriched = []
        for row in (r.data or []):
            candidate = row.get("max_ma") or row.get("breakout_price_candidate") or row.get("breakout_price") or 0
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
    result["monitor_count"] = get_db_count(date_str)
    # 3. LIVE PRICES: Fetch current for fast symbols (Kite, non-blocking)
    try:
        if result["fast_tier_symbols"]:
            nse_symbols = [f"NSE:{s['symbol']}" for s in result["fast_tier_symbols"]]
            quotes = kite.quote(nse_symbols[:50])  # Batch limit
            for s in result["fast_tier_symbols"]:
                q_key = f"NSE:{s['symbol']}"
                q = quotes.get(q_key, {})
                s["current_price"] = round(q.get('last_price', s['breakout_price'] or 0), 2)
            
    except Exception as e:
        logger.error(f"[Dashboard] Live prices failed: {e}")
    
    # 4. BREAKOUTS: From live_breakouts table
    try:
        breakouts_r = supabase.table("live_breakouts").select("*").eq("breakout_date", date_str).order("breakout_time").execute()
        breakouts_data = breakouts_r.data or []
        
        # 🟢 FETCH LIVE PRICES FOR BREAKOUTS
        if breakouts_data:
            breakout_symbols = [f"NSE:{b['symbol']}" for b in breakouts_data]
            live_quotes = kite.quote(breakout_symbols[:50]) # Batch limit
            
            for b in breakouts_data:
                sym_key = f"NSE:{b['symbol']}"
                if sym_key in live_quotes:
                    live_ltp = live_quotes[sym_key].get("last_price")
                    b["current_price"] = live_ltp
                    
                    # 🟢 LIVE RECALCULATION: Override DB % move
                    if b.get("breakout_price") and float(b["breakout_price"]) > 0:
                        b["percent_move"] = round(((live_ltp - float(b["breakout_price"])) / float(b["breakout_price"])) * 100, 2)
                else:
                    b["current_price"] = b.get("breakout_price")
                    
        result["breakouts"] = breakouts_data
       
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
                if not pnl_str: pnl_str = "0"  # 🟢 FIX: Prevent empty string crash
                pnl_list.append({"symbol": row[1], "pnl": float(pnl_str), "percent_move": float(pnl_str)})
    except Exception as e:
        logger.error(f"[Dashboard] Sheet P&L failed: {e}")
    
    result["exit_conditions"] = {"sl_hit": sl_hit, "target_hit": target_hit, "eod_exit": eod_exit}
    result["pnl"] = {"overall": round(sum(p["pnl"] for p in pnl_list) / len(pnl_list), 2) if pnl_list else 0, "stocks": pnl_list}
    
    logger.info(f"[Dashboard] COMPLETE: fast={result['fast_tier_count']}, monitor={result['monitor_count']}, breakouts={len(result['breakouts'])}")
    return result

_dashboard_cache = {"data": None, "date": None, "ts": 0}
async def dashboard_endpoint(date: str = None):
    """Dashboard endpoint for breakout page (defaults to today)."""
    import time
    if not date:
        date = get_ist_time().strftime("%Y-%m-%d")
    try:
        # Serve from cache if same date and less than 30 seconds old
        if (_dashboard_cache["date"] == date and
                _dashboard_cache["data"] is not None and
                time.time() - _dashboard_cache["ts"] < 30):
            return _dashboard_cache["data"]

        data = build_dashboard_data(date)
        _dashboard_cache["data"] = data
        _dashboard_cache["date"] = date
        _dashboard_cache["ts"] = time.time()
        logger.info(f"[API/Dashboard] {date}: fast={data['fast_tier_count']}, monitor={data['monitor_count']}, breakouts={len(data.get('breakouts', []))}")
        return data
    except Exception as e:
        logger.error(f"[API/Dashboard] Error for {date}: {e}")
        return {"error": str(e), "fast_tier_count": 0, "fast_tier_symbols": [], "monitor_count": 0}

class PayloadRequest(BaseModel):
    token: str

@app.get("/api/get-paper-balance")
async def get_paper_balance():
    return {"balance": get_db_balance()}
    
def run_startup_algo_flow():
    """Runs the monitor list build and monitoring loop on server startup."""
    try:
        today_str = date.today().strftime("%Y-%m-%d")
        completion_flag = supabase.table("kite_config") \
            .select("value") \
            .eq("key_name", f"monitor_list_complete_{today_str}") \
            .execute()

        if completion_flag.data and completion_flag.data[0]["value"] == "true":
            logger.info(f"✅ Monitor list fully built for {today_str}. Skipping build.")
        else:
            logger.info(f"🔄 Monitor list incomplete or missing for {today_str}. Building now...")
            create_monitor_list()
        start_finding_breakouts()
    except Exception as e:
        logger.error(f"❌ Startup algo flow error: {e}")

def preload_nfo_data():
    """Background task to fetch NFO instruments so UI loads instantly."""
    try:
        logger.info("⏳ Preloading NFO instruments to RAM...")
        _nfo_cache["data"] = kite.instruments("NFO")
        _nfo_cache["date"] = date.today()
        logger.info("✅ NFO instruments preloaded! UI will now load instantly.")
    except Exception as e:
        logger.error(f"Failed to preload NFO: {e}")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting FastAPI Server & Syncing Session...")
    
    # 1. Sync Token
    token = get_active_token()
    if token:
        kite.set_access_token(token)
    
    # 🟢 NEW: Preload the massive NFO instrument list in the background
    threading.Thread(target=preload_nfo_data, daemon=True).start()
    
    # 2. Start the Options Ticker (The "Ears")
    start_kite_ticker()
    
    # 3. Fire and forget the full algo flow without nested functions
    await asyncio.to_thread(run_startup_algo_flow)
    logger.info("📡 Stock Breakout Monitor + Monitor List build started in background.")

@app.post("/api/place-option-order")
async def place_option_order(data: dict):
    symbol = data.get("symbol")
    price = data.get("price")
    side = data.get("side", "BUY")
    user_qty = data.get("quantity")  # 🟢 Capture what the user actually typed in the UI

    # 🟢 DUPLICATE PROTECTION
    existing_check = supabase.table("paper_trades").select("id").eq("symbol", symbol).eq("status", "OPEN").execute()
    if existing_check.data and len(existing_check.data) > 0:
        return {"status": "ignored", "message": "Position already open"}
    
    # 🟢 FETCH DYNAMIC DATA: Lot Size & Nifty Spot
    try:
        nifty_quote = kite.quote("NSE:NIFTY 50")
        nifty_spot_at_order = nifty_quote["NSE:NIFTY 50"]["last_price"]
        
        # 🟢 If user typed a quantity, use it. Otherwise, fetch Kite's default lot size.
        if user_qty and int(user_qty) > 0:
            qty = int(user_qty)
        else:
            instruments = kite.instruments("NFO")
            instrument_info = next((i for i in instruments if i['tradingsymbol'] == symbol), None)
            qty = instrument_info['lot_size'] if instrument_info else 50
            
    except Exception as e:
        logger.warning(f"⚠️ Kite fetch failed, using fallbacks: {e}")
        nifty_spot_at_order = None
        qty = int(user_qty) if (user_qty and int(user_qty) > 0) else 50

    # 🟢 VERIFY FUNDS
    current_balance = get_db_balance()
    margin_required = price * qty
    if margin_required > current_balance:
        return {"status": "error", "message": "Insufficient funds."}
    
    # 🟢 CALCULATE RISK/REWARD
    sl = round(price * 0.99, 2) if side == "BUY" else round(price * 1.01, 2)
    target = round(price * 1.02, 2) if side == "BUY" else round(price * 0.98, 2)

    # 🟢 SAVE TO MEMORY & DB
    ACTIVE_OPTION_TRADES[symbol] = {"entry": price, "qty": qty, "type": side, "sl": sl, "target": target}
    
    trade_record = {
        "symbol": symbol, "entry_price": price, "quantity": qty,
        "side": side, "status": "OPEN", "sl_price": sl, "target_price": target,
        "nifty_spot_at_order": nifty_spot_at_order,
        "created_at": datetime.now().isoformat()
    }
    
    supabase.table("paper_trades").insert(trade_record).execute()
    
    # Update Balance
    new_balance = round(current_balance - margin_required, 2)
    supabase.table("kite_config").update({"value": str(new_balance)}).eq("key_name", "paper_balance").execute()
    
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


def trigger_full_algo_flow():
    """Background task to force a rebuild and restart of monitoring."""
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
            return

        # 1. Check/Build monitor list
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

@app.api_route("/start-finding-breakouts", methods=["GET", "POST"])
async def handle_find_breakouts(background_tasks: BackgroundTasks):
    """Triggers the build and monitor flow in the background."""
    background_tasks.add_task(trigger_full_algo_flow)
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

_option_chain_cache = {"data": [], "ts": 0}

@app.get("/api/get-live-option-chain")
async def get_live_option_chain():
    """Fetches real-time LTP from Kite — works during AND after market hours."""
    
    # 🟢 FAST CACHE: If fetched within the last 2 seconds, return instantly!
    if time.time() - _option_chain_cache["ts"] < 2 and _option_chain_cache["data"]:
        return _option_chain_cache["data"]

    try:
        # Use daily cache — instruments only change on expiry day
        today = date.today()
        if _nfo_cache["data"] is None or _nfo_cache["date"] != today:
            logger.info("🔄 Refreshing NFO instruments cache...")
            _nfo_cache["data"] = kite.instruments("NFO")
            _nfo_cache["date"] = today
        df = pd.DataFrame(_nfo_cache["data"])

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
        
        # 🟢 SAVE TO CACHE: Store the result and current timestamp
        _option_chain_cache["data"] = live_data
        _option_chain_cache["ts"] = time.time()
        
        return live_data

    except Exception as e:
        logger.error(f"❌ get_live_option_chain error: {e}")
        return []

@app.post("/api/exit-option-trade")
async def exit_option_trade(data: dict):
    symbol = data.get("symbol")
    trade_id = data.get("trade_id")
    
    # 1. Fetch trade directly from DB (Immune to server restarts)
    db_trade = supabase.table("paper_trades").select("*").eq("id", trade_id).execute()
    if not db_trade.data:
        return {"status": "error", "message": "Trade not found in database."}
        
    trade_data = db_trade.data[0]
    entry_price = trade_data["entry_price"]
    quantity = trade_data["quantity"]
    side = trade_data["side"]
    
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
    pnl = (exit_price - entry_price) * quantity if side == "BUY" else (entry_price - exit_price) * quantity
    refund_amount = (entry_price * quantity) + pnl

    # 4. Update Database
    try:
        current_balance = get_db_balance()
        new_balance = round(current_balance + refund_amount, 2)
        supabase.table("kite_config").update({"value": str(new_balance)}).eq("key_name", "paper_balance").execute()
        supabase.table("paper_trades").update({"status": "CLOSED"}).eq("id", trade_id).execute()
        
        # 5. Clear Memory if it exists
        if symbol in ACTIVE_OPTION_TRADES:
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