import requests
import uvicorn
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
from utils.common import supabase
from datetime import date, datetime
from utils.refresh_token import get_kite_access_token, update_supabase_token
from utils.common import kite

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
    allow_origins=["*"],  # ⚠️ Use only for development!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PayloadRequest(BaseModel):
    token: str # Example field

@app.post("/refresh-kite-session")
async def trigger_token_refresh(background_tasks: BackgroundTasks):
    """
    Endpoint for cron-job.org to refresh the Kite access token.
    Runs in background to avoid cron-job.org timeout.
    """
    def task():
        try:
            print("🔄 Automation: Starting headless login...")
            new_token = get_kite_access_token()
            update_supabase_token(new_token)
            kite.set_access_token(new_token)
            print("✅ Automation: Token refresh successful.")
        except Exception as e:
            print(f"❌ Automation: Refresh failed: {e}")

    background_tasks.add_task(task)
    return {"status": "request_received", "message": "Token refresh started in background."}

@app.post("/start-finding-breakouts")
async def handle_find_breakouts(background_tasks: BackgroundTasks):
    """
    Triggers the build and monitor flow in the background to prevent 
    HTTP timeouts and handle memory spikes safely.
    """
    def full_algo_flow():
        try:
            today_str = date.today().strftime("%Y-%m-%d")

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
    
    
def build_dashboard_data(date_str: str):
    """All the existing dashboard logic moved here so we can reuse it."""
    result = {}
    
    # 1. Monitor list count
    try:
        r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str).execute()
        result["monitor_list_count"] = r.count
    except:
        result["monitor_list_count"] = 0

    # 2. Slow tier count
    try:
        r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date_str).eq("monitoring_tier", "slow").execute()
        result["slow_tier_count"] = r.count
    except:
        result["slow_tier_count"] = 0
        
    # 3. Fast tier count + symbols
    try:
        r = supabase.table("monitor_list") \
            .select("symbol, max_ma, tick_size", count="exact") \
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
                "current_price": None
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

    # 4. Breakout details
    try:
        r = supabase.table("live_breakouts") \
            .select("symbol, breakout_price, breakout_time, percent_move, high_price, exit_reason") \
            .eq("breakout_date", date_str) \
            .execute()

        result["breakouts"] = r.data or []
        for b in result["breakouts"]:
            exit_reason = b.get("exit_reason") or ""
            if "Target" in exit_reason:
                b["status"] = "Target Hit"
            elif "SL" in exit_reason:
                b["status"] = "SL Hit"
            elif "EOD" in exit_reason:
                b["status"] = "EOD Exit"
            elif b.get("percent_move") is not None:
                b["status"] = "In Play"
            else:
                b["status"] = "In Play"
        result["breakout_count"] = len(result["breakouts"])

    except Exception:
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