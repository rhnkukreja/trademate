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
from datetime import date

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

@app.post("/start-finding-breakouts")
async def handle_find_breakouts(background_tasks: BackgroundTasks):
    try:
        today = date.today().strftime("%Y-%m-%d")

        # 🔍 Check if monitor list already exists for today
        existing = supabase.table("monitor_list") \
            .select("symbol", count="exact") \
            .eq("date", today) \
            .limit(1) \
            .execute()

        if not existing.count:
            print("🔄 No monitor list found for today. Building...")
            create_monitor_list()
            print("✅ Monitor list built.")
        else:
            print("✅ Monitor list already exists. Skipping build.")

        # 🚀 Start breakout monitoring
        background_tasks.add_task(start_finding_breakouts)

        return {
            "status": "success",
            "message": "Breakout monitoring started."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["Health"])
async def health_check():
    return PlainTextResponse("OK", status_code=200)

@app.get("/api/dashboard")
def get_dashboard(date: str):
    result = {}
    # 1. Monitor list count
    try:
        r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date).execute()
        result["monitor_list_count"] = r.count
    except:
        result["monitor_list_count"] = 0

    # 2. Slow tier count
    try:
        r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date).eq("monitoring_tier", "slow").execute()
        result["slow_tier_count"] = r.count
    except:
        result["slow_tier_count"] = 0
        
    # 3. Fast tier count + symbols
    try:
        r = supabase.table("monitor_list").select("symbol", count="exact").eq("date", date).eq("monitoring_tier", "fast").execute()
        result["fast_tier_count"] = r.count
        result["fast_tier_symbols"] = [row["symbol"] for row in r.data]
    except:
        result["fast_tier_count"] = 0
        result["fast_tier_symbols"] = []

    # 4. Breakout details
    try:
        r = supabase.table("live_breakouts") \
            .select("symbol, breakout_price, breakout_time") \
            .eq("breakout_date", date) \
            .execute()

        result["breakouts"] = r.data or []
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
                if row_date != date:
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
                    pnl_list.append({"symbol": row[1], "pnl": float(pnl_str)})
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