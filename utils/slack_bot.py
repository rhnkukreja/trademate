import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# ============================
# LOAD ENV
# ============================
load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")

if not SLACK_BOT_TOKEN:
    raise RuntimeError("Missing SLACK_BOT_TOKEN")

if not SLACK_CHANNEL_ID:
    raise RuntimeError("Missing SLACK_CHANNEL_ID")

# ============================
# APP
# ============================
app = FastAPI(title="Breakout Alert Bot")

# ============================
# SCHEMA
# ============================
class BreakoutAlert(BaseModel):
    symbol: str
    price: float
    timeframe: str
    confidence: float

# ============================
# ENDPOINT
# ============================
@app.post("/alert")
async def alert(alert: BreakoutAlert):
    message = (
        "🚨 *BREAKOUT ALERT*\n\n"
        f"*Symbol:* {alert.symbol}\n"
        f"*Price:* {alert.price}\n"
        f"*Timeframe:* {alert.timeframe}\n"
        f"*Confidence:* {alert.confidence:.2f}\n\n"
        "👀 Check and keep your eye on this breakout."
    )

    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "channel": SLACK_CHANNEL_ID,
        "text": message,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        res = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload,
        )

    data = res.json()

    if not data.get("ok"):
        raise HTTPException(
            status_code=500,
            detail=f"Slack API failed: {data}"
        )

    return {
        "status": "sent",
        "channel": data["channel"],
        "ts": data["ts"]
    }