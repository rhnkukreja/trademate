import pandas as pd
from datetime import datetime, timedelta, time
import os
import time as t
import logging
from supabase import create_client, Client
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import json

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_breakout.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

# --------------------------
# Supabase Configuration
# --------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------
# Zerodha Configuration
# --------------------------
API_KEY = os.getenv("ZERODHA_API_KEY")
ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN")
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# --------------------------
# Trading Configuration
# --------------------------
MARKET_OPEN = time(9, 1)      # Start monitoring at 9:01 AM
MARKET_CLOSE = time(15, 30)   # Stop monitoring at 3:30 PM
CHECK_INTERVAL = 60           # Check every 1 minute
HISTORICAL_DAYS = 200         # Days of historical data to fetch

# Global Watchlist (kept in memory)
WATCHLIST = set()

def is_trading_day(date):
    """Check if date is a weekday (Mon-Fri)"""
    return date.weekday() < 5

def get_last_trading_date():
    """Get the last trading day date"""
    today = datetime.now().date()
    if is_trading_day(today):
        return today
    check_date = today
    while not is_trading_day(check_date):
        check_date -= timedelta(days=1)
    return check_date

def is_market_hours():
    """Check if current time is within market hours"""
    now = datetime.now()
    current_time = now.time()
    return (is_trading_day(now.date()) and 
            MARKET_OPEN <= current_time <= MARKET_CLOSE)

def fetch_nse_instruments():
    """Fetch all active NSE equity instruments"""
    try:
        instruments_raw = kite.instruments("NSE")
        instruments = [
            inst for inst in instruments_raw
            if inst.get("instrument_type") == "EQ"
            and inst.get("exchange") == "NSE"
            and inst.get("segment") == "NSE"
            and "-" not in inst.get("tradingsymbol", "")
        ]
        return instruments
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        return []

def calculate_moving_averages(data):
    """Calculate moving averages from historical data"""
    if not data or len(data) < 200:
        return None
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.tail(200).reset_index(drop=True)
    df['ma_44'] = df['close'].rolling(window=44).mean().round(2)
    df['ma_50'] = df['close'].rolling(window=50).mean().round(2)
    df['ma_100'] = df['close'].rolling(window=100).mean().round(2)
    df['ma_200'] = df['close'].rolling(window=200).mean().round(2)
    latest = df.tail(1).iloc[0]
    return {
        'ma_44': float(latest['ma_44']),
        'ma_50': float(latest['ma_50']),
        'ma_100': float(latest['ma_100']),
        'ma_200': float(latest['ma_200']),
        'current_price': float(latest['close']),
        'date': latest['date']
    }

def percentage_diff(ma, cp):
    """Calculate % difference if CP < MA"""
    if cp < ma:
        return ((ma - cp) / ma) * 100
    return None

def eligible_for_watchlist(stock_data):
    """Check if stock qualifies for watchlist (CP < any MA within 20%)"""
    cp = stock_data['current_price']
    for ma in ['ma_44', 'ma_50', 'ma_100', 'ma_200']:
        if stock_data[ma] is not None:
            diff = percentage_diff(stock_data[ma], cp)
            if diff is not None and diff <= 20:
                return True
    return False

def check_cp_above_all_mas(stock_data):
    """Check if current price is above all 4 moving averages"""
    if not stock_data:
        return False
    mas = ['ma_44', 'ma_50', 'ma_100', 'ma_200']
    for ma in mas:
        if stock_data[ma] is None or stock_data['current_price'] <= stock_data[ma]:
            return False
    return True

def trigger_breakout_notification(symbol, company_name, previous_data, current_data):
    """Trigger notification when stock breaks above all MAs"""
    current_time = datetime.now().strftime("%I:%M %p")
    print("=" * 60)
    print(f"🚨 BREAKOUT ALERT 🚨")
    print(f"Stock: {symbol} ({company_name})")
    print(f"Previous CP: {previous_data['current_price']:.2f} (Below all MAs)")
    print(f"Current CP: {current_data['current_price']:.2f} (Above all MAs)")
    print(f"Time: {current_time}")
    print("=" * 60)

def fetch_and_analyze_stocks(initial_run=False):
    """Main function to fetch stock data and detect breakouts"""
    global WATCHLIST
    instruments = fetch_nse_instruments()
    if not instruments:
        return

    all_stock_data = []
    breakout_count = 0
    target_date = get_last_trading_date()

    for instrument in instruments:
        try:
            symbol = instrument['tradingsymbol']
            company_name = instrument['name']
            token = instrument['instrument_token']
            from_date = target_date - timedelta(days=300)
            to_date = target_date
            historical_data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            if not historical_data:
                continue
            stock_analysis = calculate_moving_averages(historical_data)
            if not stock_analysis:
                continue

            # Add to watchlist (initial or dynamically)
            if eligible_for_watchlist(stock_analysis):
                if symbol not in WATCHLIST:
                    WATCHLIST.add(symbol)
                    if not initial_run:
                        print(f"🆕 {symbol} added to watchlist")
                        print(f"📊 Now monitoring {len(WATCHLIST)} stocks")

            # Only analyze stocks in watchlist
            if symbol not in WATCHLIST:
                continue

            # Check for breakout
            cp_above_all = check_cp_above_all_mas(stock_analysis)
            if cp_above_all:
                trigger_breakout_notification(symbol, company_name,
                                              {"current_price": stock_analysis['current_price']},
                                              stock_analysis)
                breakout_count += 1

            # Prepare Supabase data
            stock_summary = {
                "company_name": company_name,
                "ma_44": stock_analysis['ma_44'],
                "ma_50": stock_analysis['ma_50'],
                "ma_100": stock_analysis['ma_100'],
                "ma_200": stock_analysis['ma_200'],
                "current_price": stock_analysis['current_price'],
                "updated_at": datetime.now().isoformat()
            }
            all_stock_data.append(stock_summary)

        except Exception as e:
            logger.error(f"Error processing {instrument.get('name', symbol)}: {e}")
            continue

    # Store data in Supabase
    if all_stock_data:
        try:
            supabase.table("nse_stock_summary").upsert(
                all_stock_data, on_conflict="company_name"
            ).execute()
        except Exception as e:
            logger.error(f"Error storing data in Supabase: {e}")

    return len(all_stock_data), breakout_count

def run_market_monitoring():
    global WATCHLIST
    print("🚀 Starting NSE Stock Breakout Detection System")
    while True:
        current_time = datetime.now()
        if not is_trading_day(current_time.date()):
            print("📅 Non-trading day. Sleeping...")
            t.sleep(3600)
            continue
        if not is_market_hours():
            if current_time.time() < MARKET_OPEN:
                print("🌅 Preparing initial watchlist at 9:00 AM...")
                fetch_and_analyze_stocks(initial_run=True)
                print(f"📊 Initial Watchlist: {len(WATCHLIST)} stocks being monitored today")
                sleep_seconds = (datetime.combine(current_time.date(), MARKET_OPEN) - current_time).total_seconds()
                t.sleep(sleep_seconds)
            else:
                print("🌙 Market closed. Sleeping...")
                t.sleep(3600)
            continue

        # Market open - monitoring
        print(f"🔍 Analyzing watchlist ({len(WATCHLIST)} stocks) at {current_time.strftime('%I:%M:%S %p')}")
        fetch_and_analyze_stocks(initial_run=False)
        t.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_market_monitoring()
