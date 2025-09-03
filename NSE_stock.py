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
HISTORICAL_HOURS = 200        # Hours of historical data to fetch

def is_trading_day(date):
    """Check if date is a weekday (Mon-Fri)"""
    return date.weekday() < 5

def get_last_trading_date():
    """Get the last trading day date"""
    today = datetime.now().date()
    
    # If today is a trading day, return today
    if is_trading_day(today):
        return today
    
    # Otherwise, go back to find the last trading day
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
        logger.info(f"Total NSE instruments fetched: {len(instruments_raw)}")

        # Filter: only main-board NSE equities (no suffix like -SM, -BE, etc.)
        instruments = [
            inst for inst in instruments_raw
            if inst.get("instrument_type") == "EQ"
            and inst.get("exchange") == "NSE"
            and inst.get("segment") == "NSE"
            and "-" not in inst.get("tradingsymbol", "")
        ]
        
        logger.info(f"Filtered active equity instruments: {len(instruments)}")
        return instruments
    
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        return []

def calculate_moving_averages(data, interval_type="day"):
    """Calculate moving averages from historical data"""
    if not data or len(data) < 200:
        return None
    
    df = pd.DataFrame(data)
    if interval_type == "day":
        df['date'] = pd.to_datetime(df['date']).dt.date
    else:  # hour
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.tail(200).reset_index(drop=True)  # Get last 200 periods
    
    # Calculate moving averages
    df['ma_44'] = df['close'].rolling(window=44, min_periods=44).mean().round(2)
    df['ma_50'] = df['close'].rolling(window=50, min_periods=50).mean().round(2)
    df['ma_100'] = df['close'].rolling(window=100, min_periods=100).mean().round(2)
    df['ma_200'] = df['close'].rolling(window=200, min_periods=200).mean().round(2)
    
    # Get latest values
    latest = df.tail(1).iloc[0]
    
    return {
        'ma_44': None if pd.isna(latest['ma_44']) else float(latest['ma_44']),
        'ma_50': None if pd.isna(latest['ma_50']) else float(latest['ma_50']),
        'ma_100': None if pd.isna(latest['ma_100']) else float(latest['ma_100']),
        'ma_200': None if pd.isna(latest['ma_200']) else float(latest['ma_200']),
        'current_price': float(latest['close']),
        'date': latest['date']
    }

def check_cp_above_all_mas(stock_data):
    """Check if current price is above all 4 moving averages"""
    if not stock_data:
        return False
    
    mas = ['ma_44', 'ma_50', 'ma_100', 'ma_200']
    
    # Check if all MAs are available and CP is above all of them
    for ma in mas:
        if stock_data[ma] is None or stock_data['current_price'] <= stock_data[ma]:
            return False
    
    return True

def get_previous_status_from_supabase():
    """Fetch previous run status from Supabase for both daily and hourly"""
    daily_status = {}
    hourly_status = {}
    
    try:
        # Get daily status
        response = supabase.table("nse_stock_summary").select("*").execute()
        
        for record in response.data:
            symbol = record['company_name']
            
            # Check if CP was above all MAs in previous run
            mas_available = all([
                record.get('ma_44') is not None,
                record.get('ma_50') is not None,
                record.get('ma_100') is not None,
                record.get('ma_200') is not None
            ])
            
            if mas_available:
                cp_above_all_mas = (
                    record['current_price'] > record['ma_44'] and
                    record['current_price'] > record['ma_50'] and
                    record['current_price'] > record['ma_100'] and
                    record['current_price'] > record['ma_200']
                )
                
                daily_status[symbol] = {
                    'cp_above_all_mas': cp_above_all_mas,
                    'current_price': record['current_price'],
                    'updated_at': record.get('updated_at')
                }
        
        logger.info(f"Retrieved daily status for {len(daily_status)} stocks")
        
    except Exception as e:
        logger.error(f"Error fetching daily status: {e}")
    
    try:
        # Get hourly status
        response = supabase.table("nse_stock_hourly").select("*").execute()
        
        for record in response.data:
            symbol = record['company_name']
            
            # Check if CP was above all MAs in previous run
            mas_available = all([
                record.get('ma_44h') is not None,
                record.get('ma_50h') is not None,
                record.get('ma_100h') is not None,
                record.get('ma_200h') is not None
            ])
            
            if mas_available:
                cp_above_all_mas = (
                    record['current_price'] > record['ma_44h'] and
                    record['current_price'] > record['ma_50h'] and
                    record['current_price'] > record['ma_100h'] and
                    record['current_price'] > record['ma_200h']
                )
                
                hourly_status[symbol] = {
                    'cp_above_all_mas': cp_above_all_mas,
                    'current_price': record['current_price'],
                    'updated_at': record.get('updated_at')
                }
        
        logger.info(f"Retrieved hourly status for {len(hourly_status)} stocks")
        
    except Exception as e:
        logger.error(f"Error fetching hourly status: {e}")
    
    return daily_status, hourly_status

def trigger_breakout_notification(symbol, company_name, daily_breakout, hourly_breakout, 
                                daily_previous, daily_current, hourly_previous, hourly_current):
    """Trigger notification when stock breaks above all MAs"""
    current_time = datetime.now().strftime("%I:%M %p")
    
    # Determine breakout type
    if daily_breakout and hourly_breakout:
        breakout_type = "BOTH DAILY & HOURLY BREAKOUT"
        alert_emoji = "🚀🚀🚀"
    elif daily_breakout:
        breakout_type = "DAILY BREAKOUT"
        alert_emoji = "📈"
    elif hourly_breakout:
        breakout_type = "HOURLY BREAKOUT"
        alert_emoji = "⚡"
    else:
        return  # No breakout
    
    message = f"""
{alert_emoji} *** {breakout_type} ALERT *** {alert_emoji}
Stock: {symbol} ({company_name})
Time: {current_time}
"""
    
    if daily_breakout and daily_previous and daily_current:
        message += f"""
🔸 DAILY BREAKOUT:
Previous: CP {daily_previous['current_price']:.2f} was BELOW all 4 Daily MAs
Current: CP {daily_current['current_price']:.2f} is now ABOVE all 4 Daily MAs

Daily Moving Averages:
MA-44d: {daily_current['ma_44']:.2f}
MA-50d: {daily_current['ma_50']:.2f}
MA-100d: {daily_current['ma_100']:.2f}
MA-200d: {daily_current['ma_200']:.2f}
"""
    
    if hourly_breakout and hourly_previous and hourly_current:
        message += f"""
⚡ HOURLY BREAKOUT:
Previous: CP {hourly_previous['current_price']:.2f} was BELOW all 4 Hourly MAs
Current: CP {hourly_current['current_price']:.2f} is now ABOVE all 4 Hourly MAs

Hourly Moving Averages:
MA-44h: {hourly_current['ma_44']:.2f}
MA-50h: {hourly_current['ma_50']:.2f}
MA-100h: {hourly_current['ma_100']:.2f}
MA-200h: {hourly_current['ma_200']:.2f}
"""
    
    # Use the most recent current price for price change calculation
    current_price = daily_current['current_price'] if daily_current else hourly_current['current_price']
    previous_price = daily_previous['current_price'] if daily_previous else hourly_previous['current_price']
    
    if previous_price:
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100
        message += f"""
Current Price: {current_price:.2f}
Price Change: {price_change:+.2f} ({price_change_pct:+.2f}%)
"""
    
    print("=" * 80)
    print(message)
    print("=" * 80)
    
    logger.info(f"{breakout_type}: {symbol} - CP: {current_price:.2f}")

def fetch_and_analyze_stocks():
    """Main function to fetch stock data and detect breakouts"""
    logger.info("Starting stock data fetch and analysis...")
    
    # Get previous status from Supabase
    daily_previous_status, hourly_previous_status = get_previous_status_from_supabase()
    
    # Fetch instruments
    instruments = fetch_nse_instruments()
    if not instruments:
        logger.error("No instruments found. Exiting.")
        return 0, 0
    
    all_daily_data = []
    all_hourly_data = []
    daily_breakout_count = 0
    hourly_breakout_count = 0
    
    # Determine the trading date to fetch data for
    target_date = get_last_trading_date()
    print(f"📊 Analyzing data for trading date: {target_date}")
    logger.info(f"Analyzing data for trading date: {target_date}")
    
    for i, instrument in enumerate(instruments, start=1):
        try:
            symbol = instrument['tradingsymbol']
            company_name = instrument['name']
            token = instrument['instrument_token']
            
            # Fetch daily historical data
            daily_from_date = target_date - timedelta(days=300)
            daily_to_date = target_date
            
            daily_historical_data = kite.historical_data(
                instrument_token=token,
                from_date=daily_from_date,
                to_date=daily_to_date,
                interval="day"
            )
            
            # Fetch hourly historical data
            hourly_from_date = datetime.now() - timedelta(days=30)  # ~720 hours of data
            hourly_to_date = datetime.now()
            
            hourly_historical_data = kite.historical_data(
                instrument_token=token,
                from_date=hourly_from_date.date(),
                to_date=hourly_to_date.date(),
                interval="hour"
            )
            
            # Calculate daily moving averages
            daily_analysis = None
            if daily_historical_data:
                daily_analysis = calculate_moving_averages(daily_historical_data, "day")
            
            # Calculate hourly moving averages
            hourly_analysis = None
            if hourly_historical_data:
                hourly_analysis = calculate_moving_averages(hourly_historical_data, "hour")
            
            # Prepare daily data for Supabase
            if daily_analysis:
                daily_stock_summary = {
                    "company_name": company_name,
                    "ma_44": daily_analysis['ma_44'],
                    "ma_50": daily_analysis['ma_50'],
                    "ma_100": daily_analysis['ma_100'],
                    "ma_200": daily_analysis['ma_200'],
                    "current_price": daily_analysis['current_price'],
                    "updated_at": datetime.now().isoformat()
                }
                all_daily_data.append(daily_stock_summary)
            
            # Prepare hourly data for Supabase
            if hourly_analysis:
                hourly_stock_summary = {
                    "company_name": company_name,
                    "ma_44h": hourly_analysis['ma_44'],
                    "ma_50h": hourly_analysis['ma_50'],
                    "ma_100h": hourly_analysis['ma_100'],
                    "ma_200h": hourly_analysis['ma_200'],
                    "current_price": hourly_analysis['current_price'],
                    "updated_at": datetime.now().isoformat()
                }
                all_hourly_data.append(hourly_stock_summary)
            
            # Check for breakouts
            daily_cp_above_all_mas = check_cp_above_all_mas(daily_analysis) if daily_analysis else False
            hourly_cp_above_all_mas = check_cp_above_all_mas(hourly_analysis) if hourly_analysis else False
            
            # Check daily breakout
            daily_breakout = False
            if symbol in daily_previous_status and daily_analysis:
                previous_daily_status = daily_previous_status[symbol]['cp_above_all_mas']
                if not previous_daily_status and daily_cp_above_all_mas:
                    daily_breakout = True
                    daily_breakout_count += 1
            
            # Check hourly breakout
            hourly_breakout = False
            if symbol in hourly_previous_status and hourly_analysis:
                previous_hourly_status = hourly_previous_status[symbol]['cp_above_all_mas']
                if not previous_hourly_status and hourly_cp_above_all_mas:
                    hourly_breakout = True
                    hourly_breakout_count += 1
            
            # Trigger notification if any breakout occurred
            if daily_breakout or hourly_breakout:
                trigger_breakout_notification(
                    symbol, company_name, 
                    daily_breakout, hourly_breakout,
                    daily_previous_status.get(symbol), daily_analysis,
                    hourly_previous_status.get(symbol), hourly_analysis
                )
            
            # Log new stocks above all MAs
            if daily_cp_above_all_mas and symbol not in daily_previous_status:
                print(f"🆕 New stock above all Daily MAs: {symbol} - CP: {daily_analysis['current_price']:.2f}")
                logger.info(f"New stock above all Daily MAs: {symbol} - CP: {daily_analysis['current_price']:.2f}")
            
            if hourly_cp_above_all_mas and symbol not in hourly_previous_status:
                print(f"⚡ New stock above all Hourly MAs: {symbol} - CP: {hourly_analysis['current_price']:.2f}")
                logger.info(f"New stock above all Hourly MAs: {symbol} - CP: {hourly_analysis['current_price']:.2f}")
            
            # Progress logging
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(instruments)} stocks...")
        
        except Exception as e:
            logger.error(f"Error processing {instrument.get('name', symbol)}: {e}")
            continue
    
    # Store data in Supabase
    if all_daily_data:
        try:
            supabase.table("nse_stock_summary").upsert(
                all_daily_data, on_conflict="company_name"
            ).execute()
            logger.info(f"Updated {len(all_daily_data)} daily stocks in Supabase")
        except Exception as e:
            logger.error(f"Error storing daily data in Supabase: {e}")
    else:
        logger.warning("No daily data to store")
    
    if all_hourly_data:
        try:
            logger.info(f"Attempting to store {len(all_hourly_data)} hourly stocks...")
            supabase.table("nse_stock_hourly").upsert(
                all_hourly_data, on_conflict="company_name"
            ).execute()
            logger.info(f"Updated {len(all_hourly_data)} hourly stocks in Supabase")
        except Exception as e:
            logger.error(f"Error storing hourly data in Supabase: {e}")
            logger.error(f"Sample hourly data: {all_hourly_data[:2] if all_hourly_data else 'None'}")
    else:
        logger.warning("No hourly data to store - check if hourly historical data is available")
    
    total_breakouts = daily_breakout_count + hourly_breakout_count
    logger.info(f"Found {daily_breakout_count} daily breakouts, {hourly_breakout_count} hourly breakouts, {total_breakouts} total breakouts")
    
    return len(all_daily_data), total_breakouts

def run_market_monitoring():
    """Main monitoring loop during market hours"""
    print("🚀 Starting NSE Stock Breakout Detection System (Daily + Hourly)")
    logger.info("Starting NSE Stock Breakout Detection System (Daily + Hourly)")
    
    while True:
        current_time = datetime.now()
        
        # Check if it's a trading day
        if not is_trading_day(current_time.date()):
            print("📅 Non-trading day. Checking for end-of-day breakouts from last trading day...")
            logger.info("Non-trading day. Checking for end-of-day breakouts from last trading day...")
            
            # Perform analysis for the last trading day
            try:
                stocks_processed, breakouts_found = fetch_and_analyze_stocks()
                if breakouts_found > 0:
                    print(f"🎯 Found {breakouts_found} end-of-day breakouts from last trading session!")
                    logger.info(f"Found {breakouts_found} end-of-day breakouts from last trading session!")
                else:
                    print("📈 No end-of-day breakouts detected from last trading session.")
                    logger.info("No end-of-day breakouts detected from last trading session.")
            except Exception as e:
                print(f"❌ Error checking end-of-day breakouts: {e}")
                logger.error(f"Error checking end-of-day breakouts: {e}")
            
            print("💤 Sleeping until next trading day...")
            logger.info("Sleeping until next trading day...")
            # Sleep until next day and check again
            t.sleep(3600)  # Check every hour
            continue
        
        # Check if market is open
        if not is_market_hours():
            if current_time.time() < MARKET_OPEN:
                # Before market opens
                sleep_seconds = (datetime.combine(current_time.date(), MARKET_OPEN) - current_time).total_seconds()
                print(f"⏰ Market opens in {sleep_seconds/60:.0f} minutes. Waiting...")
                logger.info(f"Market opens in {sleep_seconds/60:.0f} minutes. Waiting...")
                t.sleep(min(sleep_seconds + 60, 3600))  # Don't sleep more than 1 hour
            else:
                # After market closes
                print("📊 Market closed. Performing end-of-day data fetch...")
                logger.info("Market closed. Performing end-of-day data fetch...")
                fetch_and_analyze_stocks()
                print("💤 Sleeping until next trading day...")
                logger.info("Sleeping until next trading day...")
                t.sleep(3600)  # Check every hour
            continue
        
        # Market is open - perform analysis
        try:
            print(f"🔍 Analyzing stocks at {current_time.strftime('%I:%M:%S %p')}")
            logger.info(f"Analyzing stocks at {current_time.strftime('%I:%M:%S %p')}")
            stocks_processed, breakouts_found = fetch_and_analyze_stocks()
            
            if breakouts_found > 0:
                print(f"🎯 Found {breakouts_found} new breakouts!")
                logger.info(f"Found {breakouts_found} new breakouts!")
            else:
                print("📈 No new breakouts detected.")
                logger.info("No new breakouts detected.")
            
            # Sleep for CHECK_INTERVAL
            print(f"⏳ Sleeping for {CHECK_INTERVAL} seconds...")
            logger.info(f"Sleeping for {CHECK_INTERVAL} seconds...")
            t.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("🛑 Monitoring stopped by user.")
            logger.info("Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")
            logger.error(f"Error in monitoring loop: {e}")
            print(f"🔄 Retrying in {CHECK_INTERVAL} seconds...")
            logger.info(f"Retrying in {CHECK_INTERVAL} seconds...")
            t.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        run_market_monitoring()
    except KeyboardInterrupt:
        print("🛑 System stopped by user.")
        logger.info("System stopped by user.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")