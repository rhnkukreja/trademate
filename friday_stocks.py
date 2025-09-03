import pandas as pd
from datetime import datetime, timedelta, date, time
import os
import logging
import time as t
from supabase import create_client, Client
from kiteconnect import KiteConnect
from dotenv import load_dotenv

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('8ma_openingprice_monitoring.log', encoding='utf-8'),
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
# Configuration
# --------------------------
DAILY_HISTORICAL_DAYS = 300
HOURLY_HISTORICAL_DAYS = 60
CHECK_INTERVAL = 300  # 5 minutes during market hours

def is_trading_day(check_date):
    """Check if date is a weekday (Mon-Fri)"""
    return check_date.weekday() < 5

def get_last_20_working_days(end_date):
    """Get the last 20 working days including end_date"""
    working_days = []
    current_date = end_date
    
    while len(working_days) < 20:
        if current_date.weekday() < 5:  # Monday=0, Sunday=6
            working_days.append(current_date)
        current_date -= timedelta(days=1)
    
    return sorted(working_days)  # Return in chronological order

def fetch_historical_data_with_retry(token, from_date, to_date, interval="day", max_retries=3):
    """Fetch historical data with retry logic"""
    for attempt in range(max_retries):
        try:
            historical_data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            return historical_data
        except Exception as e:
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"API timeout on attempt {attempt + 1}, retrying in {wait_time} seconds...")
                    t.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return None
            else:
                logger.error(f"API error: {e}")
                return None
    return None

def fetch_nse_instruments():
    """Fetch all active NSE equity instruments"""
    try:
        instruments_raw = kite.instruments("NSE")
        logger.info(f"Total NSE instruments fetched: {len(instruments_raw)}")
        
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

def calculate_daily_moving_averages_with_opening_price(data, target_date):
    """Calculate daily moving averages for a specific date with opening price"""
    if not data or len(data) < 200:
        return None

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df_filtered = df[df['date'] <= target_date].tail(200).reset_index(drop=True)
    
    if len(df_filtered) < 200:
        return None

    # Calculate moving averages using closing prices (standard approach)
    df_filtered['ma_44'] = df_filtered['close'].rolling(window=44, min_periods=44).mean().round(2)
    df_filtered['ma_50'] = df_filtered['close'].rolling(window=50, min_periods=50).mean().round(2)
    df_filtered['ma_100'] = df_filtered['close'].rolling(window=100, min_periods=100).mean().round(2)
    df_filtered['ma_200'] = df_filtered['close'].rolling(window=200, min_periods=200).mean().round(2)

    # Get target date data
    target_data = df_filtered[df_filtered['date'] == target_date]
    if target_data.empty:
        return None

    latest = target_data.iloc[-1]
    
    return {
        'ma_44': None if pd.isna(latest['ma_44']) else float(latest['ma_44']),
        'ma_50': None if pd.isna(latest['ma_50']) else float(latest['ma_50']),
        'ma_100': None if pd.isna(latest['ma_100']) else float(latest['ma_100']),
        'ma_200': None if pd.isna(latest['ma_200']) else float(latest['ma_200']),
        'opening_price': float(latest['open']),    # 9:15 AM opening price
        'closing_price': float(latest['close']),   # 3:30 PM closing price
        'high': float(latest['high']),
        'low': float(latest['low']),
        'date': latest['date']
    }

def calculate_hourly_moving_averages_for_date(data, target_date):
    """Calculate hourly moving averages for a specific date"""
    if not data or len(data) < 200:
        return None

    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['date'])
    df['date_only'] = df['datetime'].dt.date
    df_filtered = df[df['date_only'] <= target_date].tail(200).reset_index(drop=True)
    
    if len(df_filtered) < 200:
        return None

    # Calculate hourly moving averages
    df_filtered['ma_44h'] = df_filtered['close'].rolling(window=44, min_periods=44).mean().round(2)
    df_filtered['ma_50h'] = df_filtered['close'].rolling(window=50, min_periods=50).mean().round(2)
    df_filtered['ma_100h'] = df_filtered['close'].rolling(window=100, min_periods=100).mean().round(2)
    df_filtered['ma_200h'] = df_filtered['close'].rolling(window=200, min_periods=200).mean().round(2)

    # Get target date's last hourly candle
    target_data = df_filtered[df_filtered['date_only'] == target_date]
    if target_data.empty:
        return None

    latest_hourly = target_data.iloc[-1]
    return {
        'ma_44h': None if pd.isna(latest_hourly['ma_44h']) else float(latest_hourly['ma_44h']),
        'ma_50h': None if pd.isna(latest_hourly['ma_50h']) else float(latest_hourly['ma_50h']),
        'ma_100h': None if pd.isna(latest_hourly['ma_100h']) else float(latest_hourly['ma_100h']),
        'ma_200h': None if pd.isna(latest_hourly['ma_200h']) else float(latest_hourly['ma_200h']),
        'current_price': float(latest_hourly['close']),
        'datetime': latest_hourly['datetime']
    }

def check_opening_price_below_all_8_mas(daily_data, hourly_data):
    """Check if OPENING PRICE (9:15 AM) is BELOW all 8 moving averages"""
    if not daily_data or not hourly_data:
        return False

    opening_price = daily_data['opening_price']  # Use 9:15 AM opening price
    
    # Check all 4 daily MAs
    daily_mas = ['ma_44', 'ma_50', 'ma_100', 'ma_200']
    for ma in daily_mas:
        if daily_data[ma] is None or opening_price > daily_data[ma]:
            return False
    
    # Check all 4 hourly MAs  
    hourly_mas = ['ma_44h', 'ma_50h', 'ma_100h', 'ma_200h']
    for ma in hourly_mas:
        if hourly_data[ma] is None or opening_price > hourly_data[ma]:
            return False
    
    return True

def check_closing_price_above_all_8_mas(daily_data, hourly_data):
    """Check if CLOSING PRICE (3:30 PM) is ABOVE all 8 moving averages"""
    if not daily_data or not hourly_data:
        return False

    closing_price = daily_data['closing_price']  # Use 3:30 PM closing price
    
    # Check all 4 daily MAs
    daily_mas = ['ma_44', 'ma_50', 'ma_100', 'ma_200']
    for ma in daily_mas:
        if daily_data[ma] is None or closing_price <= daily_data[ma]:
            return False
    
    # Check all 4 hourly MAs
    hourly_mas = ['ma_44h', 'ma_50h', 'ma_100h', 'ma_200h']
    for ma in hourly_mas:
        if hourly_data[ma] is None or closing_price <= hourly_data[ma]:
            return False
    
    return True

def check_stock_opening_below_8mas_for_20_days(symbol, company_name, token, last_20_days):
    """Check if stock's opening price was below all 8 MAs for all 20 consecutive days"""
    
    # Fetch daily historical data
    daily_from_date = last_20_days[0] - timedelta(days=DAILY_HISTORICAL_DAYS)
    daily_to_date = last_20_days[-1]
    
    daily_historical_data = fetch_historical_data_with_retry(
        token=token,
        from_date=daily_from_date,
        to_date=daily_to_date,
        interval="day"
    )
    
    if not daily_historical_data:
        return False, None, None

    # Fetch hourly historical data
    hourly_from_date = last_20_days[0] - timedelta(days=HOURLY_HISTORICAL_DAYS)
    hourly_to_date = last_20_days[-1]
    
    hourly_historical_data = fetch_historical_data_with_retry(
        token=token,
        from_date=hourly_from_date,
        to_date=hourly_to_date,
        interval="hour"
    )
    
    if not hourly_historical_data:
        return False, None, None
    
    # Check each of the 20 days - opening price vs 8 MAs
    below_all_8mas_consecutive = True
    today_daily_data = None
    today_hourly_data = None
    
    for day_date in last_20_days:
        # Calculate MAs for this day
        daily_analysis = calculate_daily_moving_averages_with_opening_price(daily_historical_data, day_date)
        hourly_analysis = calculate_hourly_moving_averages_for_date(hourly_historical_data, day_date)
        
        if not daily_analysis or not hourly_analysis:
            below_all_8mas_consecutive = False
            break
        
        # Store today's data for potential breakout check
        if day_date == last_20_days[-1]:  # Today (last day)
            today_daily_data = daily_analysis
            today_hourly_data = hourly_analysis
        
        # Check if OPENING PRICE was below all 8 MAs on this day
        if not check_opening_price_below_all_8_mas(daily_analysis, hourly_analysis):
            below_all_8mas_consecutive = False
            break
    
    return below_all_8mas_consecutive, today_daily_data, today_hourly_data

def trigger_opening_price_alert(symbol, company_name, daily_data, hourly_data, is_breakout=False):
    """Alert for qualified stocks or breakouts based on opening price analysis"""
    
    if is_breakout:
        alert_type = "🚀🚀🚀 OPENING PRICE BREAKOUT ALERT 🚀🚀🚀"
        status = f"Opening price below 8 MAs for 20 days, closing price NOW ABOVE all 8 MAs!"
    else:
        alert_type = "📊 QUALIFIED STOCK (Opening Price Analysis)"
        status = f"Opening price below all 8 MAs for 20 consecutive days"
    
    message = f"""
{alert_type}
Stock: {symbol} ({company_name})
Status: {status}
Time: {datetime.now().strftime('%I:%M %p')}

TODAY'S PRICES:
Opening Price (9:15 AM): ₹{daily_data['opening_price']:.2f}
Closing Price (3:30 PM): ₹{daily_data['closing_price']:.2f}

TODAY'S MOVING AVERAGES:
📈 DAILY MAs:  MA44={daily_data['ma_44']:.2f}, MA50={daily_data['ma_50']:.2f}, MA100={daily_data['ma_100']:.2f}, MA200={daily_data['ma_200']:.2f}
⚡ HOURLY MAs: MA44h={hourly_data['ma_44h']:.2f}, MA50h={hourly_data['ma_50h']:.2f}, MA100h={hourly_data['ma_100h']:.2f}, MA200h={hourly_data['ma_200h']:.2f}

ANALYSIS:
- Opening prices were below ALL 8 MAs for 20 consecutive days
- Stock was under consistent selling pressure at market opens
"""
    
    if is_breakout:
        price_change = daily_data['closing_price'] - daily_data['opening_price']
        price_change_pct = (price_change / daily_data['opening_price']) * 100
        message += f"""
- Closing price broke ABOVE all 8 MAs today! 
- Intraday gain: ₹{price_change:+.2f} ({price_change_pct:+.2f}%)

🚀🚀🚀 ULTRA-HIGH QUALITY BREAKOUT DETECTED! 🚀🚀🚀"""
    else:
        message += f"\n- Ready for potential breakout - monitor for closing above all 8 MAs!"

    print("=" * 120)
    print(message)
    print("=" * 120)
    
    log_msg = f"OPENING PRICE QUALIFIED: {symbol}" if not is_breakout else f"OPENING PRICE BREAKOUT: {symbol}"
    logger.info(log_msg)

def analyze_opening_price_8ma_system():
    """Main function to analyze opening price vs 8-MA system"""
    
    today = datetime.now().date()
    last_20_working_days = get_last_20_working_days(today)
    
    print(f"📊 OPENING PRICE 8-MA ANALYSIS SYSTEM")
    print(f"🗓️ Analysis period: {last_20_working_days[0]} to {last_20_working_days[-1]}")
    print(f"🎯 Condition: Opening price (9:15 AM) < ALL 8 MAs for 20 consecutive days")
    print(f"🚀 Breakout: Closing price (3:30 PM) > ALL 8 MAs today")
    
    logger.info(f"Opening price analysis: {last_20_working_days[0]} to {last_20_working_days[-1]}")
    
    # Fetch instruments
    instruments = fetch_nse_instruments()
    if not instruments:
        logger.error("No instruments found. Exiting.")
        return

    qualified_stocks = []
    breakout_stocks = []
    total_processed = 0
    
    for i, instrument in enumerate(instruments, start=1):
        try:
            symbol = instrument['tradingsymbol']
            company_name = instrument['name']
            token = instrument['instrument_token']
            
            # Check if stock's opening prices were below all 8 MAs for all 20 days
            is_qualified, today_daily, today_hourly = check_stock_opening_below_8mas_for_20_days(
                symbol, company_name, token, last_20_working_days
            )
            
            if is_qualified and today_daily and today_hourly:
                total_processed += 1
                
                # Check if today's closing price broke above all 8 MAs (BREAKOUT)
                is_breaking_out = check_closing_price_above_all_8_mas(today_daily, today_hourly)
                
                if is_breaking_out:
                    # BREAKOUT DETECTED!
                    trigger_opening_price_alert(symbol, company_name, today_daily, today_hourly, is_breakout=True)
                    breakout_stocks.append({
                        'symbol': symbol,
                        'company_name': company_name,
                        'opening_price': today_daily['opening_price'],
                        'closing_price': today_daily['closing_price'],
                        'intraday_gain': today_daily['closing_price'] - today_daily['opening_price'],
                        'intraday_gain_pct': ((today_daily['closing_price'] - today_daily['opening_price']) / today_daily['opening_price'] * 100)
                    })
                else:
                    # Qualified but not breaking out yet
                    trigger_opening_price_alert(symbol, company_name, today_daily, today_hourly, is_breakout=False)
                    qualified_stocks.append({
                        'symbol': symbol,
                        'company_name': company_name,
                        'opening_price': today_daily['opening_price'],
                        'closing_price': today_daily['closing_price']
                    })

            # Progress logging
            if i % 50 == 0:
                print(f"⏳ Processed {i}/{len(instruments)} stocks... Qualified: {len(qualified_stocks)}, Breakouts: {len(breakout_stocks)}")
            
            # Small delay to avoid API limits
            if i % 10 == 0:
                t.sleep(0.5)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    # Final Summary
    print(f"\n📊 OPENING PRICE 8-MA ANALYSIS COMPLETE")
    print(f"🎯 Stocks with opening prices below all 8 MAs for 20 days: {len(qualified_stocks)}")
    print(f"🚀 Stocks breaking above all 8 MAs today (closing): {len(breakout_stocks)}")
    print(f"📈 Total qualified stocks: {len(qualified_stocks) + len(breakout_stocks)}")

    if breakout_stocks:
        print(f"\n🚀 BREAKOUT STOCKS SUMMARY:")
        print("-" * 100)
        print(f"{'SYMBOL':<12} | {'COMPANY':<25} | {'OPEN':>8} | {'CLOSE':>8} | {'GAIN':>6} | {'GAIN%':>7}")
        print("-" * 100)
        for stock in breakout_stocks:
            print(f"{stock['symbol']:<12} | {stock['company_name'][:25]:<25} | "
                  f"₹{stock['opening_price']:>7.2f} | ₹{stock['closing_price']:>7.2f} | "
                  f"₹{stock['intraday_gain']:>5.2f} | {stock['intraday_gain_pct']:>6.2f}%")
        print("-" * 100)

    logger.info(f"Opening price analysis complete. Qualified: {len(qualified_stocks)}, Breakouts: {len(breakout_stocks)}")

if __name__ == "__main__":
    try:
        print("🚀 Starting Opening Price 8-MA Analysis System")
        print("🎯 Detection Logic:")
        print("   Step 1: Find stocks with opening price < ALL 8 MAs for 20 consecutive days")
        print("   Step 2: Check if today's closing price > ALL 8 MAs (breakout)")
        print("🏆 This finds the highest quality breakouts based on opening price behavior!")
        print("-" * 120)
        
        analyze_opening_price_8ma_system()
        
    except KeyboardInterrupt:
        print("🛑 Analysis stopped by user.")
        logger.info("Analysis stopped by user.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
