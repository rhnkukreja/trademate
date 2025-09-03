import pandas as pd
from datetime import datetime, timedelta, time
import os
import logging
import time as t
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
        logging.FileHandler('single_day_8ma_breakout_monitor.log', encoding='utf-8'),
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

def is_trading_day(date):
    """
    Check if the given date is a trading day (Monday to Friday, excluding weekends).
    """
    return date.weekday() < 5  # Monday=0, Sunday=6

def get_user_input_date():
    """
    Get a valid trading date from user input.
    Keeps asking until user provides a valid trading day.
    """
    while True:
        try:
            # Get date input from user
            date_str = input("\n📅 Enter the trading date (YYYY-MM-DD format): ").strip()
            
            # Parse the date string
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Check if it's a trading day
            if is_trading_day(target_date):
                print(f"✅ Valid trading date: {target_date}")
                return target_date
            else:
                # Get day name for better user feedback
                day_name = target_date.strftime("%A")
                print(f"❌ {target_date} ({day_name}) is not a trading day (weekends are not trading days).")
                print("Please enter a weekday date (Monday-Friday).")
                
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-01-15)")
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")

def fetch_historical_data_with_retry(token, from_date, to_date, interval="day", max_retries=3):
    """
    Fetch historical data from Zerodha API with retry logic for handling timeouts.
    Args:
        token: Instrument token from Zerodha
        from_date: Start date for data
        to_date: End date for data
        interval: Data interval (day, hour, minute)
        max_retries: Maximum number of retry attempts
    Returns:
        list: Historical data or None if failed
    """
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
            # Check if it's a timeout or connection error
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Progressive wait: 2, 4, 6 seconds
                    logger.warning(f"API timeout on attempt {attempt + 1}, retrying in {wait_time} seconds...")
                    t.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return None
            else:
                # For non-timeout errors, don't retry
                logger.error(f"API error (not timeout): {e}")
                return None
    return None

def fetch_nse_instruments():
    """
    Fetch all active NSE equity instruments from Zerodha.
    Filters for main-board equity stocks only.
    Returns:
        list: List of instrument dictionaries
    """
    try:
        # Get all NSE instruments
        instruments_raw = kite.instruments("NSE")
        logger.info(f"Total NSE instruments fetched: {len(instruments_raw)}")
        
        # Filter for equity stocks only (main-board, no derivatives)
        instruments = [
            inst for inst in instruments_raw
            if inst.get("instrument_type") == "EQ"          # Equity type
            and inst.get("exchange") == "NSE"               # NSE exchange
            and inst.get("segment") == "NSE"                # NSE segment
            and "-" not in inst.get("tradingsymbol", "")    # No derivatives (avoid stocks with '-' in symbol)
        ]
        
        logger.info(f"Filtered active equity instruments: {len(instruments)}")
        return instruments
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        return []

def calculate_moving_averages_at_9am(daily_data, hourly_data, target_date):
    """
    Calculate all 8 moving averages (4 daily + 4 hourly) at 9:00 AM for the target date.
    Args:
        daily_data: List of daily OHLCV data
        hourly_data: List of hourly OHLCV data
        target_date: The date to calculate MAs for
    Returns:
        dict: Contains current price and all 8 MAs, or None if calculation fails
    """
    if not daily_data or not hourly_data:
        return None
    
    # Process Daily Data for Daily MAs
    daily_df = pd.DataFrame(daily_data)
    daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
    
    # Filter daily data up to target date and get last 200 days for MA calculation
    daily_filtered = daily_df[daily_df['date'] <= target_date].tail(200).reset_index(drop=True)
    
    if len(daily_filtered) < 200:  # Need enough data for MA200
        return None
    
    # Calculate daily moving averages
    daily_filtered['ma_44'] = daily_filtered['close'].rolling(window=44, min_periods=44).mean().round(2)
    daily_filtered['ma_50'] = daily_filtered['close'].rolling(window=50, min_periods=50).mean().round(2)
    daily_filtered['ma_100'] = daily_filtered['close'].rolling(window=100, min_periods=100).mean().round(2)
    daily_filtered['ma_200'] = daily_filtered['close'].rolling(window=200, min_periods=200).mean().round(2)
    
    # Get daily MAs for target date
    daily_target = daily_filtered[daily_filtered['date'] == target_date]
    if daily_target.empty:
        return None
    
    latest_daily = daily_target.iloc[-1]
    
    # Process Hourly Data for Hourly MAs
    hourly_df = pd.DataFrame(hourly_data)
    hourly_df['datetime'] = pd.to_datetime(hourly_df['date'])
    hourly_df['date_only'] = hourly_df['datetime'].dt.date
    hourly_df['time_only'] = hourly_df['datetime'].dt.time
    
    # Filter hourly data up to 9:00 AM on target date
    morning_time = time(9, 15)  # Market opens at 9:15 AM, use this as 9:00 AM proxy
    hourly_filtered = hourly_df[
        (hourly_df['date_only'] < target_date) | 
        ((hourly_df['date_only'] == target_date) & (hourly_df['time_only'] <= morning_time))
    ].tail(200).reset_index(drop=True)
    
    if len(hourly_filtered) < 200:  # Need enough data for MA200h
        return None
    
    # Calculate hourly moving averages
    hourly_filtered['ma_44h'] = hourly_filtered['close'].rolling(window=44, min_periods=44).mean().round(2)
    hourly_filtered['ma_50h'] = hourly_filtered['close'].rolling(window=50, min_periods=50).mean().round(2)
    hourly_filtered['ma_100h'] = hourly_filtered['close'].rolling(window=100, min_periods=100).mean().round(2)
    hourly_filtered['ma_200h'] = hourly_filtered['close'].rolling(window=200, min_periods=200).mean().round(2)
    
    # Get current price at 9:00 AM (from latest hourly candle)
    current_price_data = hourly_filtered[
        (hourly_filtered['date_only'] == target_date) & 
        (hourly_filtered['time_only'] <= morning_time)
    ]
    
    if current_price_data.empty:
        return None
    
    latest_hourly = current_price_data.iloc[-1]
    
    # Return all calculated data
    return {
        'date': target_date,
        'current_price': float(latest_hourly['close']),
        'ma_44': None if pd.isna(latest_daily['ma_44']) else float(latest_daily['ma_44']),
        'ma_50': None if pd.isna(latest_daily['ma_50']) else float(latest_daily['ma_50']),
        'ma_100': None if pd.isna(latest_daily['ma_100']) else float(latest_daily['ma_100']),
        'ma_200': None if pd.isna(latest_daily['ma_200']) else float(latest_daily['ma_200']),
        'ma_44h': None if pd.isna(latest_hourly['ma_44h']) else float(latest_hourly['ma_44h']),
        'ma_50h': None if pd.isna(latest_hourly['ma_50h']) else float(latest_hourly['ma_50h']),
        'ma_100h': None if pd.isna(latest_hourly['ma_100h']) else float(latest_hourly['ma_100h']),
        'ma_200h': None if pd.isna(latest_hourly['ma_200h']) else float(latest_hourly['ma_200h']),
        'datetime': latest_hourly['datetime']
    }

def is_below_all_8_mas(analysis_data):
    """
    Check if current price at 9:00 AM is BELOW all 8 moving averages.
    This determines if stock should be added to monitoring list.
    
    Args:
        analysis_data: Dictionary containing price and MA data
        
    Returns:
        bool: True if price is below all 8 MAs, False otherwise
    """
    if not analysis_data:
        return False
    
    # List of all 8 moving averages
    mas = ['ma_44', 'ma_50', 'ma_100', 'ma_200', 'ma_44h', 'ma_50h', 'ma_100h', 'ma_200h']
    
    # Check if current price is below ALL moving averages
    for ma in mas:
        if analysis_data[ma] is None or analysis_data['current_price'] >= analysis_data[ma]:
            return False  # If price is above any MA or MA is None, return False
    
    return True  # Price is below all 8 MAs

def calculate_volume_averages(daily_data, hourly_data, target_date, breakout_time):
    """
    Calculate volume averages for different periods at the time of breakout.
    
    Args:
        daily_data: List of daily OHLCV data
        hourly_data: List of hourly OHLCV data
        target_date: Date of breakout
        breakout_time: Time of breakout
        
    Returns:
        dict: Volume averages for different periods
    """
    try:
        # Process Daily Volume Data
        daily_df = pd.DataFrame(daily_data)
        daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
        daily_filtered = daily_df[daily_df['date'] <= target_date].reset_index(drop=True)
        
        # Calculate daily volume averages
        volume_data = {
            'daily_volume_avg_44': daily_filtered['volume'].tail(44).mean() if len(daily_filtered) >= 44 else None,
            'daily_volume_avg_50': daily_filtered['volume'].tail(50).mean() if len(daily_filtered) >= 50 else None,
            'daily_volume_avg_100': daily_filtered['volume'].tail(100).mean() if len(daily_filtered) >= 100 else None,
            'daily_volume_avg_200': daily_filtered['volume'].tail(200).mean() if len(daily_filtered) >= 200 else None,
        }
        
        # Process Hourly Volume Data
        hourly_df = pd.DataFrame(hourly_data)
        hourly_df['datetime'] = pd.to_datetime(hourly_df['date'])
        hourly_df['date_only'] = hourly_df['datetime'].dt.date
        hourly_df['time_only'] = hourly_df['datetime'].dt.time
        
        # Filter hourly data up to breakout time
        hourly_filtered = hourly_df[
            (hourly_df['date_only'] < target_date) | 
            ((hourly_df['date_only'] == target_date) & (hourly_df['time_only'] <= breakout_time))
        ].reset_index(drop=True)
        
        # Calculate hourly volume averages
        volume_data.update({
            'hourly_volume_avg_44h': hourly_filtered['volume'].tail(44).mean() if len(hourly_filtered) >= 44 else None,
            'hourly_volume_avg_50h': hourly_filtered['volume'].tail(50).mean() if len(hourly_filtered) >= 50 else None,
            'hourly_volume_avg_100h': hourly_filtered['volume'].tail(100).mean() if len(hourly_filtered) >= 100 else None,
            'hourly_volume_avg_200h': hourly_filtered['volume'].tail(200).mean() if len(hourly_filtered) >= 200 else None,
        })
        
        return volume_data
        
    except Exception as e:
        logger.error(f"Error calculating volume averages: {e}")
        return {}

def scan_stocks_at_9am(instruments, target_date):
    """
    Scan all NSE stocks at 9:00 AM and create monitoring list.
    Find stocks where current price < all 8 MAs.
    
    Args:
        instruments: List of instrument dictionaries from Zerodha
        target_date: Date to analyze
        
    Returns:
        list: Monitoring list of stocks that are below all 8 MAs
    """
    monitoring_list = []
    total_scanned = 0
    
    print(f"\n🔍 9:00 AM SCAN for {target_date}")
    print("📊 Finding stocks where current price < ALL 8 Moving Averages...")
    
    for i, instrument in enumerate(instruments, start=1):
        try:
            symbol = instrument['tradingsymbol']
            company_name = instrument['name']
            token = instrument['instrument_token']
            
            # Fetch daily data (last 300 days for MA calculation)
            daily_from_date = target_date - timedelta(days=300)
            daily_data = fetch_historical_data_with_retry(
                token=token,
                from_date=daily_from_date,
                to_date=target_date,
                interval="day"
            )
            
            if not daily_data:
                continue
            
            # Fetch hourly data (last 60 days for 200h MA calculation)
            hourly_from_date = target_date - timedelta(days=60)
            hourly_data = fetch_historical_data_with_retry(
                token=token,
                from_date=hourly_from_date,
                to_date=target_date,
                interval="hour"
            )
            
            if not hourly_data:
                continue
            
            # Calculate MAs at 9:00 AM
            analysis_9am = calculate_moving_averages_at_9am(daily_data, hourly_data, target_date)
            
            if not analysis_9am:
                continue
            
            total_scanned += 1
            
            # Check if stock price is below all 8 MAs at 9:00 AM
            if is_below_all_8_mas(analysis_9am):
                monitoring_list.append({
                    'symbol': symbol,
                    'company_name': company_name,
                    'token': token,
                    'price_at_9am': analysis_9am['current_price'],
                    'mas': {
                        'ma_44': analysis_9am['ma_44'],
                        'ma_50': analysis_9am['ma_50'],
                        'ma_100': analysis_9am['ma_100'],
                        'ma_200': analysis_9am['ma_200'],
                        'ma_44h': analysis_9am['ma_44h'],
                        'ma_50h': analysis_9am['ma_50h'],
                        'ma_100h': analysis_9am['ma_100h'],
                        'ma_200h': analysis_9am['ma_200h']
                    },
                    'daily_data': daily_data,    # Store for volume calculation later
                    'hourly_data': hourly_data   # Store for volume calculation later
                })
            
            # Progress update every 100 stocks
            if i % 200 == 0:
                print(f"⏳ Scanned {i}/{len(instruments)} stocks... Found {len(monitoring_list)} candidates")
                logger.info(f"9AM scan progress: {i}/{len(instruments)}")
            
            # Rate limiting to avoid API limits
            if i % 10 == 0:
                t.sleep(0.1)  # Small delay every 10 API calls
                
        except Exception as e:
            logger.error(f"Error in 9AM scan for {instrument.get('tradingsymbol', 'unknown')}: {e}")
            continue
    
    print(f"✅ 9:00 AM SCAN COMPLETE for {target_date}")
    print(f"📊 Total stocks scanned: {total_scanned}")
    print(f"🎯 Stocks added to monitoring list: {len(monitoring_list)}")
    
    return monitoring_list

def get_minute_data_for_day(token, target_date):
    """
    Get minute-by-minute data for the entire trading day.
    
    Args:
        token: Instrument token
        target_date: Date to get data for
        
    Returns:
        list: Minute-wise OHLCV data for the day
    """
    # Market timing: 9:15 AM to 3:30 PM
    from_datetime = datetime.combine(target_date, time(9, 15))  # Market opens
    to_datetime = datetime.combine(target_date, time(15, 30))   # Market closes
    
    minute_data = fetch_historical_data_with_retry(
        token=token,
        from_date=from_datetime,
        to_date=to_datetime,
        interval="minute"
    )
    
    return minute_data

def monitor_for_breakouts(monitoring_list, target_date):
    """
    Monitor stocks throughout the trading day for 8-MA breakouts.
    A breakout occurs when highest price during the day > ALL 8 MAs.
    
    Args:
        monitoring_list: List of stocks to monitor
        target_date: Date to monitor
        
    Returns:
        list: List of stocks that broke out with detailed information
    """
    breakout_stocks = []
    
    if not monitoring_list:
        print(f"🔭 No stocks to monitor for {target_date}")
        return breakout_stocks
    
    print(f"\n⚡ INTRADAY BREAKOUT MONITORING for {target_date}")
    print(f"🎯 Monitoring {len(monitoring_list)} stocks for 8-MA breakouts...")
    print("📈 Breakout condition: Highest price during day > ALL 8 MAs")
    
    for i, stock in enumerate(monitoring_list, start=1):
        try:
            symbol = stock['symbol']
            token = stock['token']
            company_name = stock['company_name']
            price_at_9am = stock['price_at_9am']
            mas = stock['mas']
            
            # Get minute-by-minute data for the day
            minute_data = get_minute_data_for_day(token, target_date)
            
            if not minute_data:
                continue
            
            # Find the highest price during the day
            highest_price = max([candle['high'] for candle in minute_data])
            
            # Check if highest price is above all 8 MAs
            above_all_mas = all([
                highest_price > mas['ma_44'] if mas['ma_44'] else False,
                highest_price > mas['ma_50'] if mas['ma_50'] else False,
                highest_price > mas['ma_100'] if mas['ma_100'] else False,
                highest_price > mas['ma_200'] if mas['ma_200'] else False,
                highest_price > mas['ma_44h'] if mas['ma_44h'] else False,
                highest_price > mas['ma_50h'] if mas['ma_50h'] else False,
                highest_price > mas['ma_100h'] if mas['ma_100h'] else False,
                highest_price > mas['ma_200h'] if mas['ma_200h'] else False
            ])
            
            if above_all_mas:
                # Find the exact time when breakout occurred
                breakout_time = None
                breakout_price = None
                
                # Check each minute to find when price first crossed above all MAs
                for candle in minute_data:
                    candle_high = candle['high']
                    candle_time = pd.to_datetime(candle['date']).time()
                    
                    # Check if this candle's high crossed above all MAs
                    if all([
                        candle_high > mas['ma_44'] if mas['ma_44'] else False,
                        candle_high > mas['ma_50'] if mas['ma_50'] else False,
                        candle_high > mas['ma_100'] if mas['ma_100'] else False,
                        candle_high > mas['ma_200'] if mas['ma_200'] else False,
                        candle_high > mas['ma_44h'] if mas['ma_44h'] else False,
                        candle_high > mas['ma_50h'] if mas['ma_50h'] else False,
                        candle_high > mas['ma_100h'] if mas['ma_100h'] else False,
                        candle_high > mas['ma_200h'] if mas['ma_200h'] else False
                    ]):
                        breakout_time = candle_time
                        breakout_price = candle_high
                        break
                
                # Calculate volume averages at breakout time
                volume_data = calculate_volume_averages(
                    stock['daily_data'], 
                    stock['hourly_data'], 
                    target_date, 
                    breakout_time
                )
                
                # Create breakout record
                breakout_record = {
                    'date': target_date,
                    'symbol': symbol,
                    'company_name': company_name,
                    'price_at_9am': price_at_9am,
                    'breakout_time': breakout_time,
                    'breakout_price': breakout_price,
                    'highest_price': highest_price,
                    'price_change': highest_price - price_at_9am,
                    'percentage_change': ((highest_price - price_at_9am) / price_at_9am * 100),
                    'mas': mas,
                    'volume_analysis': volume_data
                }
                
                breakout_stocks.append(breakout_record)
                
                # Trigger breakout alert
                trigger_breakout_alert(breakout_record)
            
            # Progress update
            if i % 10 == 0:
                print(f"⏳ Monitored {i}/{len(monitoring_list)} stocks... Breakouts found: {len(breakout_stocks)}")
                
        except Exception as e:
            logger.error(f"Error monitoring {stock['symbol']} on {target_date}: {e}")
            continue
    
    print(f"✅ BREAKOUT MONITORING COMPLETE for {target_date}")
    print(f"🚀 Total breakouts detected: {len(breakout_stocks)}")
    
    return breakout_stocks

def trigger_breakout_alert(breakout):
    """
    Display detailed breakout alert in terminal with volume analysis.
    
    Args:
        breakout: Dictionary containing breakout information
    """
    vol_data = breakout['volume_analysis']
    
    alert_message = f"""
🚀🚀🚀 *** 8-MA BREAKOUT DETECTED *** 🚀🚀🚀
📅 Date: {breakout['date']}
📈 Stock: {breakout['symbol']} ({breakout['company_name']})

⏰ TIMELINE:
9:00 AM Price: ₹{breakout['price_at_9am']:.2f} (BELOW all 8 MAs)
{breakout['breakout_time']} Breakout: ₹{breakout['breakout_price']:.2f} (ABOVE all 8 MAs)
Highest Price: ₹{breakout['highest_price']:.2f}

📊 8 MOVING AVERAGES:
Daily MAs:  MA44=₹{breakout['mas']['ma_44']:.2f}, MA50=₹{breakout['mas']['ma_50']:.2f}, MA100=₹{breakout['mas']['ma_100']:.2f}, MA200=₹{breakout['mas']['ma_200']:.2f}
Hourly MAs: MA44h=₹{breakout['mas']['ma_44h']:.2f}, MA50h=₹{breakout['mas']['ma_50h']:.2f}, MA100h=₹{breakout['mas']['ma_100h']:.2f}, MA200h=₹{breakout['mas']['ma_200h']:.2f}

💰 PERFORMANCE:
Price Change: ₹{breakout['price_change']:+.2f}
Percentage Change: {breakout['percentage_change']:+.2f}%

📊 VOLUME ANALYSIS AT BREAKOUT:
Daily Volume Averages:
  - 44 days: {vol_data.get('daily_volume_avg_44', 'N/A'):,.0f} shares
  - 50 days: {vol_data.get('daily_volume_avg_50', 'N/A'):,.0f} shares
  - 100 days: {vol_data.get('daily_volume_avg_100', 'N/A'):,.0f} shares
  - 200 days: {vol_data.get('daily_volume_avg_200', 'N/A'):,.0f} shares

Hourly Volume Averages:
  - 44 hours: {vol_data.get('hourly_volume_avg_44h', 'N/A'):,.0f} shares
  - 50 hours: {vol_data.get('hourly_volume_avg_50h', 'N/A'):,.0f} shares
  - 100 hours: {vol_data.get('hourly_volume_avg_100h', 'N/A'):,.0f} shares
  - 200 hours: {vol_data.get('hourly_volume_avg_200h', 'N/A'):,.0f} shares

🎯 BREAKOUT TIME: {breakout['breakout_time']}
🚀🚀🚀 SUCCESSFULLY BROKE ABOVE ALL 8 MOVING AVERAGES! 🚀🚀🚀
"""
    
    print("=" * 120)
    print(alert_message)
    print("=" * 120)
    
    # Also log the breakout
    logger.info(f"BREAKOUT: {breakout['symbol']} on {breakout['date']} at {breakout['breakout_time']} - "
                f"₹{breakout['price_at_9am']:.2f} → ₹{breakout['breakout_price']:.2f} "
                f"({breakout['percentage_change']:+.2f}%)")

def analyze_single_day_breakouts():
    """
    Main function to analyze 8-MA breakouts for a single trading day.
    Orchestrates the entire flow from user input to final results.
    
    Returns:
        list: List of breakout stocks found
    """
    print("🚀 SINGLE DAY INTRADAY 8-MA BREAKOUT ANALYZER")
    print("🎯 Strategy: Find stocks below 8 MAs at 9 AM → Monitor for intraday breakouts")
    print("⚡ 8 Moving Averages: MA44, MA50, MA100, MA200 (Daily) + MA44h, MA50h, MA100h, MA200h (Hourly)")
    print("📈 Breakout Condition: Highest price during day > ALL 8 MAs")
    print("=" * 120)
    
    # Step 1: Get valid trading date from user
    target_date = get_user_input_date()
    
    # Step 2: Fetch all NSE instruments
    print(f"\n📡 Fetching NSE instruments...")
    instruments = fetch_nse_instruments()
    if not instruments:
        print("❌ No instruments found. Exiting.")
        return []
    
    # Optional: Limit instruments for testing (uncomment for faster testing)
    # instruments = instruments[:100]  # Test with first 100 stocks only
    
    print(f"✅ Found {len(instruments)} NSE equity instruments to analyze")
    
    try:
        # Step 3: 9:00 AM Scan - Find stocks below all 8 MAs
        monitoring_list = scan_stocks_at_9am(instruments, target_date)
        
        if not monitoring_list:
            print(f"\n🔭 No stocks found below all 8 MAs at 9:00 AM on {target_date}")
            print("💡 This could mean:")
            print("   - Market was in strong bullish mode")
            print("   - Most stocks were already above their moving averages")
            print("   - Try analyzing a different date")
            return []
        
        """# Display monitoring list
        print(f"\n📋 MONITORING LIST for {target_date}:")
        print("-" * 100)
        print(f"{'SYMBOL':<15} | {'COMPANY':<30} | {'9AM PRICE':<10} | {'HIGHEST MA':<10}")
        print("-" * 100)"""
        
        for stock in monitoring_list[:20]:  # Show first 20 stocks
            highest_ma = max([ma for ma in stock['mas'].values() if ma is not None])
            print(f"{stock['symbol']:<15} | {stock['company_name'][:30]:<30} | "
                  f"₹{stock['price_at_9am']:<9.2f} | ₹{highest_ma:<9.2f}")
        
        if len(monitoring_list) > 20:
            print(f"... and {len(monitoring_list) - 20} more stocks")
        
        # Step 4: Monitor for breakouts throughout the day
        breakout_stocks = monitor_for_breakouts(monitoring_list, target_date)
        
        # Step 5: Final Summary
        print("\n" + "="*120)
        print("🎯 SINGLE DAY ANALYSIS COMPLETE")
        print("="*120)
        
        print(f"\n📊 SUMMARY for {target_date}:")
        print(f"Total stocks analyzed: {len(instruments)}")
        print(f"Stocks below 8 MAs at 9 AM: {len(monitoring_list)}")
        print(f"Breakouts detected: {len(breakout_stocks)}")
        
        if breakout_stocks:
            print(f"\n🚀 BREAKOUT DETAILS:")
            print("-" * 120)
            print(f"{'SYMBOL':<12} | {'BREAKOUT TIME':<12} | {'9AM PRICE':<10} | {'BREAKOUT PRICE':<14} | {'HIGHEST PRICE':<13} | {'CHANGE %':<8}")
            print("-" * 120)
            
            # Sort by percentage change (highest first)
            sorted_breakouts = sorted(breakout_stocks, key=lambda x: x['percentage_change'], reverse=True)
            
            for breakout in sorted_breakouts:
                print(f"{breakout['symbol']:<12} | {breakout['breakout_time']:<12} | "
                      f"₹{breakout['price_at_9am']:<9.2f} | ₹{breakout['breakout_price']:<13.2f} | "
                      f"₹{breakout['highest_price']:<12.2f} | {breakout['percentage_change']:>+7.2f}%")
            
            # Save results to file
            results_file = f"breakouts_{target_date.strftime('%Y%m%d')}.json"
            json_breakouts = []
            
            for breakout in breakout_stocks:
                json_breakout = breakout.copy()
                json_breakout['date'] = str(json_breakout['date'])
                json_breakout['breakout_time'] = str(json_breakout['breakout_time'])
                json_breakouts.append(json_breakout)
            
            with open(results_file, 'w') as f:
                json.dump(json_breakouts, f, indent=2, default=str)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
            
        else:
            print(f"\n🔭 No breakouts detected on {target_date}")
            print("💡 Possible reasons:")
            print("   - Market was bearish/sideways")
            print("   - Stocks didn't gain enough momentum")
            print("   - Strong resistance at MA levels")
        
        return breakout_stocks
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        print(f"❌ Analysis failed: {e}")
        return []

if __name__ == "__main__":
    """
    Main entry point of the program.
    Handles user interruption and provides clean exit.
    """
    try:
        print("🚀 STARTING SINGLE DAY 8-MA BREAKOUT ANALYSIS")
        print("-" * 120)
        
        # Run the analysis
        breakout_results = analyze_single_day_breakouts()
        
        if breakout_results:
            print(f"\n✅ Analysis completed successfully!")
            print(f"🎯 Found {len(breakout_results)} breakout opportunities")
        else:
            print(f"\n📊 Analysis completed - no breakouts found")
            
    except KeyboardInterrupt:
        print("\n🛑 Analysis stopped by user (Ctrl+C pressed)")
        logger.info("Analysis stopped by user.")
    except Exception as e:
        print(f"\n❌ Fatal error occurred: {e}")
        logger.error(f"Fatal error: {e}")
        print("💡 Please check your API credentials and internet connection")
    
    finally:
        print("\n👋 Thank you for using the 8-MA Breakout Analyzer!")
        print("💡 For questions or support, check the log file for detailed information.")