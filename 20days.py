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
        logging.FileHandler('intraday_8ma_breakout_monitor.log', encoding='utf-8'),
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
    """Check if date is a weekday (Mon-Fri)"""
    return date.weekday() < 5

def get_last_n_trading_days(n_days=20):
    """Get last N trading days"""
    trading_days = []
    current_date = datetime.now().date()
    days_back = 0
    
    while len(trading_days) < n_days:
        check_date = current_date - timedelta(days=days_back)
        if is_trading_day(check_date):
            trading_days.append(check_date)
        days_back += 1
    
    return list(reversed(trading_days))  # Return in chronological order

def fetch_historical_data_with_retry(token, from_date, to_date, interval="day", max_retries=3):
    """Fetch historical data with retry logic for timeout handling"""
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
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
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
    """Fetch all active NSE equity instruments"""
    try:
        instruments_raw = kite.instruments("NSE")
        logger.info(f"Total NSE instruments fetched: {len(instruments_raw)}")
        
        # Filter: only main-board NSE equities
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

def calculate_moving_averages_at_time(daily_data, hourly_data, target_date, target_time=None):
    """Calculate moving averages at a specific date and time (9:00 AM or any time)"""
    if not daily_data or not hourly_data:
        return None
    
    # Calculate Daily MAs up to target date
    daily_df = pd.DataFrame(daily_data)
    daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
    daily_filtered = daily_df[daily_df['date'] <= target_date].tail(200).reset_index(drop=True)
    
    if len(daily_filtered) < 200:
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
    
    # Calculate Hourly MAs
    hourly_df = pd.DataFrame(hourly_data)
    hourly_df['datetime'] = pd.to_datetime(hourly_df['date'])
    hourly_df['date_only'] = hourly_df['datetime'].dt.date
    hourly_df['time_only'] = hourly_df['datetime'].dt.time
    
    # Filter hourly data up to target date and time
    if target_time:
        hourly_filtered = hourly_df[
            (hourly_df['date_only'] < target_date) | 
            ((hourly_df['date_only'] == target_date) & (hourly_df['time_only'] <= target_time))
        ].tail(200).reset_index(drop=True)
    else:
        hourly_filtered = hourly_df[hourly_df['date_only'] <= target_date].tail(200).reset_index(drop=True)
    
    if len(hourly_filtered) < 200:
        return None
    
    # Calculate hourly moving averages
    hourly_filtered['ma_44h'] = hourly_filtered['close'].rolling(window=44, min_periods=44).mean().round(2)
    hourly_filtered['ma_50h'] = hourly_filtered['close'].rolling(window=50, min_periods=50).mean().round(2)
    hourly_filtered['ma_100h'] = hourly_filtered['close'].rolling(window=100, min_periods=100).mean().round(2)
    hourly_filtered['ma_200h'] = hourly_filtered['close'].rolling(window=200, min_periods=200).mean().round(2)
    
    # Get current price from latest hourly candle
    if target_time:
        current_price_data = hourly_filtered[
            (hourly_filtered['date_only'] == target_date) & 
            (hourly_filtered['time_only'] <= target_time)
        ]
    else:
        current_price_data = hourly_filtered[hourly_filtered['date_only'] == target_date]
    
    if current_price_data.empty:
        return None
    
    latest_hourly = current_price_data.iloc[-1]
    
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
    """Check if current price is BELOW all 8 moving averages"""
    if not analysis_data:
        return False
    
    mas = ['ma_44', 'ma_50', 'ma_100', 'ma_200', 'ma_44h', 'ma_50h', 'ma_100h', 'ma_200h']
    
    for ma in mas:
        if analysis_data[ma] is None or analysis_data['current_price'] > analysis_data[ma]:
            return False
    
    return True

def is_above_all_8_mas(analysis_data):
    """Check if current price is ABOVE all 8 moving averages"""
    if not analysis_data:
        return False
    
    mas = ['ma_44', 'ma_50', 'ma_100', 'ma_200', 'ma_44h', 'ma_50h', 'ma_100h', 'ma_200h']
    
    for ma in mas:
        if analysis_data[ma] is None or analysis_data['current_price'] <= analysis_data[ma]:
            return False
    
    return True

def get_minute_data_for_day(token, target_date):
    """Get minute-by-minute data for a specific day"""
    from_datetime = datetime.combine(target_date, time(9, 15))  # Market opens at 9:15 AM
    to_datetime = datetime.combine(target_date, time(15, 30))   # Market closes at 3:30 PM
    
    minute_data = fetch_historical_data_with_retry(
        token=token,
        from_date=from_datetime,
        to_date=to_datetime,
        interval="minute"
    )
    
    return minute_data

def scan_stocks_at_9am(instruments, target_date):
    """Scan all stocks at 9:00 AM and find those below all 8 MAs"""
    monitoring_list = []
    total_scanned = 0
    
    print(f"\n🔍 9:00 AM SCAN for {target_date}")
    print("📊 Finding stocks below ALL 8 Moving Averages...")
    
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
            
            # Fetch hourly data (last 60 days for 200h MA)
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
            morning_time = time(9, 15)  # Market opens at 9:15 AM, use this as 9 AM proxy
            analysis_9am = calculate_moving_averages_at_time(daily_data, hourly_data, target_date, morning_time)
            
            if not analysis_9am:
                continue
            
            total_scanned += 1
            
            # Check if stock is below all 8 MAs at 9:00 AM
            if is_below_all_8_mas(analysis_9am):
                monitoring_list.append({
                    'symbol': symbol,
                    'company_name': company_name,
                    'token': token,
                    'morning_price': analysis_9am['current_price'],
                    'mas': {
                        'ma_44': analysis_9am['ma_44'],
                        'ma_50': analysis_9am['ma_50'],
                        'ma_100': analysis_9am['ma_100'],
                        'ma_200': analysis_9am['ma_200'],
                        'ma_44h': analysis_9am['ma_44h'],
                        'ma_50h': analysis_9am['ma_50h'],
                        'ma_100h': analysis_9am['ma_100h'],
                        'ma_200h': analysis_9am['ma_200h']
                    }
                })
            
            # Progress update
            if i % 100 == 0:
                print(f"⏳ Scanned {i}/{len(instruments)} stocks... Found {len(monitoring_list)} candidates")
                logger.info(f"9AM scan progress: {i}/{len(instruments)}")
            
            # Rate limiting
            if i % 10 == 0:
                t.sleep(0.1)  # Reduced from 0.3 to 0.1 seconds
                
        except Exception as e:
            logger.error(f"Error in 9AM scan for {instrument.get('tradingsymbol', 'unknown')}: {e}")
            continue
    
    print(f"✅ 9:00 AM SCAN COMPLETE for {target_date}")
    print(f"📊 Total stocks scanned: {total_scanned}")
    print(f"🎯 Stocks added to monitoring list: {len(monitoring_list)}")
    
    return monitoring_list

def monitor_intraday_breakouts(monitoring_list, target_date):
    """Monitor stocks throughout the day for 8-MA breakouts with optimized MA calculation"""
    breakout_stocks = []
    
    if not monitoring_list:
        print(f"📭 No stocks to monitor for {target_date}")
        return breakout_stocks
    
    print(f"\n⚡ INTRADAY MONITORING for {target_date}")
    print(f"🎯 Monitoring {len(monitoring_list)} stocks for 8-MA breakouts...")
    
    for stock in monitoring_list:
        try:
            symbol = stock['symbol']
            token = stock['token']
            company_name = stock['company_name']
            morning_price = stock['morning_price']
            morning_mas = stock['mas']  # MAs at 9 AM
            
            # Get minute data for the entire day
            minute_data = get_minute_data_for_day(token, target_date)
            
            if not minute_data:
                continue
            
            # Convert to DataFrame for easier processing
            minute_df = pd.DataFrame(minute_data)
            minute_df['datetime'] = pd.to_datetime(minute_df['date'])
            minute_df['time'] = minute_df['datetime'].dt.time
            
            # Check each minute after 9:15 AM for breakout
            breakout_found = False
            for _, row in minute_df.iterrows():
                current_time = row['time']
                current_price = row['close']
                
                # Skip if before 9:15 AM
                if current_time < time(9, 15):
                    continue
                
                # Use morning MAs for initial check (conservative approach)
                # This ensures we only catch TRUE breakouts where price moves significantly above MAs
                above_all_mas = all([
                    current_price > morning_mas['ma_44'] if morning_mas['ma_44'] else False,
                    current_price > morning_mas['ma_50'] if morning_mas['ma_50'] else False,
                    current_price > morning_mas['ma_100'] if morning_mas['ma_100'] else False,
                    current_price > morning_mas['ma_200'] if morning_mas['ma_200'] else False,
                    current_price > morning_mas['ma_44h'] if morning_mas['ma_44h'] else False,
                    current_price > morning_mas['ma_50h'] if morning_mas['ma_50h'] else False,
                    current_price > morning_mas['ma_100h'] if morning_mas['ma_100h'] else False,
                    current_price > morning_mas['ma_200h'] if morning_mas['ma_200h'] else False
                ])
                
                if above_all_mas:
                    # Breakout detected!
                    breakout_time = current_time
                    breakout_price = current_price
                    
                    breakout_stocks.append({
                        'date': target_date,
                        'symbol': symbol,
                        'company_name': company_name,
                        'morning_price': morning_price,
                        'breakout_time': breakout_time,
                        'breakout_price': breakout_price,
                        'price_change': breakout_price - morning_price,
                        'percentage_change': ((breakout_price - morning_price) / morning_price * 100),
                        'mas': morning_mas
                    })
                    
                    print(f"🚀 BREAKOUT DETECTED: {symbol} at {breakout_time} - Price: ₹{breakout_price:.2f} (+{((breakout_price - morning_price) / morning_price * 100):.2f}%)")
                    logger.info(f"Breakout: {symbol} on {target_date} at {breakout_time} - ₹{morning_price:.2f} → ₹{breakout_price:.2f}")
                    
                    breakout_found = True
                    break  # Stop monitoring this stock once breakout is detected
                    
        except Exception as e:
            logger.error(f"Error monitoring {stock['symbol']} on {target_date}: {e}")
            continue
    
    print(f"✅ INTRADAY MONITORING COMPLETE for {target_date}")
    print(f"🚀 Breakouts detected: {len(breakout_stocks)}")
    
    return breakout_stocks

def trigger_breakout_notification(breakout):
    """Display detailed breakout notification"""
    message = f"""
🚀🚀🚀 *** INTRADAY 8-MA BREAKOUT ALERT *** 🚀🚀🚀
Date: {breakout['date']}
Stock: {breakout['symbol']} ({breakout['company_name']})

⏰ TIMELINE:
9:00 AM Price: ₹{breakout['morning_price']:.2f} (BELOW all 8 MAs)
{breakout['breakout_time']} Price: ₹{breakout['breakout_price']:.2f} (ABOVE all 8 MAs)

📈 8 MOVING AVERAGES:
Daily MAs:  MA44=₹{breakout['mas']['ma_44']:.2f}, MA50=₹{breakout['mas']['ma_50']:.2f}, MA100=₹{breakout['mas']['ma_100']:.2f}, MA200=₹{breakout['mas']['ma_200']:.2f}
Hourly MAs: MA44h=₹{breakout['mas']['ma_44h']:.2f}, MA50h=₹{breakout['mas']['ma_50h']:.2f}, MA100h=₹{breakout['mas']['ma_100h']:.2f}, MA200h=₹{breakout['mas']['ma_200h']:.2f}

💰 PERFORMANCE:
Price Change: ₹{breakout['price_change']:+.2f}
Percentage Change: {breakout['percentage_change']:+.2f}%

🎯 BREAKOUT TIME: {breakout['breakout_time']}
🚀🚀🚀 SUCCESSFULLY BROKE ABOVE ALL 8 MOVING AVERAGES! 🚀🚀🚀
"""
    
    print("=" * 120)
    print(message)
    print("=" * 120)

def analyze_20_day_8ma_breakouts():
    """Main function to analyze 8-MA breakouts over last 20 trading days"""
    print("🚀 STARTING 20-DAY INTRADAY 8-MA BREAKOUT ANALYSIS")
    print("🎯 Strategy: Find stocks below 8 MAs at 9 AM → Monitor for intraday breakouts")
    print("⚡ 8 Moving Averages: MA44, MA50, MA100, MA200 (Daily) + MA44h, MA50h, MA100h, MA200h (Hourly)")
    print("=" * 120)
    
    # Get last 20 trading days
    trading_days = get_last_n_trading_days(20)
    print(f"📅 Analyzing {len(trading_days)} trading days: {trading_days[0]} to {trading_days[-1]}")
    
    # Fetch instruments once
    instruments = fetch_nse_instruments()
    if not instruments:
        logger.error("No instruments found. Exiting.")
        return []
    
    # Limit instruments for testing/faster execution (uncomment next line for testing)
    # instruments = instruments[:500]  # Test with first 500 stocks only
    
    all_breakouts = []
    daily_summary = []
    global_breakout_stocks = set()  # Track stocks that have already broken out
    
    for day_num, target_date in enumerate(trading_days, start=1):
        print(f"\n" + "="*120)
        print(f"📅 DAY {day_num}/20: ANALYZING {target_date}")
        print("="*120)
        
        try:
            # Step 1: 9:00 AM Scan - but exclude stocks that already broke out
            all_monitoring_candidates = scan_stocks_at_9am(instruments, target_date)
            
            # Filter out stocks that have already broken out in previous days
            monitoring_list = []
            for stock in all_monitoring_candidates:
                if stock['symbol'] not in global_breakout_stocks:
                    monitoring_list.append(stock)
                else:
                    logger.info(f"Skipping {stock['symbol']} - already broke out in previous days")
            
            if len(all_monitoring_candidates) != len(monitoring_list):
                print(f"📋 Filtered out {len(all_monitoring_candidates) - len(monitoring_list)} stocks that already broke out")
            
            # Step 2: Intraday Monitoring
            daily_breakouts = monitor_intraday_breakouts(monitoring_list, target_date)
            
            # Step 3: Add newly broken out stocks to global tracking
            for breakout in daily_breakouts:
                global_breakout_stocks.add(breakout['symbol'])
            
            # Step 4: Record results
            daily_summary.append({
                'date': target_date,
                'candidates_found': len(all_monitoring_candidates),
                'stocks_monitored': len(monitoring_list),
                'breakouts_found': len(daily_breakouts)
            })
            
            all_breakouts.extend(daily_breakouts)
            
            # Display daily breakouts
            if daily_breakouts:
                print(f"\n🚀 {len(daily_breakouts)} BREAKOUTS FOUND on {target_date}:")
                print("-" * 80)
                for breakout in daily_breakouts:
                    print(f"  {breakout['symbol']:<12} | {breakout['breakout_time']} | "
                          f"₹{breakout['morning_price']:.2f} → ₹{breakout['breakout_price']:.2f} "
                          f"({breakout['percentage_change']:+.2f}%)")
                    
                    # Trigger detailed notification for significant breakouts
                    if breakout['percentage_change'] > 2.0:  # More than 2% gain
                        trigger_breakout_notification(breakout)
            else:
                print(f"📭 No breakouts found on {target_date}")
            
            print(f"📊 Daily Summary: {len(all_monitoring_candidates)} candidates → {len(monitoring_list)} monitored → {len(daily_breakouts)} breakouts")
            
        except Exception as e:
            logger.error(f"Error analyzing {target_date}: {e}")
            continue
    
    # Final Summary
    print("\n" + "="*120)
    print("🎯 20-DAY INTRADAY 8-MA BREAKOUT ANALYSIS COMPLETE")
    print("="*120)
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"Total breakouts found: {len(all_breakouts)}")
    print(f"Unique stocks that broke out: {len(global_breakout_stocks)}")
    if trading_days:
        print(f"Average breakouts per day: {len(all_breakouts)/len(trading_days):.1f}")
    
    if all_breakouts:
        print(f"\n📋 TOP PERFORMERS (by percentage gain):")
        print("-" * 100)
        print(f"{'DATE':<12} | {'SYMBOL':<12} | {'TIME':<8} | {'FROM':<8} | {'TO':<8} | {'CHANGE %':<8}")
        print("-" * 100)
        
        # Sort by percentage change
        top_performers = sorted(all_breakouts, key=lambda x: x['percentage_change'], reverse=True)[:20]
        
        for breakout in top_performers:
            print(f"{breakout['date']} | {breakout['symbol']:<12} | {breakout['breakout_time']} | "
                  f"₹{breakout['morning_price']:<7.2f} | ₹{breakout['breakout_price']:<7.2f} | "
                  f"{breakout['percentage_change']:>+7.2f}%")
        
        print("\n📈 DAILY BREAKDOWN:")
        print("-" * 80)
        print(f"{'DATE':<12} | {'CANDIDATES':<10} | {'MONITORED':<10} | {'BREAKOUTS':<10}")
        print("-" * 80)
        
        for summary in daily_summary:
            print(f"{summary['date']} | {summary['candidates_found']:<10} | {summary['stocks_monitored']:<10} | {summary['breakouts_found']:<10}")
    
    else:
        print("📭 No breakouts found in the analyzed period")
    
    logger.info(f"20-day analysis complete. Found {len(all_breakouts)} total breakouts from {len(global_breakout_stocks)} unique stocks.")
    return all_breakouts

if __name__ == "__main__":
    try:
        print("🚀 STARTING COMPREHENSIVE INTRADAY 8-MA BREAKOUT ANALYSIS")
        print("🎯 This will analyze the last 20 trading days for intraday breakout patterns")
        print("⚠️  This is a comprehensive analysis and may take 30-60 minutes to complete")
        print("-" * 120)
        
        # Run the comprehensive analysis
        all_breakouts = analyze_20_day_8ma_breakouts()
        
        # Save results to JSON file
        if all_breakouts:
            results_file = f"8ma_breakouts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert datetime objects to strings for JSON serialization
            json_breakouts = []
            for breakout in all_breakouts:
                json_breakout = breakout.copy()
                json_breakout['date'] = str(json_breakout['date'])
                json_breakout['breakout_time'] = str(json_breakout['breakout_time'])
                json_breakouts.append(json_breakout)
            
            with open(results_file, 'w') as f:
                json.dump(json_breakouts, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("🛑 Analysis stopped by user.")
        logger.info("Analysis stopped by user.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")