import datetime
import time
import threading
import pandas as pd
import gspread
from common import kite, logger, next_price_above, load_token_map, token_map
from build_monitor_list import main as run_monitor_builder
from minute_monitor_stocks import get_stocks_by_tier, run_breakout_check, ARMED_SYMBOLS
from monitor_breakouts import process_breakout, get_monitor_list
from live_paper_trader import ACTIVE_EXIT_MONITORS, finalize_trade


# --- Google Sheets Setup ---
SHEET_LOCK = threading.Lock()

try:
    gc = gspread.service_account(filename='credentials.json')
    sh = gc.open("Live_Paper_Trading")
    
    # Get Sheet8 directly by name
    sheet = sh.worksheet("Sheet8")
    
    logger.info(f"✅ Connected to Google Sheet: {sheet.title}")
except Exception as e:
    sheet = None
    logger.error(f"❌ Google Sheets connection failed: {e}")


# Track armed times for real money trading
ARMED_TIMES_REAL = {}

def update_sheet_on_breakout(symbol, breakout_time, armed_time, buy_price, breakout_price, quantity):
    """
    Updates Google Sheet when a breakout buy order executes.
    Fills columns A through I.
    """
    if sheet is None:
        logger.error(f"❌ Sheet not connected. Cannot log {symbol}")
        return False
    
    money_traded = round(quantity * buy_price, 2)
    
    row = [
        breakout_time,      # A: Breakout Time
        symbol,             # B: Symbol
        armed_time,         # C: Armed Time
        breakout_time,      # D: Buy Time (same as breakout for SL-L orders)
        buy_price,          # E: Buy Price
        breakout_price,     # F: Breakout Price
        quantity,           # G: Quantity
        money_traded,       # H: Money Traded
        "OPEN",             # I: Status
        "",                 # J: Selling Price (empty, filled on exit)
        "",                 # K: Exit Reason (empty, filled on exit)
        ""                  # L: P&L % (empty, filled on exit)
    ]
    
    try:
        with SHEET_LOCK:
            sheet.append_row(row)
        logger.info(f"✅ Sheet updated for {symbol} | Buy={buy_price} | Qty={quantity} | Money=₹{money_traded}")
        return True
    except Exception as e:
        logger.error(f"❌ Sheet update failed for {symbol}: {e}")
        return False


def update_sheet_on_exit(symbol, selling_price, exit_reason):
    """
    Finds the OPEN trade row for this symbol and fills exit details.
    Updates columns I, J, K, L.
    """
    if sheet is None:
        logger.error(f"❌ Sheet not connected. Cannot update exit for {symbol}")
        return False
    
    try:
        all_rows = sheet.get_all_values()
        
        for i, row in enumerate(all_rows[1:], start=2):  # Skip header, gspread is 1-indexed
            # Find row where Symbol matches AND Status is OPEN
            if len(row) >= 9 and row[1] == symbol and row[8].strip() == "OPEN":
                buy_price = float(row[4])  # Column E
                pnl_pct = round(((selling_price - buy_price) / buy_price) * 100, 2)
                
                with SHEET_LOCK:
                    sheet.update_cell(i, 9, "CLOSED")                # I: Status
                    sheet.update_cell(i, 10, selling_price)          # J: Selling Price
                    sheet.update_cell(i, 11, exit_reason)            # K: Exit Reason
                    sheet.update_cell(i, 12, pnl_pct / 100)          # L: P&L %
                
                logger.info(f"✅ Sheet exit updated for {symbol} | Sold={selling_price} | Reason={exit_reason} | P&L={pnl_pct}%")
                return True
        
        logger.warning(f"⚠️ No OPEN trade found for {symbol} in sheet")
        return False
        
    except Exception as e:
        logger.error(f"❌ Sheet exit update failed for {symbol}: {e}")
        return False

ARMED_SYMBOLS_REAL = set()
LOGGED_SKIPS = set()

# --- CONFIGURATION ---
LIVE_TRADING = True
TARGET_PCT = 1.03
EXIT_TIME = datetime.time(15, 15)
DAILY_BUDGET = 2400            # Total budget for the day
daily_spent = 0                # Tracks how much has been spent

# Time-based expected breakout slots
# Format: (start_time, end_time, expected_breakouts_remaining)
BREAKOUT_SLOTS = [
    (datetime.time(9, 15),  datetime.time(11, 0),  3),   # Morning: assume 3 more
    (datetime.time(11, 0),  datetime.time(13, 0),  2),   # Midday: assume 2 more
    (datetime.time(13, 0),  datetime.time(15, 30), 1),   # Afternoon: assume 1 more
]

def get_investment_for_current_trade():
    """
    Dynamically calculates how much to invest in the current trade
    based on remaining budget and time of day.
    
    Logic:
    - Divides remaining budget by expected remaining breakouts
    - Expected breakouts decreases as the day progresses
    - If budget already spent, returns 0
    """
    remaining_budget = DAILY_BUDGET - daily_spent
    
    # No budget left
    if remaining_budget <= 0:
        logger.warning(f"❌ Budget exhausted: ₹{daily_spent}/₹{DAILY_BUDGET}")
        return 0
    
    # Find current time slot
    now = datetime.datetime.now().time()
    expected_remaining = 1  # Default: assume at least 1 breakout
    
    for start, end, slots in BREAKOUT_SLOTS:
        if start <= now < end:
            expected_remaining = slots
            break
    
    # Calculate investment for this trade
    investment = remaining_budget / expected_remaining
    
    logger.info(
        f"💰 Budget Calc: "
        f"Remaining=₹{remaining_budget:.2f} | "
        f"Expected Slots={expected_remaining} | "
        f"This Trade=₹{investment:.2f}"
    )
    
    return investment

def get_live_allocation_qty(price):
    """
    Calculates quantity based on dynamic investment amount.
    Also verifies actual account balance before placing order.
    """
    
    try:
        # 1. Get dynamic investment amount for this trade
        investment = get_investment_for_current_trade()
        if investment <= 0:
            return 0
        
        # 2. Verify actual account balance
        margins = kite.margins()
        available_cash = margins["equity"]["available"]["live_balance"]
        
        # Use the LOWER of: dynamic investment OR available cash
        actual_investment = min(investment, available_cash)
        
        if actual_investment <= 0:
            logger.warning(f"❌ No funds available: Balance=₹{available_cash:.2f}")
            return 0
        
        # 3. Calculate quantity
        qty = int(actual_investment / price)

        # --- REFINED LOGIC TO IGNORE EXPENSIVE STOCKS ---
        if actual_investment < price:
            logger.info(f"⏭️ Ignoring: Stock price (₹{price}) exceeds allocated investment (₹{actual_investment:.2f})")
            return 0
        
        if qty <= 0:
            logger.warning(f"❌ Qty is 0: Price ₹{price} too high for ₹{actual_investment:.2f}")
            return 0
        
        # 4. Calculate real investment (qty * price)
        real_investment = qty * price
        
        # 5. Update daily spent
        
        logger.info(
            f"💰 Order Details: "
            f"Qty={qty} | "
            f"Price=₹{price} | "
            f"Investment=₹{real_investment:.2f} | "
            f"Daily: ₹{daily_spent:.2f}/₹{DAILY_BUDGET}"
        )
        
        return qty
        
    except Exception as e:
        logger.error(f"Error in get_live_allocation_qty: {e}")
        return 0

def place_real_order(symbol, trigger_price, tick_size):
    """
    Places SL-L order for real money trading.
    This is an ADVANCE order that executes when price hits trigger_price.
    """
    if not LIVE_TRADING:
        logger.info(f"[SIMULATION] ARM SL-L ORDER: {symbol} | Trigger={trigger_price} Limit={round(trigger_price + tick_size, 2)}")
        return "SIM_ID"

    limit_price = round(trigger_price + tick_size, 2)
    qty = get_live_allocation_qty(trigger_price)
    
    if qty <= 0:
        logger.error(f"❌ Cannot arm {symbol}: Insufficient funds for qty")
        return None

    try:
        # Check current LTP to ensure we're not already past trigger
        quote = kite.quote(f"NSE:{symbol}")
        ltp = quote[f"NSE:{symbol}"]["last_price"]
        
        if ltp >= trigger_price:
            logger.warning(f"⚠️ Skip arming {symbol}: LTP {ltp} already >= trigger {trigger_price}")
            return None
        
        # Place SL-L order
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=qty,
            order_type=kite.ORDER_TYPE_SL,
            product=kite.PRODUCT_MIS,
            price=limit_price,
            trigger_price=trigger_price,
            validity=kite.VALIDITY_DAY
        )

        # ONLY update budget IF order_id is returned successfully
        if order_id:
            global daily_spent
            daily_spent += (qty * trigger_price)
            logger.info(f"✅ ARMED SL-L ORDER: {symbol} | OrderID={order_id} | Spent: ₹{daily_spent}")
            return order_id
    except Exception as e:
        logger.error(f"❌ Order Failed for {symbol}: {e}")
        return None
    
def monitor_order_status(symbol, order_id, breakout_price):
    """
    Monitors an armed SL-L order and starts exit monitoring when it executes.
    Runs in background thread.
    """
    logger.info(f"📡 Monitoring order status for {symbol} (OrderID: {order_id})")
    
    max_wait_time = 3600  # 1 hour max wait
    start_time = time.time()
    check_interval = 5  # Check every 5 seconds
    
    while True:
        try:
            # Check if we've waited too long
            if time.time() - start_time > max_wait_time:
                logger.warning(f"⏰ Order monitoring timeout for {symbol} after 1 hour")
                # Cancel the order
                try:
                    kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
                    logger.info(f"🗑️ Cancelled stale order for {symbol}")
                except:
                    pass
                ARMED_SYMBOLS_REAL.discard(symbol)
                break
            
            # Check if market is closed
            now = datetime.datetime.now()
            if now.time() >= datetime.time(15, 30):
                logger.info(f"🔔 Market closed, cancelling order for {symbol}")
                try:
                    kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
                except:
                    pass
                ARMED_SYMBOLS_REAL.discard(symbol)
                break
            
            # Get order status
            order_info = kite.order_history(order_id)
            if not order_info:
                time.sleep(check_interval)
                continue
            
            latest_status = order_info[-1]['status']
            
            if latest_status == 'COMPLETE':
                avg_price = order_info[-1]['average_price']
                filled_qty = order_info[-1]['filled_quantity']
                logger.info(f"✅ ORDER EXECUTED: {symbol} at ₹{avg_price} | Qty={filled_qty}")
                
                # ============================================
                # UPDATE GOOGLE SHEET
                # ============================================
                breakout_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                armed_time_str = ARMED_TIMES_REAL.get(symbol, "")
                
                update_sheet_on_breakout(
                    symbol=symbol,
                    breakout_time=breakout_time_str,
                    armed_time=armed_time_str,
                    buy_price=avg_price,
                    breakout_price=breakout_price,
                    quantity=filled_qty
                )
                
                # Start exit monitoring
                ACTIVE_EXIT_MONITORS.add(symbol)
                threading.Thread(
                    target=monitor_live_exit,
                    args=(symbol, avg_price, filled_qty),
                    daemon=True
                ).start()
                
                ARMED_SYMBOLS_REAL.discard(symbol)
                break
                
            elif latest_status in ['CANCELLED', 'REJECTED']:
                logger.warning(f"❌ Order {latest_status} for {symbol}")
                ARMED_SYMBOLS_REAL.discard(symbol)
                break
            
            # Order still pending, keep waiting
            time.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error monitoring order for {symbol}: {e}")
            time.sleep(check_interval)

def monitor_live_exit(symbol, buy_price, quantity, target_order_id=None):
    """
    Monitors live exit conditions for real money trades.
    
    Args:
        symbol: Stock symbol
        buy_price: Price at which we bought
        quantity: Number of shares we bought
        target_order_id: Not used currently, kept for future use
    """
    token = token_map.get(symbol)
    if not token:
        logger.error(f"❌ Token not found for {symbol}, cannot monitor exit")
        return
    
    sl = buy_price                      # Stop Loss = Buy Price (cost)
    target = round(buy_price * TARGET_PCT, 2)  # Target = Buy Price + 3%
    
    logger.info(f"📡 Exit monitor started: {symbol} | Buy={buy_price} | SL={sl} | Target={target} | Qty={quantity}")
    
    # --- MONITORING LOOP ---
    while symbol in ACTIVE_EXIT_MONITORS:
        try:
            now = datetime.datetime.now()
            
            # ============================================
            # EXIT 1: EOD EXIT at 15:15
            # ============================================
            if now.time() >= EXIT_TIME:
                logger.info(f"🔔 EOD time reached for {symbol}, placing market sell...")
                
                try:
                    kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        exchange=kite.EXCHANGE_NSE,
                        tradingsymbol=symbol,
                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                        quantity=quantity,                          # ✅ CORRECT quantity
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=kite.PRODUCT_MIS
                    )
                    
                    # Get actual exit price from latest candle
                    candles = kite.historical_data(token, now.date(), now, "minute")
                    exit_price = round(candles[-1]['close'], 2) if candles else buy_price
                    
                    # ✅ Update real money sheet
                    update_sheet_on_exit(symbol, exit_price, "EOD Exit @15:15")
                    logger.info(f"🔔 EOD EXIT: {symbol} | Sold at ₹{exit_price}")
                    
                except Exception as e:
                    logger.error(f"❌ EOD sell order failed for {symbol}: {e}")
                    # Still update sheet with buy_price as fallback
                    update_sheet_on_exit(symbol, buy_price, "EOD Exit (order failed)")
                
                break
            
            # ============================================
            # FETCH CURRENT PRICE (Using Quotes))
            # ============================================
            try:
                quote = kite.quote(f"NSE:{symbol}")
                curr_price = quote[f"NSE:{symbol}"]["last_price"]
            except Exception as e:
                logger.error(f"Quote fetch failed for {symbol}: {e}")
                time.sleep(5)
                continue
            
            # ============================================
            # EXIT 2: STOP LOSS HIT (Price <= Buy Price)
            # ============================================
            if curr_price <= sl:
                logger.info(f"❌ SL triggered for {symbol} | Current={curr_price} | SL={sl}")
                
                try:
                    kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        exchange=kite.EXCHANGE_NSE,
                        tradingsymbol=symbol,
                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                        quantity=quantity,                          # ✅ CORRECT quantity
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=kite.PRODUCT_MIS
                    )
                    
                    # Get actual exit price
                    candles = kite.historical_data(token, now.date(), now, "minute")
                    exit_price = round(candles[-1]['close'], 2) if candles else sl
                    
                    # ✅ Update real money sheet
                    update_sheet_on_exit(symbol, exit_price, "SL Hit Breakout Price")
                    logger.info(f"❌ SL EXIT: {symbol} | Sold at ₹{exit_price}")
                    
                except Exception as e:
                    logger.error(f"❌ SL sell order failed for {symbol}: {e}")
                    update_sheet_on_exit(symbol, sl, "SL Hit (order failed)")
                
                break
            
            # ============================================
            # EXIT 3: TARGET HIT (Price >= Buy Price + 3%)
            # ============================================
            elif curr_price >= target:
                logger.info(f"🎯 Target triggered for {symbol} | Current={curr_price} | Target={target}")
                
                try:
                    kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        exchange=kite.EXCHANGE_NSE,
                        tradingsymbol=symbol,
                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                        quantity=quantity,                          # ✅ CORRECT quantity
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=kite.PRODUCT_MIS
                    )
                    
                    # Get actual exit price
                    candles = kite.historical_data(token, now.date(), now, "minute")
                    exit_price = round(candles[-1]['close'], 2) if candles else target
                    
                    # ✅ Update real money sheet
                    update_sheet_on_exit(symbol, exit_price, "Target Hit 3%")
                    logger.info(f"🎯 TARGET EXIT: {symbol} | Sold at ₹{exit_price}")
                    
                except Exception as e:
                    logger.error(f"❌ Target sell order failed for {symbol}: {e}")
                    update_sheet_on_exit(symbol, target, "Target Hit (order failed)")
                
                break
            
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error monitoring exit for {symbol}: {e}")
            time.sleep(5)

    # --- RE-ENTRY CLEANUP ---
    ACTIVE_EXIT_MONITORS.discard(symbol)
    if symbol in ARMED_SYMBOLS_REAL:
        ARMED_SYMBOLS_REAL.discard(symbol)
    # Clean up armed time tracking
    ARMED_TIMES_REAL.pop(symbol, None)
    logger.info(f"♻️ Re-entry enabled for {symbol}")

def process_stocks(symbols, today, tier):
    """
    Processes a list of stocks for potential arming.
    Handles both SLOW and FAST tier stocks.
    
    Args:
        symbols: List of stock symbols to check
        today: Current date
        tier: "slow" or "fast" (for logging purposes)
    """
    
    # ============================================
    # OPTIMIZATION: Fetch ALL stock data in ONE call
    # ============================================
    all_stock_data = get_monitor_list(today, symbols=symbols)
    
    # Create a dictionary for fast lookup
    stock_data_map = {stock["symbol"]: stock for stock in all_stock_data}
    
    logger.info(f"📊 {tier.upper()} tier: Fetched {len(stock_data_map)} stocks from monitor list")
    
    # ============================================
    # Fetch ALL quotes in batches of 50 (Kite limit)
    # ============================================
    all_quotes = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        try:
            batch_keys = [f"NSE:{s}" for s in batch]
            quotes = kite.quote(batch_keys)
            all_quotes.update(quotes)
        except Exception as e:
            logger.error(f"Quote batch failed: {e}")
    
    # ============================================
    # Now process each symbol with pre-fetched data
    # ============================================
    for symbol in symbols:
        if datetime.datetime.now().time() >= datetime.time(15, 10):
            logger.info("🕒 Past 15:10 - Stopping new entry checks for today.")
            break # Use break instead of continue to stop the whole loop faster

        # Skip if already in active trade
        if symbol in ACTIVE_EXIT_MONITORS: 
            continue
        
        # Skip if already armed
        if symbol in ARMED_SYMBOLS_REAL:
            continue
        
        # Get stock data from map (no database call)
        stock_data = stock_data_map.get(symbol)
        if not stock_data:
            continue

        # Get quote from batch fetch (no API call)
        quote = all_quotes.get(f"NSE:{symbol}")
        if not quote:
            continue
        
        current_price = quote["last_price"]

        mas = []
        for p in [20, 50, 100, 200]:
            for t in ['daily', 'hourly']:
                val = stock_data.get(f"ma{p}_{t}")
                if val is not None:
                    mas.append(val)
        
        if len(mas) != 8:
            continue
        
        mas.sort()

        # ============================================
        # TIER UPGRADE: SLOW → FAST
        # ============================================
        if tier == "slow":
            sixth_ma = mas[-3]  # 6th MA (3rd highest)
        
            # If price crosses 6th MA, upgrade to FAST
            if current_price >= sixth_ma:
                try:
                    from common import supabase
                    supabase.table("monitor_list").update(
                        {"monitoring_tier": "fast"}
                    ).eq("symbol", symbol).eq("date", today.strftime("%Y-%m-%d")).execute()
                    
                    logger.info(f"⚡ {symbol} upgraded to FAST | LTP {current_price} >= 6th MA {sixth_ma}")
                except Exception as e:
                    logger.error(f"Failed to upgrade {symbol}: {e}")
                
                # Don't arm yet - wait for next FAST tier check
                continue

        # ============================================
        # TIER DOWNGRADE: FAST → SLOW
        # ============================================
        if tier == "fast":
            # Identify the 5th MA (out of 8). If price falls below it, the stock is 'cooling off'.
            fifth_ma = mas[-4] if len(mas) >= 4 else mas[0]
            
            if current_price <= fifth_ma:
                try:
                    from common import supabase
                    # Update the database so the script checks this stock less frequently
                    supabase.table("monitor_list").update(
                        {"monitoring_tier": "slow"}
                    ).eq("symbol", symbol).eq("date", today.strftime("%Y-%m-%d")).execute()
                    
                    logger.info(f"🐢 {symbol} downgraded to SLOW | LTP {current_price} <= 5th MA {fifth_ma}")
                except Exception as e:
                    logger.error(f"Failed to downgrade {symbol}: {e}")

        # ============================================
        # ARMING LOGIC: Only for FAST tier OR if already at midpoint
        # ============================================
        
        max_ma = mas[-1]      # 8th MA (highest)
        seventh_ma = mas[-2]  # 7th MA (2nd highest)
        
        tick_size = stock_data.get("tick_size", 0.05)
        
        # Calculate midpoint and arm threshold
        midpoint = (seventh_ma + max_ma) / 2
        arm_threshold = midpoint - tick_size
        
        # Check if price reached arming point
        if current_price >= arm_threshold:
            breakout_price = next_price_above(max_ma, tick_size)

            if (breakout_price - current_price) < (tick_size * 2):
                logger.warning(f"⏩ {symbol} too close to breakout ({current_price} vs {breakout_price}). Skipping advance arming.")
                LOGGED_SKIPS.add(symbol)
                continue
            
            # Place advance SL-L order
            order_id = place_real_order(symbol, breakout_price, tick_size)
            
            if order_id:
                ARMED_SYMBOLS_REAL.add(symbol)
                ARMED_TIMES_REAL[symbol] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Start monitoring order status in background
                threading.Thread(
                    target=monitor_order_status,
                    args=(symbol, order_id, breakout_price),
                    daemon=True
                ).start()
                
                logger.info(f"🟦 ARMED: {symbol} ({tier.upper()}) | Current={current_price:.2f} | Midpoint={midpoint:.2f} | Breakout={breakout_price:.2f}")

def live_engine():
    global daily_spent
    daily_spent = 0
    
    today = datetime.date.today()
    load_token_map()
    last_fast_check = None
    last_slow_check = None
    
    logger.info(f"🚀 Starting live engine | Daily Budget: ₹{DAILY_BUDGET}")
    
    # Flag to prevent overlapping slow checks
    slow_check_running = False
    
    def run_slow_tier_background():
        """Runs SLOW tier checks in background thread"""
        nonlocal slow_check_running, last_slow_check
        try:
            slow_check_running = True
            slow_stocks = get_stocks_by_tier("slow", today)
            logger.info(f"🟢 SLOW tier: Checking {len(slow_stocks)} stocks")
            
            if slow_stocks:
                process_stocks(slow_stocks, today, "slow")
            
            last_slow_check = datetime.datetime.now()
        except Exception as e:
            logger.error(f"Error in SLOW tier thread: {e}")
        finally:
            slow_check_running = False
    
    while True:
        now = datetime.datetime.now()
        
        # Only run during market hours
        if now.time() < datetime.time(9, 15) or now.time() > datetime.time(15, 30):
            logger.info("Market closed. Sleeping...")
            time.sleep(60)
            continue
        
        # ============================================
        # 1. FAST TIER: Check every 60 seconds
        # ============================================
        if last_fast_check is None or (now - last_fast_check).seconds >= 60:
            fast_stocks = get_stocks_by_tier("fast", today)
            
            if fast_stocks:
                logger.debug(f"⚡ FAST tier: Checking {len(fast_stocks)} stocks")
                process_stocks(fast_stocks, today, "fast")
            
            last_fast_check = now
        
        # ============================================
        # 2. SLOW TIER: Check every 15 minutes (900 seconds)
        # ============================================
        if (last_slow_check is None or (now - last_slow_check).total_seconds() >= 900) and not slow_check_running:
            # Update timestamp BEFORE starting to prevent race conditions
            last_slow_check = now
            t = threading.Thread(target=run_slow_tier_background)
            t.daemon = True
            t.start()
        
        time.sleep(5)

if __name__ == "__main__":
    live_engine()