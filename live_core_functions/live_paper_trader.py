import gspread
import pandas as pd
import datetime
import time
from utils.common import kite, token_map, load_token_map, logger, supabase, batch_upsert_supabase, next_price_above
import threading
from dotenv import load_dotenv
import json
import os
import base64
import pytz

load_dotenv()
load_token_map()
# --- Google Sheets Setup ---
creds_b64 = os.getenv("GOOGLE_SHEET_CREDS_B64")

if not creds_b64:
    raise Exception("GOOGLE_SHEET_CREDS_B64 not set")

decoded = base64.b64decode(creds_b64)
info = json.loads(decoded)

gc = gspread.service_account_from_dict(info)
sh = gc.open("Live_Paper_Trading")
sheet = sh.sheet1

PAPER_TRADE_ONLY = True
SHEET_LOCK = threading.Lock()
ACTIVE_EXIT_MONITORS = set()
ARMED_TIMES = {}


def backfill_exit_for_open_trades():
    if sheet is None:
        return

    all_rows = sheet.get_all_values()
    today = datetime.date.today()

    for i, row in enumerate(all_rows[1:], start=2):

        if len(row) < 11:
            continue

        if row[10].strip() != "OPEN":
            continue

        symbol = row[1]
        buy_price = float(row[6])
        buy_time_str = row[0]   # Breakout Time column

        try:
            buy_dt = datetime.datetime.strptime(buy_time_str, "%Y-%m-%d %H:%M:%S")
        except:
            continue

        token = token_map.get(symbol)
        if not token:
            continue

        try:
            candles = kite.historical_data(
                token,
                from_date=buy_dt,
                to_date=datetime.date.today(),
                interval="minute"
            )
        except:
            continue

        if not candles:
            continue

        sl = buy_price
        target = buy_price * 1.03

        exit_price = None
        reason = None

        for c in candles:
            low = c["low"]
            high = c["high"]

            if high >= target:
                exit_price = target
                reason = "Target Hit 3%"
                break

            if low <= sl:
                exit_price = sl
                reason = "SL Hit Breakout Price"
                break


        if exit_price is None:
            exit_price = candles[-1]["close"]
            reason = "EOD Exit @15:15"

        pnl_pct = round(((exit_price - buy_price) / buy_price) * 100, 2)

        with SHEET_LOCK:
            sheet.update_cell(i, 11, "CLOSED")
            sheet.update_cell(i, 12, exit_price)
            sheet.update_cell(i, 13, reason)
            sheet.update_cell(i, 14, f"{pnl_pct}%")

        print(f"📌 Backfilled {symbol}: {reason}")

DAILY_BUDGET = 20000
daily_spent = 0

def place_sl_l_buy_order(symbol, trigger_price, limit_price, investment_per_trade=5000):
    """
    Places SL-L BUY order in Kite.
    Does NOT buy immediately.
    It will execute ONLY if price hits trigger_price.
    """

    tradingsymbol = symbol

    # Check LTP so Kite doesn't reject trigger order
    q = kite.quote([f"NSE:{tradingsymbol}"])
    ltp = q[f"NSE:{tradingsymbol}"]["last_price"]

    if ltp >= trigger_price:
        raise Exception(f"Skip arming: LTP {ltp} already >= trigger {trigger_price}")

    qty = int(investment_per_trade / limit_price)
    if qty <= 0:
        raise Exception("Quantity becomes 0 (investment too low for this stock price)")

    if PAPER_TRADE_ONLY:
        armed_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ARMED_TIMES[symbol] = armed_time

        print(
            f"[PAPER MODE] ARMED SL-L BUY {symbol} | "
            f"trigger={trigger_price} limit={limit_price} investment={investment_per_trade}"
        )

        # OPTIONAL: log to Google Sheet so you can verify everything
        try:
            ok = update_armed_time(symbol, armed_time)
            if not ok:
                print(f"ℹ️ Trade row not created yet for {symbol}, armed time stored in memory.")

        except Exception as e:
            print(f"⚠️ Failed to log ARM to sheet: {e}")

        return "PAPER_ARMED"

    # order_id = kite.place_order(
    #     variety=kite.VARIETY_REGULAR,
    #     exchange=kite.EXCHANGE_NSE,
    #     tradingsymbol=tradingsymbol,
    #     transaction_type=kite.TRANSACTION_TYPE_BUY,
    #     quantity=qty,
    #     order_type=kite.ORDER_TYPE_SL,       # ✅ SL-L order
    #     product=kite.PRODUCT_MIS,            # use MIS for intraday, CNC for delivery
    #     price=limit_price,                   # ✅ limit price
    #     trigger_price=trigger_price,         # ✅ trigger price
    #     validity=kite.VALIDITY_DAY
    # )

    # print(f"✅ SL-L ARMED {symbol} qty={qty} trigger={trigger_price} limit={limit_price} order_id={order_id}")
    # return order_id

def update_armed_time(symbol, armed_time):
    if sheet is None:
        return False

    all_rows = sheet.get_all_values()
    for i, row in enumerate(all_rows[1:], start=2):  # start=2 because header row
        if len(row) >= 11 and row[1] == symbol and row[10] == "OPEN":
            with SHEET_LOCK:
                sheet.update_cell(i, 5, armed_time)  # Column E = 5
            return True
    return False

def reset_daily_budget():
    """Reset budget at market open"""
    global daily_spent
    daily_spent = 0

def update_sheet(data):
    if sheet is None:
        print("❌ Google Sheet not connected, cannot append.")
        return False

    for attempt in range(1, 6):
        try:
            with SHEET_LOCK:
                sheet.append_row(data)
            return True
        except Exception as e:
            print(f"⚠️ append_row failed attempt {attempt}/5: {e}")
            time.sleep(1.5 * attempt)

    return False

def update_trade_analysis(symbol, model_pred, ai_dec):
    """Finds the open trade in Google Sheets and fills in the AI/ML results."""
    if sheet is None: return
    try:
        all_rows = sheet.get_all_values()
        for i, row in enumerate(all_rows[1:], start=2):
            # Find the row for this symbol that is still 'OPEN'
            if len(row) >= 11 and row[1] == symbol and row[10].strip() == "OPEN":
                with SHEET_LOCK:
                    sheet.update_cell(i, 3, str(model_pred)) # Column C: ML Pred
                    sheet.update_cell(i, 4, str(ai_dec))    # Column D: AI Dec
                print(f"✅ Analysis results filled in for {symbol}")
                break
    except Exception as e:
        print(f"Error updating analysis: {e}")

def start_paper_trade(symbol, breakout_price, breakout_time, model_pred, ai_dec):
    """
    Call this from monitor_breakouts.py when a cross is detected.
    """
    global daily_spent

    # Ensure breakout_time is used as the primary timestamp for the record
    if isinstance(breakout_time, datetime.datetime):
        breakout_dt = breakout_time
    else:
        breakout_dt = datetime.datetime.now() # Fallback
    
    buy_time = breakout_dt # Sync Buy Time to the Breakout Detection Time
    
    investment_per_trade = 5000
    quantity = int(investment_per_trade / breakout_price)
    money_traded = quantity * breakout_price

    if daily_spent + money_traded > DAILY_BUDGET:
        print(f"❌ Budget exhausted: ₹{daily_spent}/₹{DAILY_BUDGET}")
        return
    
    ai_dec = str(ai_dec)
    armed_time = ARMED_TIMES.get(symbol, "")

    row = [
        breakout_dt.strftime("%Y-%m-%d %H:%M:%S"),  # ✅ Breakout Time (actual candle time)
        symbol,
        model_pred,                               # Model Prediction
        ai_dec,                                   # AI Decision
        armed_time,                               # E Armed Time
        buy_time.strftime("%H:%M:%S"),            # Buy Time
        breakout_price,                           # Buy Price
        breakout_price,                           # Actual Breakout Price
        quantity,                                 # ADD: Quantity
        money_traded,                             # ADD: Money Traded
        "OPEN"                                    # Status
    ]

    daily_spent += money_traded
    print(f"💰 Budget used: ₹{daily_spent}/₹{DAILY_BUDGET}")

    # First insert row
    ok = update_sheet(row)
    if not ok:
        print(f"❌ Sheet insert failed for {symbol}. Trade aborted.")
        return

    print(f"Successfully logged trade for {symbol} to Google Sheets.")

    # Then register monitor
    if symbol in ACTIVE_EXIT_MONITORS:
        return

    ACTIVE_EXIT_MONITORS.add(symbol)

    # Fetch token directly from Kite instruments
    instruments = kite.instruments("NSE")
    token = None

    for ins in instruments:
        if ins["tradingsymbol"] == symbol:
            token = ins["instrument_token"]
            break

    if not token:
        print(f"❌ Token not found for {symbol}. Cannot start exit monitor.")
        return

    exit_thread = threading.Thread(
        target=monitor_live_exit,
        args=(symbol, breakout_price, token),
        daemon=True
    )
    exit_thread.start()

def finalize_trade(symbol, exit_price, reason):
    """
    Finds the active trade in Google Sheets and updates it with exit data.
    """
    try:
        # 1. Fetch all rows to find the matching symbol with 'OPEN' status
        all_rows = sheet.get_all_values()
        row_num = -1
        
        for i, row in enumerate(all_rows):
            if len(row) >= 11 and row[1] == symbol and row[10].strip() == "OPEN":
                row_num = i + 1  # gspread uses 1-based indexing
                buy_price = float(row[6]) # Column index 5 is Buy Price
                break
        
        if row_num != -1:
            pnl_pct = round(((exit_price - buy_price) / buy_price) * 100, 2)

            # ✅ Make updates thread-safe
            with SHEET_LOCK:
                sheet.update_cell(row_num, 11, "CLOSED")
                sheet.update_cell(row_num, 12, exit_price)
                sheet.update_cell(row_num, 13, reason)
                sheet.update_cell(row_num, 14, f"{pnl_pct}%")
            
            # Update Supabase live_breakouts with final EOD move and exit info
            try:
                supabase.table("live_breakouts").update({
                    "percent_move": pnl_pct,
                    "high_price": float(exit_price),
                    "exit_reason": reason
                }).eq("symbol", symbol).eq("breakout_date", datetime.date.today().strftime("%Y-%m-%d")).execute()
            except Exception as e:
                print(f"⚠️ Failed to update Supabase with final move for {symbol}: {e}")

            print(f"Successfully finalized {symbol}: {reason} at {exit_price} ({pnl_pct}%)")

            # ✅ COMPLETE cleanup for re-entry
            ACTIVE_EXIT_MONITORS.discard(symbol)
            from live_core_functions.minute_monitor_stocks import PAPER_TRADES_TODAY, ARMED_SYMBOLS
            PAPER_TRADES_TODAY.discard(symbol)
            ARMED_SYMBOLS.discard(symbol)  # ✅ ADD THIS LINE

        else:
            print(f"Error: Could not find an active 'OPEN' trade for {symbol}.")
            ACTIVE_EXIT_MONITORS.discard(symbol)
            from live_core_functions.minute_monitor_stocks import PAPER_TRADES_TODAY, ARMED_SYMBOLS
            PAPER_TRADES_TODAY.discard(symbol)
            ARMED_SYMBOLS.discard(symbol)

    except Exception as e:
        print(f"CRITICAL ERROR in finalize_trade: {e}")

# --- Live Exit Monitor ---
def monitor_live_exit(symbol, buy_price,token):
    """
    Background loop to check for SL or 3% Target.
    """        
    sl = buy_price
    target = buy_price * 1.03
    
    while symbol in ACTIVE_EXIT_MONITORS:
        try:
            IST = pytz.timezone("Asia/Kolkata")
            to_dt = datetime.datetime.now(IST).replace(tzinfo=None)
            today = to_dt.date()
            from_dt = datetime.datetime.combine(today, datetime.time(9, 15))
            candles = kite.historical_data(token, from_dt, to_dt, "minute")
            
            if not candles:
                time.sleep(30)
                continue
            
            latest_candle = candles[-1]
            curr_price = latest_candle['close']
            
            # Check Exit Conditions using CANDLE data
            if curr_price <= sl:
                finalize_trade(symbol, sl, "SL Hit Breakout Price")
                break
            elif curr_price >= target:
                finalize_trade(symbol, target, "Target Hit 3%")
                break
            elif to_dt.time() >= datetime.time(15, 15):
                finalize_trade(symbol, curr_price, "EOD Exit @15:15")
                break
                
            time.sleep(3) # Poll every 3 seconds for faster SL/Target execution
        except Exception as e:
            print(f"Error monitoring {symbol}: {e}")
            time.sleep(3)