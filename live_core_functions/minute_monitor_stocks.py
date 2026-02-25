import time as time_module
from datetime import datetime, date, time
import pandas as pd
import numpy as np
from utils.common import kite, logger, supabase, batch_upsert_supabase, next_price_above
import threading
from live_core_functions.monitor_breakouts import get_monitor_list, process_breakout, SCRIPT_START_TIME
from live_core_functions.live_paper_trader import start_paper_trade
import pytz
from utils.common import get_ist_time

ARMED_SYMBOLS = set()
PAPER_TRADES_TODAY = set()
TRADE_LOCK = threading.Lock()

def load_monitor_list_from_excel():
    try:
        df = pd.read_excel("monitor_list_backup.xlsx")
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"Failed to load monitor list from Excel: {e}")
        return []

def get_stocks_by_tier(tier, date):
    """Fetch ALL stocks by monitoring tier using pagination with tier verification."""

    if supabase is None:
        logger.warning("Supabase down → using Excel monitor list")
        stocks = load_monitor_list_from_excel()
        return [s["symbol"] for s in stocks if s.get("monitoring_tier") == tier]

    try:
        all_stocks = []
        page_size = 1000
        offset = 0
        
        while True:
            response = supabase.table("monitor_list")\
                .select("symbol, monitoring_tier")\
                .eq("date", date.strftime("%Y-%m-%d"))\
                .eq("monitoring_tier", tier)\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            # Double-check tier matches (handles race conditions from tier upgrades)
            for row in response.data:
                if row.get("monitoring_tier") == tier:
                    all_stocks.append(row["symbol"])
            
            if len(response.data) < page_size:
                break  # Last page
            
            offset += page_size
        
        logger.info(f"Fetched {len(all_stocks)} stocks for {tier} tier (verified)")
        return all_stocks
        
    except Exception as e:
        logger.error(f"Failed to fetch {tier} tier stocks: {e}")
        return []

def run_breakout_check(symbols, tier):
    """Trigger breakout monitoring for given symbols."""
    if not symbols:
        return
    
    logger.info(f"Checking {len(symbols)} stocks ({tier} tier) for breakouts...")
    
    today = date.today()

    # Use the global token_map from common instead of re-fetching every minute
    from utils.common import token_map as global_map
    if not global_map:
        from utils.common import load_token_map
        load_token_map()
    token_map_local = global_map
    
    # 1. Get LTP for ALL symbols in batches of 50 (Kite limit)
    all_quotes = {}
    for i in range(0, len(symbols), 50):
        batch = [f"NSE:{s}" for s in symbols[i:i+50]]
        try:
            all_quotes.update(kite.quote(batch))
        except Exception as e:
            logger.error(f"Quote batch failed: {e}")

    # Get Nifty data
    nifty_token = token_map_local.get("NIFTY 50")

    if nifty_token:
        try:
            nifty_data_raw = kite.historical_data(
                instrument_token=nifty_token,
                from_date=today,
                to_date=today,
                interval="minute"
            )
            nifty_df = pd.DataFrame(nifty_data_raw)
            if not nifty_df.empty:
                nifty_df["date"] = pd.to_datetime(nifty_df["date"]).dt.tz_localize(None)
                nifty_data = nifty_df.set_index("date")[["open", "close"]]
            else:
                nifty_data = pd.DataFrame()
        except:
            nifty_data = pd.DataFrame()
    else:
        nifty_data = pd.DataFrame()
    
    # 3. Fetch monitor data for these symbols
    try:
        fresh_monitor_list = get_monitor_list(today, symbols=symbols)
    except Exception as e:
        logger.error(f"Supabase monitor_list fetch failed: {e}")
        fresh_monitor_list = []

    # 🔁 Excel fallback
    if not fresh_monitor_list:
        logger.warning("Using Excel fallback for monitor list")
        fresh_monitor_list = load_monitor_list_from_excel()


    # Create a mapping of current tiers
    monitor_by_symbol = {stock["symbol"]: stock for stock in fresh_monitor_list}
    
    for symbol in symbols:
        # Get latest monitor data
        stock_data = monitor_by_symbol.get(symbol)
        if not stock_data:
            continue
        
        quote = all_quotes.get(f"NSE:{symbol}")
        if not quote: 
            continue
        
        current_price = quote["last_price"]
        
        # Use FRESH tier from database, not the stale one from function parameter
        current_tier = stock_data.get("monitoring_tier", tier)
        
        # --- TIER MANAGEMENT (UPGRADE & DOWNGRADE) ---
        mas = sorted([
            val
            for p in [20, 50, 100, 200]
            for t in ['daily', 'hourly']
            for val in [stock_data.get(f"ma{p}_{t}")]
            if val is not None and not pd.isna(val)
        ])

        if not mas: continue
        
        max_ma = mas[-1]                     # 8th MA (highest)
        seventh_ma = mas[-2] if len(mas) >= 2 else mas[-1]  # 7th MA (2nd highest)
        sixth_ma = mas[-3] if len(mas) >= 3 else seventh_ma # 6th MA (3rd highest)
        fifth_ma = mas[-4] if len(mas) >= 4 else mas[0]     # 5th MA

        # Upgrade Slow -> Fast
        if current_price >= sixth_ma and current_tier == "slow":
            try:
                supabase.table("monitor_list").update({"monitoring_tier": "fast"}).eq("symbol", symbol).eq("date", today.strftime("%Y-%m-%d")).execute()
                logger.info(f"⚡ {symbol} upgraded to FAST (LTP {current_price} >= 6th MA {sixth_ma})")
            except: pass

            instrument_token = token_map_local.get(symbol)
            if not instrument_token:
                continue
            # ============================================
            # 🚀 PRECOMPUTE ML FEATURES AT FAST TIER
            # ============================================

            # analysis_date = datetime.date.today()

            # existing = supabase.table("live_ml_features") \
            #     .select("symbol") \
            #     .eq("symbol", symbol) \
            #     .eq("date", analysis_date.strftime("%Y-%m-%d")) \
            #     .execute()

            # if not existing.data:
            #     logger.info(f"⚡ Precomputing ML features for FAST tier: {symbol}")

            #     stock_df = fetch_30d_daily_ohlcv(symbol, instrument_token, analysis_date)

            #     if stock_df is not None and not stock_df.empty:
            #         nifty_map = fetch_nifty_daily_map(analysis_date)

            #         rows = []
            #         for _, r in stock_df.iterrows():
            #             d = r["date"]
            #             n = nifty_map.get(d)
            #             rows.append({
            #                 "open": r["open"],
            #                 "high": r["high"],
            #                 "low": r["low"],
            #                 "close": r["close"],
            #                 "volume": r["volume"],
            #                 "nifty_open": n["open"] if n is not None else 0.0,
            #                 "nifty_close": n["close"] if n is not None else 0.0
            #             })

            #         ml_df = pd.DataFrame(rows)
            #         nifty_ml_df = ml_df[["nifty_open", "nifty_close"]]

            #         ml_features = get_ml_model_features(ml_df, nifty_ml_df)

            #     else:
            #         ml_features = {feat: 0.0 for feat in ML_FEATURES}

            #     supabase.table("live_ml_features").upsert(
            #         {
            #             "symbol": symbol,
            #             "date": analysis_date.strftime("%Y-%m-%d"),
            #             "features": ml_features
            #         },
            #         on_conflict="symbol,date"
            #     ).execute()

            #     logger.info(f"✅ ML features cached early for {symbol}")

        # Downgrade Fast -> Slow
        if current_price <= fifth_ma and current_tier == "fast":
            try:
                supabase.table("monitor_list").update({"monitoring_tier": "slow"}).eq("symbol", symbol).eq("date", today.strftime("%Y-%m-%d")).execute()
                logger.info(f"🐢 {symbol} downgraded to SLOW (LTP {current_price} <= 5th MA {fifth_ma})")
            except: pass    


        # ============================================================
        # ⚡ ARM SL-L ORDER when LTP reaches midpoint of 7th & 8th MA
        # Only for FAST tier stocks
        # ============================================================
        if current_tier == "fast" and symbol not in ARMED_SYMBOLS:
            midpoint = (seventh_ma + max_ma) / 2
            tick_size = stock_data.get("tick_size") or 0.05
            arm_threshold = midpoint - tick_size

            if current_price >= arm_threshold:
                # 1. Define the prices BEFORE the try block
                breakout_price = next_price_above(max_ma, tick_size)
                trigger_price = breakout_price
                limit_price = round(breakout_price + tick_size, 2)

                try:
                    from live_core_functions.live_paper_trader import place_sl_l_buy_order
                    place_sl_l_buy_order(symbol, trigger_price, limit_price, investment_per_trade=5000)
                    ARMED_SYMBOLS.add(symbol)
                    logger.info(f"🟦 ARMED {symbol} | TRIGGER={trigger_price} LIMIT={limit_price}")
                except Exception as e:
                    if "already >= trigger" in str(e):
                        ARMED_SYMBOLS.add(symbol)
                        logger.debug(f"Symbol {symbol} already past trigger, skipping further arming.")
                    else:
                        logger.error(f"❌ Failed to ARM order for {symbol}: {e}")

        # ============================================================
        # 🚀 IMMEDIATE LTP BREAKOUT TRIGGER
        # ============================================================
        tick_size = stock_data.get("tick_size") or 0.05
        breakout_price = next_price_above(max_ma, tick_size)

        # Trigger trade IMMEDIATELY when price crosses level
        if current_price >= breakout_price:
            # 🆕 TIME GATE: Prevent new entries at or after 3:15 PM IST
            # Using get_ist_time() to ensure timezone consistency
            now_ist = get_ist_time()
            if now_ist.time() >= time(15, 15):
                logger.info(f"⏭️ Skipping late breakout for {symbol} at {current_price} (Time: {now_ist.strftime('%H:%M')})")
                continue

            # ✅ Thread-safe check and add
            with TRADE_LOCK:
                if symbol in PAPER_TRADES_TODAY:
                    continue
                PAPER_TRADES_TODAY.add(symbol)
            
            # Check database to prevent duplicate trades for the same day
            existing = supabase.table("live_breakouts") \
                .select("symbol") \
                .eq("symbol", symbol) \
                .eq("breakout_date", today.strftime("%Y-%m-%d")) \
                .limit(1).execute()

            if not existing.data:
                detection_time = datetime.now()

                if detection_time <= SCRIPT_START_TIME:
                    logger.info(f"⏭️ Skipping old breakout for {symbol} (before script start)")
                    with TRADE_LOCK:
                        PAPER_TRADES_TODAY.discard(symbol)  # Allow re-check next cycle
                    continue

                logger.info(f"🎯 BREAKOUT DETECTED: {symbol} at {current_price}")
                start_paper_trade(
                    symbol=symbol,
                    breakout_price=breakout_price,
                    breakout_time=detection_time, # Pass the exact detection time
                    model_pred="Processing...",
                    ai_dec="Processing..."
                )

                # 2. RUN ANALYSIS IN BACKGROUND THREAD
                # This keeps the main loop fast so it can check other stocks
                analysis_thread = threading.Thread(target=process_breakout, kwargs={
                    "symbol": symbol,
                    "monitor_data": stock_data,
                    "current_price": current_price,
                    "analysis_date": today,
                    "instrument_token": token_map_local.get(symbol)
                })
                analysis_thread.daemon = True
                analysis_thread.start()

def check_stagnant_exits(now_str):
    """Exits trades where price has hit a circuit or stopped moving."""
    try:
        # 1. Fetch all active trades for today
        active_trades = supabase.table("live_breakouts") \
            .select("id, symbol, last_price_checked, stagnant_count, breakout_price") \
            .eq("breakout_date", now_str) \
            .is_("exit_reason", None) \
            .execute()

        if not active_trades.data:
            return

        # 2. Get live quotes for all active symbols
        symbols = [f"NSE:{t['symbol']}" for t in active_trades.data]
        quotes = kite.quote(symbols)

        for trade in active_trades.data:
            symbol = trade['symbol']
            q = quotes.get(f"NSE:{symbol}")
            if not q: continue

            current_ltp = q['last_price']
            last_ltp = trade['last_price_checked']
            count = trade['stagnant_count'] or 0

            # 3. Check if price is stagnant
            if last_ltp is not None and current_ltp == last_ltp:
                count += 1
                logger.info(f"⚠️ {symbol} is stagnant. Count: {count}/3 (LTP: {current_ltp})")
            else:
                count = 0 # Reset if price moves

            if count >= 3:
                logger.info(f"🛑 EXIT: {symbol} hit stagnant/circuit limit at {current_ltp}")
                from live_core_functions.live_paper_trader import finalize_trade
                finalize_trade(symbol, current_ltp, "Stagnant/Circuit Exit")
                continue

            # 4. Update tracking columns
            supabase.table("live_breakouts").update({
                "last_price_checked": current_ltp,
                "stagnant_count": count
            }).eq("id", trade['id']).execute()

    except Exception as e:
        logger.error(f"Error in stagnant check: {e}")

def start_finding_breakouts():
    """Main monitoring loop with non-blocking Slow Tier."""
    logger.info("Starting tiered monitoring system...")
    
    today = date.today()
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    last_slow_check = None
    last_fast_check = None
    
    # Flag to prevent overlapping slow checks
    slow_check_running = False

    def run_slow_tier_background():
        nonlocal slow_check_running, last_slow_check
        try:
            slow_check_running = True
            slow_stocks = get_stocks_by_tier("slow", today)
            logger.info(f"🐢 SLOW tier fetched {len(slow_stocks)} stocks")

            if slow_stocks:
                run_breakout_check(slow_stocks, "slow")
            last_slow_check = get_ist_time()
        except Exception as e:
            logger.error(f"Error in Slow Tier thread: {e}")
        finally:
            slow_check_running = False

    while True:
        now = get_ist_time()
        current_time = now.time()
        
        # Only run during market hours
        if current_time < market_open or current_time > market_close:
            logger.info("Market closed. Sleeping...")
            time_module.sleep(600)
            continue

        # 0. STAGNANT CHECK: Run every minute
        if last_fast_check is None or (now - last_fast_check).seconds >= 60:
            check_stagnant_exits(today.strftime("%Y-%m-%d"))
            
        # 1. FAST TIER: Run immediately in main thread (High Priority)
        if last_fast_check is None or (now - last_fast_check).seconds >= 60:
            fast_stocks = get_stocks_by_tier("fast", today)
            fast_stocks = [
                s for s in fast_stocks
                if s not in PAPER_TRADES_TODAY
            ]
            if fast_stocks:
                logger.info(f"⚡ FAST tier stocks: ({len(fast_stocks)}):{fast_stocks} ")
                run_breakout_check(fast_stocks, "fast")
            last_fast_check = now
        
        # 2. SLOW TIER: Run every 15 minutes (900 seconds)
        if (last_slow_check is None or (now - last_slow_check).total_seconds() >= 900) and not slow_check_running:
            # We update the timestamp HERE before starting to prevent race conditions
            last_slow_check = now 
            t = threading.Thread(target=run_slow_tier_background)
            t.daemon = True
            t.start()
        
        time_module.sleep(5)

        if current_time >= time(15, 15) and current_time < time(15, 30):
            try:
                # Fetch all OPEN trades for today
                open_trades = supabase.table("live_breakouts") \
                    .select("symbol, breakout_price") \
                    .eq("breakout_date", now.strftime("%Y-%m-%d")) \
                    .is_("exit_reason", None) \
                    .execute()

                if open_trades.data:
                    logger.info(f"⏰ EOD Cleanup: Closing {len(open_trades.data)} remaining trades.")
                    for b in open_trades.data:
                        symbol = b["symbol"]
                        # Get final LTP
                        try:
                            q = kite.quote(f"NSE:{symbol}")
                            ltp = q[f"NSE:{symbol}"]["last_price"]
                        except:
                            ltp = b["breakout_price"] # Fallback

                        # This calls the finalize_trade helper to update Sheet and Supabase
                        from live_core_functions.live_paper_trader import finalize_trade
                        finalize_trade(symbol, ltp, "EOD Exit @15:15")
            except Exception as e:
                logger.error(f"Global EOD Force Exit Error: {e}")

        # EOD SNAPSHOT at 15:30 — Save exact UI state for historical view
        if current_time >= time(15, 30):
            snapshot_taken = supabase.table("dashboard_snapshots") \
                .select("date") \
                .eq("date", today.strftime("%Y-%m-%d")) \
                .execute()

            if not snapshot_taken.data:   # Only take once per day
                try:
                    from main import build_dashboard_data   # Import the helper we created
                    snapshot_data = build_dashboard_data(today.strftime("%Y-%m-%d"))

                    supabase.table("dashboard_snapshots").upsert({
                        "date": today.strftime("%Y-%m-%d"),
                        "data": snapshot_data
                    }).execute()

                    logger.info(f"📸 Dashboard snapshot saved for {today} at 15:30")
                except Exception as e:
                    logger.error(f"Snapshot failed: {e}")

if __name__ == "__main__":
    start_finding_breakouts()