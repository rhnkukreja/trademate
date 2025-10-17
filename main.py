"""This new main.py is much cleaner. It uses the other modules to orchestrate the entire scanning process, 
from getting symbols to processing batches and saving results."""

import logging
import datetime
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from config import Config
from database import SupabaseHandler
from data_provider import DataProvider
from analyzer import find_breakout_for_day

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('scanner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BreakoutScanner:
    """
    Orchestrates the entire process of scanning for intraday breakouts.
    """
    def __init__(self):
        # This __init__ method is perfect, no changes needed.
        self.db_handler = SupabaseHandler(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.all_stocks = []
        self.processed_stock_dates = set()

    def _prepare_tasks_for_batch(self, active_stocks, provider):
        """Creates the list of task dictionaries for the multiprocessing pool."""
        # This task preparation method is also perfect, no changes needed.
        tasks = []
        nifty_trading_days = provider.data_dict['nifty']['daily']['date'].dt.date.unique()
        date_range = nifty_trading_days[-Config.ANALYSIS_PERIOD_DAYS:]
        logger.info(f"Set analysis period to the last {len(date_range)} trading days.")
        
        nifty_intraday_full = provider.data_dict['nifty'].get('minute', pd.DataFrame())

        for stock in active_stocks:
            symbol = stock['symbol']
            if symbol not in provider.data_dict or symbol not in provider.ma_dict:
                continue
            
            dates_to_process = [d for d in date_range if (symbol, d.strftime('%Y-%m-%d')) not in self.processed_stock_dates]
            
            for analysis_date in dates_to_process:
                tasks.append({
                    'symbol': symbol,
                    'analysis_date': analysis_date,
                    'ma_data': {symbol: provider.ma_dict[symbol]},
                    'intraday_data': provider.data_dict[symbol].get('minute', pd.DataFrame()),
                    'nifty_intraday': nifty_intraday_full[nifty_intraday_full['date'].dt.date == analysis_date].set_index('date'),
                    'nifty_50ma': provider.nifty_50ma_dict.get(analysis_date),
                    'tick_size': Config.DEFAULT_TICK_SIZE
                })
        return tasks

    def run(self):
        """Executes the main scanning loop."""
        logger.info("--- Starting Intraday Breakout Scanner ---")
        
        self.all_stocks = DataProvider.get_all_nse_symbols()
        if not self.all_stocks:
            logger.error("No stocks found. Exiting.")
            return

        self.processed_stock_dates = self.db_handler.get_processed_stock_dates()
        
        total_batches = (len(self.all_stocks) + Config.STOCK_BATCH_SIZE - 1) // Config.STOCK_BATCH_SIZE
        
        for i in range(0, len(self.all_stocks), Config.STOCK_BATCH_SIZE):
            active_stocks = self.all_stocks[i:i + Config.STOCK_BATCH_SIZE]
            batch_num = (i // Config.STOCK_BATCH_SIZE) + 1
            logger.info(f"--- Processing Batch {batch_num}/{total_batches} ({len(active_stocks)} stocks) ---")

            today = datetime.date.today()
            from_date = today - datetime.timedelta(days=Config.HISTORY_DAYS)
            
            provider = DataProvider(active_stocks, from_date, today)
            if not provider.load_data_in_parallel():
                continue

            provider.precompute_indicators()
            tasks = self._prepare_tasks_for_batch(active_stocks, provider)
            logger.info(f"Created {len(tasks)} new tasks for this batch.")

            all_results = []
            
            # ✅ --- START: ENHANCED RESULT PROCESSING BLOCK ---
            if tasks:
                # Use a dictionary to count different failure reasons
                failure_counts = {}
                num_workers = min(Config.NUM_WORKERS, cpu_count())
                with Pool(processes=num_workers) as pool:
                    logger.info(f"Starting parallel processing for batch {batch_num} with {num_workers} workers...")
                    results_list = pool.map(find_breakout_for_day, tasks)
                    
                    for status, data in results_list:
                        if status == 'success':
                            all_results.append(data)
                        else:
                            # Increment the count for the specific failure reason
                            failure_counts[status] = failure_counts.get(status, 0) + 1
                
                logger.info(f"Batch {batch_num} analysis complete.")
                # Log the summary of failures
                for reason, count in failure_counts.items():
                    logger.info(f"Skipped {count} tasks due to reason: '{reason}'")
            # ✅ --- END: ENHANCED RESULT PROCESSING BLOCK ---
            
            if all_results:
                logger.info(f"Found {len(all_results)} potential breakouts in batch {batch_num}.")
                self.db_handler.batch_upsert(all_results)
            else:
                logger.info(f"No new breakout patterns found in batch {batch_num}.")
            
            if batch_num < total_batches:
                 logger.info(f"--- Pausing for {Config.PAUSE_BETWEEN_BATCHES_SECONDS} seconds before next batch ---")
                 time.sleep(Config.PAUSE_BETWEEN_BATCHES_SECONDS)

        logger.info("--- All batches processed. Scanner run finished. ---")


if __name__ == "__main__":
    scanner = BreakoutScanner()
    scanner.run()