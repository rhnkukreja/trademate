"""This file will store all your constants and environment variables in one place, making them easy to change."""

import os
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration class for the breakout scanner.
    """
    # --- Supabase Configuration ---
    SUPABASE_URL = os.getenv("N_SUPABASE_URL")
    SUPABASE_KEY = os.getenv("N_SUPABASE_ANON_KEY")
    RESULTS_TABLE_NAME = 'new_algo_breakouts'
    UPSERT_BATCH_SIZE = 100

    # --- Data Fetching Configuration ---
    NSE_EQUITIES_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    NIFTY_50_SYMBOL = "^NSEI"
    YFINANCE_THREADS = 3 # Number of parallel threads for yfinance downloads
    
    # --- Analysis Period ---
    # How many days of history to fetch for calculating indicators
    HISTORY_DAYS = 400
    # How many of the most recent trading days to scan for breakouts
    ANALYSIS_PERIOD_DAYS = 21 

    # --- Script Execution ---
    # How many stocks to load into memory at once. Adjust based on available RAM.
    STOCK_BATCH_SIZE = 100
    # Number of CPU cores to use for parallel analysis.
    # We use min(3, cpu_count()) to avoid overwhelming systems with many cores.
    NUM_WORKERS = 3
    # Pause between batches to respect API rate limits
    PAUSE_BETWEEN_BATCHES_SECONDS = 10
    
    # --- Technical Analysis Parameters ---
    MA_PERIODS = [20, 50, 100, 200]
    RSI_PERIOD = 14
    BBANDS_PERIOD = 20
    BBANDS_STD_DEV = 2
    BB_SQUEEZE_LOOKBACK = 120
    AVG_VOL_LOOKBACK = 20
    DEFAULT_TICK_SIZE = 0.05