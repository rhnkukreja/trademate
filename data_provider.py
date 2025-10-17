"""This is a new, powerful class responsible for all data acquisition and preparation. 
It fetches symbols, downloads historical data, and pre-computes indicators for a given batch of stocks."""

import logging
import datetime
import time
import pandas as pd
import yfinance as yf
import concurrent.futures
from config import Config

logger = logging.getLogger(__name__)

class DataProvider:
    """
    Handles fetching and preparing all required market data.
    """
    def __init__(self, stocks_to_process: list, from_date: datetime.date, to_date: datetime.date):
        self.stocks_to_process = stocks_to_process
        self.from_date = from_date
        self.to_date = to_date
        self.data_dict = {}
        self.ma_dict = {}
        self.nifty_50ma_dict = {}

    @staticmethod
    def get_all_nse_symbols() -> list:
        """Downloads the official list of all NSE-listed equity symbols."""
        logger.info("Downloading latest list of NSE stock symbols...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = pd.read_csv(Config.NSE_EQUITIES_URL)
                df.columns = df.columns.str.strip()
                df = df[df['SERIES'] == 'EQ']
                symbols = df['SYMBOL'].tolist()
                all_stocks = [{"symbol": f"{symbol}.NS"} for symbol in symbols]
                logger.info(f"Successfully fetched {len(all_stocks)} NSE equity symbols.")
                return all_stocks
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to download NSE CSV (attempt {attempt + 1}/{max_retries}). Retrying in 5s... Error: {e}")
                    time.sleep(5)
                else:
                    logger.error(f"Failed to download or process NSE symbol list after {max_retries} attempts: {e}", exc_info=True)
                    return []
    
    @staticmethod
    def _fetch_minute_data(ticker_obj, start_date, end_date):
        """Fetches 15-minute data in 60-day chunks using a Ticker object."""
        df_list = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + datetime.timedelta(days=60), end_date)
            try:
                df_chunk = ticker_obj.history(
                    start=current_start,
                    end=current_end,
                    interval="15m",
                    auto_adjust=True
                )
                if not df_chunk.empty:
                    df_list.append(df_chunk)
            except Exception as e:
                logger.warning(f"Failed to fetch 15m chunk for {ticker_obj.ticker} from {current_start} to {current_end}: {e}")
            current_start = current_end
        return pd.concat(df_list) if df_list else pd.DataFrame()

    def _fetch_data_for_ticker(self, ticker):
        """Fetches and cleans data for a single ticker."""
        try:
            logger.info(f"Fetching data for {ticker}...")
            required_cols = {'open', 'high', 'low', 'close', 'volume'}
            ticker_obj = yf.Ticker(ticker)

            df_daily_raw = ticker_obj.history(
                start=self.from_date, # Uses self.from_date explicitly
                end=self.to_date,     # Uses self.to_date explicitly
                interval="1d",
                auto_adjust=True
            )

            if df_daily_raw.empty:
                logger.warning(f"No daily data for {ticker}.")
                return ticker, {}
            
            # --- Data cleaning logic remains the same ---
            df_daily = df_daily_raw.copy(deep=True).reset_index().rename(columns=str.lower)
            if 'datetime' in df_daily.columns:
                df_daily = df_daily.rename(columns={'datetime': 'date'})
            if 'date' not in df_daily.columns:
                return ticker, {}
            df_daily['date'] = pd.to_datetime(df_daily['date']).dt.tz_localize(None)
            df_daily = df_daily.sort_values(by='date').reset_index(drop=True)

            if not required_cols.issubset(df_daily.columns):
                 return ticker, {}

            # Fetch minute data
            end_date_minute = self.to_date
            start_date_minute = max(end_date_minute - datetime.timedelta(days=58), self.from_date)
            
            df_minute_raw = self._fetch_minute_data(ticker_obj, start_date_minute, end_date_minute)

            if df_minute_raw.empty:
                return ticker, {'daily': df_daily, 'hourly': pd.DataFrame(), 'minute': pd.DataFrame()}

            # --- Minute data cleaning logic remains the same ---
            df_minute = df_minute_raw.copy(deep=True).reset_index().rename(columns=str.lower)
            if 'datetime' in df_minute.columns:
                 df_minute = df_minute.rename(columns={'datetime': 'date'})
            if 'date' not in df_minute.columns:
                return ticker, {'daily': df_daily, 'hourly': pd.DataFrame(), 'minute': pd.DataFrame()}
            df_minute['date'] = pd.to_datetime(df_minute['date']).dt.tz_localize(None)
            df_minute = df_minute.sort_values(by='date').drop_duplicates(subset='date').reset_index(drop=True)
            
            if not required_cols.issubset(df_minute.columns):
                 return ticker, {'daily': df_daily, 'hourly': pd.DataFrame(), 'minute': pd.DataFrame()}

            df_hourly = df_minute.set_index('date').resample('60min').agg(
                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            ).dropna().reset_index()

            return ticker, {
                'daily': df_daily, 'hourly': df_hourly, 'minute': df_minute
            }
        except Exception as e:
            logger.error(f"CRITICAL FAILURE processing {ticker}: {e}", exc_info=True)
            return ticker, {}

    def load_data_in_parallel(self):
        """Preloads all historical data from yfinance into memory in parallel."""
        logger.info("Starting parallel download of historical data from yfinance...")
        all_tickers = [Config.NIFTY_50_SYMBOL] + [s['symbol'] for s in self.stocks_to_process]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.YFINANCE_THREADS) as executor:
            # Now maps to the class method self._fetch_data_for_ticker
            results = executor.map(self._fetch_data_for_ticker, all_tickers)

        for ticker, data in results:
            key = 'nifty' if ticker == Config.NIFTY_50_SYMBOL else ticker
            self.data_dict[key] = data

        # Check if Nifty data is valid (using .get() for safety)
        nifty_data = self.data_dict.get('nifty', {})
        if not nifty_data or nifty_data.get('daily', pd.DataFrame()).empty:
            logger.error("Failed to fetch valid Nifty 50 data. Cannot proceed with this batch.")
            return False
            
        logger.info("All data for this batch has been downloaded.")
        return True

    def precompute_indicators(self):
        """Precomputes all necessary MAs and Nifty data."""
        # This function was already well-structured, no changes needed.
        logger.info("Pre-computing moving averages for the batch...")
        for stock in self.stocks_to_process:
            symbol = stock['symbol']
            if symbol not in self.data_dict: continue
            self.ma_dict[symbol] = {}
            for interval in ['daily', 'hourly']:
                source_df = self.data_dict[symbol].get(interval)
                if source_df is None or source_df.empty: continue
                
                ma_df = pd.DataFrame()
                ma_df['date'] = source_df['date'].copy()
                close_prices = source_df['close']
                
                if isinstance(close_prices, pd.DataFrame):
                    close_prices = close_prices.iloc[:, 0]

                for period in Config.MA_PERIODS:
                    if len(close_prices) >= period:
                        ma_df[f'ma{period}'] = close_prices.rolling(window=period).mean().shift(1)
                    else:
                        ma_df[f'ma{period}'] = pd.NA
                self.ma_dict[symbol][interval] = ma_df.copy()
        
        nifty_daily = self.data_dict['nifty']['daily']
        nifty_daily['ma50'] = nifty_daily['close'].rolling(50).mean()
        self.nifty_50ma_dict = dict(zip(nifty_daily['date'].dt.date, nifty_daily['ma50']))
        
        logger.info("All indicators for this batch precomputed.")