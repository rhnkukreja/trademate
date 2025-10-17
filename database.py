"""This module isolates all Supabase logic into a single, reusable class."""

# database.py
import logging
from supabase import create_client, Client
from config import Config

logger = logging.getLogger(__name__)

class SupabaseHandler:
    """
    Handles all communication with the Supabase database.
    """
    def __init__(self, url: str, key: str):
        if not all([url, key]):
            raise ValueError("Supabase URL and Key must be provided.")
        self.client: Client = create_client(url, key)

    def get_processed_stock_dates(self) -> set:
        """Fetch (symbol, breakout_date) pairs from Supabase to avoid reprocessing."""
        try:
            response = self.client.table(Config.RESULTS_TABLE_NAME).select('symbol, breakout_date').execute()
            processed = {(row['symbol'], row['breakout_date']) for row in response.data}
            logger.info(f"Found {len(processed)} already processed (symbol, date) pairs in Supabase.")
            return processed
        except Exception as e:
            logger.error(f"Failed to fetch processed stock dates from Supabase: {e}")
            return set()

    def batch_upsert(self, data: list):
        """Upserts data into the results table in batches."""
        if not data:
            logger.info("No data to upsert.")
            return
        
        table_name = Config.RESULTS_TABLE_NAME
        batch_size = Config.UPSERT_BATCH_SIZE
        
        logger.info(f"Attempting to upsert {len(data)} rows to '{table_name}'.")
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                self.client.table(table_name).upsert(
                    batch,
                    on_conflict='symbol,breakout_date'
                ).execute()
                logger.info(f"Successfully upserted batch {i//batch_size + 1}/{len(data)//batch_size + 1}.")
            except Exception as e:
                logger.error(f"Supabase upsert failed for batch starting at index {i}: {e}")
                logger.error(f"Failed batch data sample: {batch[:2]}")