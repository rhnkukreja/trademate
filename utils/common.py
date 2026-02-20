import os
import logging
import sys
from decimal import Decimal, ROUND_CEILING
from dotenv import load_dotenv
from supabase import create_client, Client
from kiteconnect import KiteConnect
import math
import json
import time
try:
    import numpy as np
except Exception:
    np = None

# -------------------------- Configuration & Setup --------------------------

# Load .env file
load_dotenv()

# Setup Logger
# This configures the root logger. Other files can just get their own logger.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('app_shared.log', encoding='utf-8'), # A shared log file
        logging.StreamHandler()
    ]
)
# Get a logger for this common module
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load Environment Variables
SUPABASE_URL = os.getenv("K_SUPABASE_URL")
SUPABASE_KEY = os.getenv("K_SUPABASE_KEY")
KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

# Validate Environment Variables
if not all([SUPABASE_URL, SUPABASE_KEY]):
    logger.critical("Required environment variables are not set. Exiting.")
    # Use sys.exit() in a shared module to stop everything if config is missing
    sys.exit("Critical Error: Missing environment variables.")

# -------------------------- Client Initialization --------------------------

try:
    import httpx
    from supabase import create_client, Client

    # -------------------------------
    # 1️⃣ Low-level HTTP connectivity check (FIXED)
    # -------------------------------
    logger.info("Testing direct Supabase HTTP connectivity...")

    try:
        response = httpx.get(
            f"{SUPABASE_URL}/rest/v1/monitor_list",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            },
            params={"select": "symbol", "limit": 1},
            verify=False,
            timeout=5
        )

        logger.info(f"Supabase HTTP status: {response.status_code}")

    except Exception as e:
        logger.warning(f"Direct Supabase HTTP check failed: {e}")

    # -------------------------------
    # 2️⃣ Initialize Supabase Client
    # -------------------------------
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Disable SSL verification (Windows workaround)
    supabase.postgrest.session = httpx.Client(verify=False)

    # 🔍 Warmup query (REAL health check)
    try:
        supabase.table("monitor_list").select("symbol").limit(1).execute()
        logger.info("Supabase connectivity OK")
    except Exception as e:
        logger.error(
            f"Supabase unreachable at startup (will use Excel fallback): {e}"
        )
        logger.warning("Will retry Supabase on actual operations...")
        supabase = None

    # -------------------------------
    # 3️⃣ Initialize Kite Client (CRITICAL)
    # -------------------------------
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)

    # Fail fast if Kite is broken
    kite.profile()
    logger.info("Successfully initialized Kite client.")

except Exception as e:
    logger.critical(f"Failed to initialize critical clients: {e}")
    sys.exit("Critical Error: Client initialization failed.")

# -------------------------- Helper Functions --------------------------

def next_price_above(value, tick):
    """Calculates the next valid price tick above a given value."""
    dv = Decimal(str(value))
    dt = Decimal(str(tick))
    n = (dv / dt).to_integral_value(rounding=ROUND_CEILING)
    candidate = n * dt
    if candidate <= dv:
        candidate = (n + 1) * dt
    return float(candidate)

def to_json_safe(value):
    """
    Convert values to JSON-safe:
    - NaN/Inf -> None
    - numpy scalars -> python scalars
    - recursively handle dict/list
    """
    if value is None:
        return None

    # numpy scalar -> python scalar
    if np is not None:
        try:
            if isinstance(value, (np.floating, np.integer, np.bool_)):
                value = value.item()
        except Exception:
            pass

    # float NaN/Inf -> None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_json_safe(v) for v in value]

    return value

def batch_upsert_supabase(table_name, data, batch_size=100):
    """Upserts data into Supabase in batches with retry & sanitization."""
    if not data:
        logger.info("No data to upsert.")
        return

    MAX_RETRIES = 3

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        # ✅ sanitize NaN / Inf / numpy
        batch = [{k: to_json_safe(v) for k, v in row.items()} for row in batch]

        # strict JSON validation
        json.dumps(batch, allow_nan=False)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if table_name == "live_breakouts":
                    conflict_key = "symbol,breakout_date"
                elif table_name == "live_ml_features":
                    conflict_key = "symbol,date"
                else:
                    conflict_key = "symbol,date"

                if supabase is None:
                    raise Exception("Supabase client not available")

                supabase.table(table_name).upsert(
                    batch, on_conflict=conflict_key
                ).execute()

                logger.info(
                    f"Upserted batch {i//batch_size + 1} to {table_name}."
                )
                break  # ✅ success → exit retry loop

            except Exception as e:
                logger.error(
                    f"[batch_upsert_supabase] Attempt {attempt}/{MAX_RETRIES} "
                    f"failed for {table_name}: {e}"
                )

                if attempt == MAX_RETRIES:
                    logger.error(
                        f"[batch_upsert_supabase] Giving up batch "
                        f"{i//batch_size + 1} ({len(batch)} rows)"
                    )
                    raise Exception("Supabase batch upsert failed")
                else:
                    time.sleep(2* attempt)

# -------------------------- Token Map --------------------------

token_map = {}

def load_token_map():
    global token_map
    try:
        instruments = kite.instruments("NSE")
        token_map = {ins["tradingsymbol"]: ins["instrument_token"] for ins in instruments}
        logger.info(f"Loaded {len(token_map)} NSE instrument tokens.")
    except Exception as e:
        logger.error(f"Failed to load token map: {e}")
