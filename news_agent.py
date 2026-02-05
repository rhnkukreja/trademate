import yfinance as yf
import datetime
from common import logger

def get_stock_news(symbol):
    """
    Fetches news for a given symbol from Yahoo Finance.
    Uses an aggressive search for titles and timestamps.
    """
    yf_symbol = f"{symbol}.NS"
    ticker = yf.Ticker(yf_symbol)
    
    try:
        news_list = ticker.news
        logger.info(f"[Yahoo News Debug] Raw news count for {yf_symbol}: {len(news_list) if news_list else 0}")
        if not news_list:
            return None

        recent_news = []
        now = datetime.datetime.now(datetime.timezone.utc)
        lookback_window = now - datetime.timedelta(days=1)

        for item in news_list:
            # --- Aggressive Title Search ---
            # Try top-level, then try inside the 'content' block
            title = item.get('title') or item.get('headline')
            if not title and 'content' in item:
                title = item.get('content', {}).get('title')
            
            # --- Aggressive Timestamp Search ---
            ts = item.get('providerPublishTime')
            if not ts and 'content' in item:
                ts = item.get('content', {}).get('pubDate')

            if title and ts:
                try:
                    # Handle int timestamps
                    if isinstance(ts, (int, float)):
                        publish_time = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)

                    # Handle ISO date strings (THIS WAS MISSING)
                    elif isinstance(ts, str):
                        publish_time = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))

                    else:
                        continue

                    if publish_time > lookback_window:
                        recent_news.append(f"[{publish_time.strftime('%Y-%m-%d %H:%M')}] {title}")
                except Exception:
                    continue

        # Join results or return None if still empty
        return " | ".join(recent_news) if recent_news else None

    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return None