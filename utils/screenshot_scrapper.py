import argparse
import asyncio
import os
from datetime import datetime
import shutil
import uuid
from pathlib import Path
from playwright.async_api import async_playwright
import time
import os
import json
from dotenv import load_dotenv
import base64

load_dotenv()

# --- RESTORED ORIGINAL CONSTANTS ---
LOGIN_CHECK_TIMEOUT = 120_000
NAV_TIMEOUT = 60_000
PAGE_LOAD_WAIT = 4_000       # Back to 4s (Reliable)
FULL_RENDER_WAIT = 2_000     # Back to 2s (Reliable)
VIEWPORT = {"width": 1600, "height": 900}
HEADLESS = True

TV_CHART_URL = "https://www.tradingview.com/chart/kJ0Io8nr/"

class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None
    async def __aenter__(self):
        self.start = time.time()
        print(f"[timer] {self.name} started...")
        return self
    async def __aexit__(self, *args):
        print(f"[timer] {self.name} completed in {time.time() - self.start:.2f}s")


async def make_chart_screenshot(page, symbol, interval, target_dt, bars_needed, out_path, max_retries=2):
    """
    Captures a TradingView chart screenshot with retry logic.
    """
    for attempt in range(max_retries):
        try:
            step_start = time.time()
            
            # 1. Map interval to TradingView format
            tv_interval = interval
            if interval == "60":
                tv_interval = "60"  # 1h
            elif interval == "D":
                tv_interval = "1D"
            
            # 2. Construct URL using the WORKING approach from screenshot_scrapper_2
            safe_symbol = symbol.replace("&", "_")
            tv_symbol = f"NSE:{safe_symbol}" if ":" not in safe_symbol else safe_symbol

            chart_url = f"{TV_CHART_URL}?symbol={tv_symbol}&interval={tv_interval}"
            
            print(f"Loading {interval} chart for {symbol}... (Attempt {attempt + 1}/{max_retries})")

            # 3. Navigate with LONGER timeout (like screenshot_scrapper_2)
            await page.goto(chart_url, timeout=60000, wait_until="domcontentloaded")

            # Force clean layout
            await page.evaluate("""
                const selectors = [
                    'div[class*="header-toolbar"]',
                    'div[class*="side-toolbar"]',
                    '.logo',
                    '.watermark'
                ];

                const hideElements = document.querySelectorAll(selectors.join(', '));
                hideElements.forEach(el => {
                    el.style.display = 'none';
                });

                // Force dark
                document.documentElement.classList.add('theme-dark');
            """)

            await page.wait_for_timeout(2000)  # Let JS run

            print(f"[timer] [{interval}] Page goto: {time.time() - step_start:.2f}s")
            step_start = time.time()
            
            # 4. Wait for page to load completely
            await page.wait_for_timeout(4000)  # 4 seconds as in screenshot_scrapper_2
            print(f"[timer] [{interval}] Initial page load wait: {time.time() - step_start:.2f}s")
            step_start = time.time()

            # 5. Reset Chart View (Do this FIRST - from screenshot_scrapper_2)
            try:
                await page.keyboard.press("Alt+KeyR")
                await page.wait_for_timeout(1000)
            except:
                pass
            print(f"[timer] [{interval}] Alt+R jump: {time.time() - step_start:.2f}s")
            step_start = time.time()

            # 6. Final Render Wait
            await page.wait_for_timeout(3000)  # 1s + 2s as in screenshot_scrapper_2
            print(f"[timer] [{interval}] Final render wait: {time.time() - step_start:.2f}s")
            step_start = time.time()

            # 7. Capture Screenshot
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                await page.add_style_tag(content="""
                    div[class*="dialog-"], div[class*="modal-"], div[class*="popup-"], 
                    div[class*="overlay-"], div[data-role="dialog"], 
                    div[data-role="toast-container"], 
                    div[id^="overlap-manager-root"] div[class*="item-"] {
                        display: none !important;
                    }
                """)
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(500)
            except: pass

            # Wait until indicators are actually mounted
            await page.wait_for_selector(
                'div[class*="legend"]',
                timeout=15000
            )
            
            await page.screenshot(
                path=str(out_path), 
                full_page=False,
                timeout=60000
            )
            
            print(f"[timer] [{interval}] Screenshot capture: {time.time() - step_start:.2f}s")
            print(f"[OK] Saved: {out_path.name}")
            return  # Success, exit function
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] Screenshot failed (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(3)  # Wait 3 seconds before retry
                continue
            else:
                print(f"[ERROR] Failed to capture screenshot for {symbol} ({interval}) after {max_retries} attempts: {e}")
                raise e

async def run(symbol: str, dt_str: str, out_dir: str):
    safe_symbol_for_file = symbol.replace(":", "_")
    target_dt = datetime.fromisoformat(dt_str)
    day_num = target_dt.day
    suffix = "th" if 11 <= day_num <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day_num % 10, "th")
    date_str = f"{day_num}{suffix}{target_dt.strftime('%b%Y')}"

    safe_symbol_for_file = symbol.replace(':', '_')
    # Treat out_dir as the final folder where screenshots are saved
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    overall_start = time.time()
    print(f"[timer] ===== OVERALL START =====")

    try:
        async with async_playwright() as p:
            # Launch one context
            browser = await p.chromium.launch(
                headless=HEADLESS,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"],
            )

            tv_state_b64 = os.getenv("TV_STATE_JSON_B64")

            if not tv_state_b64:
                raise Exception("TV_STATE_JSON_B64 not found in .env")

            decoded = base64.b64decode(tv_state_b64)
            tv_state_dict = json.loads(decoded)

            context = await browser.new_context(
                viewport=VIEWPORT,
                storage_state=tv_state_dict
            )

            page_15 = await context.new_page()
            page_60 = await context.new_page()
            page_D  = await context.new_page()
            
            async with Timer("Taking all 3 screenshots in parallel"):
                # Extract clean symbol name (remove exchange)
                symbol_short = symbol.split(":")[-1]

                await asyncio.gather(
                    make_chart_screenshot(
                        page_15, symbol, "15", target_dt, 3,
                        out_path / f"{symbol_short}_{date_str}_15m.png"
                    ),
                    make_chart_screenshot(
                        page_60, symbol, "60", target_dt, 12,
                        out_path / f"{symbol_short}_{date_str}_1h.png"
                    ),
                    make_chart_screenshot(
                        page_D, symbol, "D", target_dt, 90,
                        out_path / f"{symbol_short}_{date_str}_D.png"
                    ),
                )
            await context.close()
        
    except Exception as e:
        print(f"CRITICAL ERROR in screenshot run: {e}")
        # Re-raise to ensure the caller knows it failed
        raise e
    
    print(f"[timer] ===== OVERALL COMPLETED IN {time.time() - overall_start:.2f}s =====\n")

def parse_args():
    parser = argparse.ArgumentParser("TradingView Screenshot Agent")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--dt", required=True)
    parser.add_argument("--out", default="./out")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    HEADLESS = args.headless or HEADLESS
    asyncio.run(run(args.symbol, args.dt, args.out))