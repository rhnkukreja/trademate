import os
import time
import pyotp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from kiteconnect import KiteConnect
from utils.common import supabase, KITE_API_KEY, KITE_API_SECRET

# Credentials from Render/Env
USER_ID = os.getenv("KITE_USER_ID")
PASSWORD = os.getenv("KITE_PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")

def get_kite_access_token():
    kite = KiteConnect(api_key=KITE_API_KEY)
    login_url = kite.login_url()

    # 1. Setup Headless Chrome for Render environment
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(login_url)
        time.sleep(2)

        # 2. Login Flow
        driver.find_element(By.XPATH, '//input[@type="text"]').send_keys(USER_ID)
        driver.find_element(By.XPATH, '//input[@type="password"]').send_keys(PASSWORD)
        driver.find_element(By.XPATH, '//button[@type="submit"]').click()
        time.sleep(2)

        # 3. Use pyotp to bypass manual 2FA
        totp = pyotp.TOTP(TOTP_SECRET).now() # Generates the 6-digit code
        driver.find_element(By.XPATH, '//input[@type="number"]').send_keys(totp)
        driver.find_element(By.XPATH, '//button[@type="submit"]').click()
        time.sleep(3)

        # 4. Extract Request Token
        current_url = driver.current_url
        request_token = current_url.split("request_token=")[1].split("&")[0]
        
        # 5. Get the Session
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        return data["access_token"]

    finally:
        driver.quit()

def update_supabase_token(token):
    if supabase:
        supabase.table("kite_config").upsert(
            {"key_name": "access_token", "value": token},
            on_conflict="key_name"
        ).execute()
        print(f"✅ Access Token updated at {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    try:
        new_token = get_kite_access_token()
        update_supabase_token(new_token)
    except Exception as e:
        print(f"❌ Automation failed: {e}")