import os
import requests

print("="*50)
print(" DHAN API DIAGNOSTIC TEST ")
print("="*50)

# 1. Check Token Loading
ACCESS_TOKEN = (os.getenv("DHAN_ACCESS_TOKEN") or os.getenv("DHAN_SANDBOX_TOKEN") or "").strip()
CLIENT_ID = "2609151409"

print(f"Token Loaded Successfully: {'YES' if ACCESS_TOKEN else 'NO'}")
if ACCESS_TOKEN:
    print(f"Token Length: {len(ACCESS_TOKEN)} characters")
    print(f"Token Prefix: {ACCESS_TOKEN[:10]}********")

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# ---------------------------------------------------------
# TEST 1: TRADING API (Can we check funds / place orders?)
# ---------------------------------------------------------
print("\n>>> TEST 1: TRADING API (Checking Funds) <<<")
try:
    url_funds = "https://sandbox.dhan.co/v2/fundlimit"
    res1 = requests.get(url_funds, headers=HEADERS, timeout=10)
    print(f"HTTP Status: {res1.status_code}")
    print(f"Server Response: {res1.text[:200]}")
except Exception as e:
    print(f"Failed to connect: {e}")

# ---------------------------------------------------------
# TEST 2: DATA API (Can we download candles for EMAs/RSI?)
# ---------------------------------------------------------
print("\n>>> TEST 2: DATA API (Fetching Chart Candles) <<<")
try:
    url_charts = "https://sandbox.dhan.co/v2/charts/intraday"
    payload = {
        "securityId": "13",
        "exchangeSegment": "IDX_I",
        "instrument": "INDEX",
        "interval": "5",
        "fromDate": "2026-09-15",
        "toDate": "2026-09-15"
    }
    res2 = requests.post(url_charts, headers=HEADERS, json=payload, timeout=10)
    print(f"HTTP Status: {res2.status_code}")
    print(f"Server Response: {res2.text[:200]}")
except Exception as e:
    print(f"Failed to connect: {e}")

print("\n" + "="*50)
