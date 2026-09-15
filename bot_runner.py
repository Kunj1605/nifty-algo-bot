from datetime import datetime, time as dtime
import math
import os
import numpy as np
import pandas as pd
import requests

# ==========================================
# CONFIGURATION & SECRETS
# ==========================================
API_BASE_URL = "https://sandbox.dhan.co/v2"
ACCESS_TOKEN = os.getenv("DHAN_SANDBOX_TOKEN")

headers = {
    "access-token": ACCESS_TOKEN,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

LOT_SIZE = 65
RISK_FREE_RATE = 0.06
IMPLIED_VOLATILITY = 0.18


# ==========================================
# EXACT BACKTEST MATH & BLACK-SCHOLES
# ==========================================
def norm_cdf(x):
  return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def black_scholes(S, K, T, r, sigma, option_type="C"):
  if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
    return max(0.0, S - K) if option_type == "C" else max(0.0, K - S), 0, 0, 0, 0
  d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
  d2 = d1 - sigma * math.sqrt(T)
  if option_type == "C":
    price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
  else:
    price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
  return max(1.0, price), 0, 0, 0, 0


def calculate_time_to_expiry(current_dt):
  days_to_tue = (1 - current_dt.weekday()) % 7
  expiry_dt = current_dt.normalize() + pd.Timedelta(days=days_to_tue)
  expiry_dt = expiry_dt.replace(hour=15, minute=30)
  if current_dt >= expiry_dt:
    expiry_dt += pd.Timedelta(days=7)
  diff_seconds = (expiry_dt - current_dt).total_seconds()
  return max(diff_seconds / (365.25 * 24 * 3600), 1 / (365.25 * 24 * 60))


def check_open_position():
  """Queries Dhan API to ensure we don't open duplicate trades"""
  try:
    res = requests.get(f"{API_BASE_URL}/positions", headers=headers)
    if res.status_code == 200:
      positions = res.json().get("data", [])
      for p in positions:
        if p.get("securityId") == "13" and int(p.get("netQty", 0)) != 0:
          return True
  except Exception as e:
    print(f"Error checking positions: {e}")
  return False


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def run_bot():
  print(
      f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running strategy"
      " check..."
  )

  # 1. Check if a position is already active
  if check_open_position():
    print("Position already active in portfolio. Holding existing trade.")
    return

  # 2. Fetch today's intraday 5-minute candles from Dhan Sandbox
  url = f"{API_BASE_URL}/charts/intraday"
  today_str = datetime.now().strftime("%Y-%m-%d")
  payload = {
      "securityId": "13",
      "exchangeSegment": "IDX_I",
      "instrument": "INDEX",
      "interval": "5",
      "oi": "false",
      "fromDate": today_str,
      "toDate": today_str,
  }

  response = requests.post(url, json=payload, headers=headers)
  if response.status_code != 200:
    print(f"API Error: {response.status_code} - {response.text}")
    return

  data = response.json()
  df = pd.DataFrame({
      "timestamp": data.get("timestamp", []),
      "open": data.get("open", []),
      "high": data.get("high", []),
      "low": data.get("low", []),
      "close": data.get("close", []),
      "volume": data.get("volume", []),
  })

  if df.empty or len(df) < 200:
    print(
        f"Insufficient candles fetched ({len(df)}). Need at least 200 for"
        " indicators."
    )
    return

  df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
  df = df.sort_values("datetime").reset_index(drop=True)

  # 3. Exact 7-Year Backtest Indicator Formulas
  df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
  df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
  df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

  prev_close_s = df["close"].shift(1)
  tr = pd.concat(
      [
          df["high"] - df["low"],
          (df["high"] - prev_close_s).abs(),
          (df["low"] - prev_close_s).abs(),
      ],
      axis=1,
  ).max(axis=1)
  df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

  delta = df["close"].diff()
  gain = delta.where(delta > 0, 0.0)
  loss = (-delta.where(delta < 0, 0.0))
  avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
  rs = avg_gain / avg_loss
  df["rsi"] = 100 - (100 / (1 + rs))
  df.loc[avg_loss == 0, "rsi"] = 100.0

  # 4. Get Latest Candle Metrics
  i = len(df) - 1
  curr_open = df.loc[i, "open"]
  curr_high = df.loc[i, "high"]
  curr_low = df.loc[i, "low"]
  curr_close = df.loc[i, "close"]
  t_stamp = df.loc[i, "datetime"]
  t_time = t_stamp.time()

  ema21 = df.loc[i, "ema21"]
  ema50 = df.loc[i, "ema50"]
  ema200 = df.loc[i, "ema200"]
  atr = df.loc[i, "atr"]
  rsi = df.loc[i, "rsi"]
  prev_rsi = df.loc[i - 1, "rsi"]

  # Active Trading Session Window (09:15 to 14:30)
  session_start = dtime(9, 15)
  session_end = dtime(14, 30)
  if not (session_start <= t_time <= session_end):
    print(
        f"Current time {t_time} is outside active trading window (09:15-14:30)."
    )
    return

  # 5. Exact Entry Conditions from Backtest
  long_cond = (
      (curr_close > ema21)
      and (ema21 > ema50)
      and (ema50 > ema200)
      and (rsi > 55)
      and (prev_rsi <= 55)
      and (curr_low <= ema21)
  )

  short_cond = (
      (curr_close < ema21)
      and (ema21 < ema50)
      and (ema50 < ema200)
      and (rsi < 45)
      and (prev_rsi >= 45)
      and (curr_high >= ema21)
  )

  if long_cond or short_cond:
    position = 1 if long_cond else -1
    strike = round(curr_open / 50) * 50
    T = calculate_time_to_expiry(t_stamp)

    entry_prem, _, _, _, _ = black_scholes(
        curr_open,
        strike,
        T,
        RISK_FREE_RATE,
        IMPLIED_VOLATILITY,
        "C" if position == 1 else "P",
    )
    entry_prem *= 1.025  # Slippage adjustment
    active_sl = max(0.05, entry_prem - (atr * 0.25))
    active_tp = entry_prem + (atr * 1.2)

    trade_type = "LONG CE" if position == 1 else "SHORT PE"
    print(
        f">>> SIGNAL TRIGGERED: {trade_type} | Strike: {strike} | Entry"
        f" Spot: {curr_open}"
    )

    # Place Mock Order via Sandbox API
    order_url = f"{API_BASE_URL}/orders"
    payload = {
        "securityId": "13",
        "exchangeSegment": "IDX_I",
        "transactionType": "BUY",
        "quantity": LOT_SIZE,
        "orderType": "MARKET",
        "productType": "INTRADAY",
        "price": curr_open,
    }
    res = requests.post(order_url, json=payload, headers=headers)
    print("Order API Response:", res.json())
  else:
    print(
        f"Checked Candle [{t_stamp.strftime('%H:%M')}]: Close={curr_close},"
        f" RSI={rsi:.2f} — No trigger conditions met."
    )


if __name__ == "__main__":
  run_bot()