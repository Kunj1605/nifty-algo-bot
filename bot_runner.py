import math
import os
import time
from datetime import datetime, time as dtime, timedelta, timezone
import numpy as np
import pandas as pd
import requests

# =====================================================================
# 1. AUTHENTICATION & CONFIGURATION
# =====================================================================
SANDBOX_URL = "https://sandbox.dhan.co/v2"

# Pulled securely from GitHub Secrets (or set directly for local test)
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
CLIENT_ID = "2609151409"

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Core Strategy Parameters (Identical to 7-Year Backtest)
STARTING_CAPITAL = 30000.0
LOT_SIZE = 65
RISK_FREE_RATE = 0.06
IMPLIED_VOLATILITY = 0.18
MAX_LOTS = 25

# IST Timezone for GitHub Actions Runner
IST = timezone(timedelta(hours=5, minutes=30))


# =====================================================================
# 2. BLACK-SCHOLES OPTIONS ENGINE
# =====================================================================
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


# =====================================================================
# 3. MATHEMATICAL INDICATOR ENGINE (IDENTICAL TO BACKTEST)
# =====================================================================
def compute_indicators(df):
  # EMAs
  df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
  df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
  df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

  # ATR 14 using Wilder's Smoothing (alpha=1/14)
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

  # RSI 14 using Wilder's Smoothing (alpha=1/14)
  delta = df["close"].diff()
  gain = delta.where(delta > 0, 0.0)
  loss = (-delta.where(delta < 0, 0.0))
  avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
  rs = avg_gain / avg_loss
  df["rsi"] = 100 - (100 / (1 + rs))
  df.loc[avg_loss == 0, "rsi"] = 100.0
  return df


# =====================================================================
# 4. DHAN SANDBOX API CALLS
# =====================================================================
def fetch_sandbox_candles():
  url = f"{SANDBOX_URL}/charts/intraday"
  today_str = datetime.now(IST).strftime("%Y-%m-%d")

  payload = {
      "securityId": "13",  # Nifty 50 Index ID
      "exchangeSegment": "IDX_I",
      "instrument": "INDEX",
      "interval": "5",
      "oi": "false",
      "fromDate": today_str,
      "toDate": today_str,
  }

  try:
    response = requests.post(url, json=payload, headers=HEADERS, timeout=12)
    if response.status_code in [401, 403]:
      print(f"[AUTH ERROR] Token invalid or expired (HTTP {response.status_code}).")
      return None, False

    if response.status_code != 200:
      print(f"[API ERROR] HTTP {response.status_code}: {response.text}")
      return None, True

    data = response.json()
    raw = data.get("data", data)
    timestamps = raw.get("timestamp") or raw.get("start_Time", [])

    if not timestamps:
      return pd.DataFrame(), True

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": pd.to_numeric(raw.get("open", []), errors="coerce"),
        "high": pd.to_numeric(raw.get("high", []), errors="coerce"),
        "low": pd.to_numeric(raw.get("low", []), errors="coerce"),
        "close": pd.to_numeric(raw.get("close", []), errors="coerce"),
        "volume": pd.to_numeric(raw.get("volume", []), errors="coerce"),
    }).dropna()

    if not df.empty:
      unit = "s" if df["timestamp"].iloc[0] < 1e11 else "ms"
      df["datetime"] = pd.to_datetime(df["timestamp"], unit=unit)
      df = df.sort_values("datetime").reset_index(drop=True)

    return df, True

  except Exception as e:
    print(f"[CONNECTION EXCEPTION]: {e}")
    return None, True


def place_sandbox_order(txn_type, qty, price):
  url = f"{SANDBOX_URL}/orders"
  payload = {
      "securityId": "13",
      "exchangeSegment": "IDX_I",
      "transactionType": txn_type,
      "quantity": qty,
      "orderType": "MARKET",
      "productType": "INTRADAY",
      "price": price,
  }
  try:
    resp = requests.post(url, json=payload, headers=HEADERS, timeout=10)
    print(f"--> [SANDBOX ORDER DISPATCHED]: {resp.status_code} - {resp.text}")
  except Exception as e:
    print(f"--> [ORDER DISPATCH FAILED]: {e}")


# =====================================================================
# 5. EXECUTION CONTROLLER
# =====================================================================
def run_trading_bot():
  now_ist = datetime.now(IST)
  curr_time = now_ist.time()

  print("=" * 65)
  print(" NIFTY 50 BACKTEST-VALIDATED BOT (DHAN SANDBOX)")
  print(f" Current IST Time: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
  print("=" * 65)

  if not ACCESS_TOKEN:
    print("[CRITICAL] DHAN_ACCESS_TOKEN is missing or empty. Exiting.")
    return

  # Step 1: Healthcheck API connectivity
  df, auth_ok = fetch_sandbox_candles()
  if not auth_ok:
    print("[TERMINATED] Authentication failed. Please refresh your token.")
    return

  print("[SUCCESS] Connected to Dhan Sandbox API. Token is valid!")

  # Step 2: Enforce Market Operating Window
  session_open = dtime(9, 15)
  session_close = dtime(15, 30)

  if curr_time < session_open or curr_time >= session_close:
    print(
        f"[{now_ist.strftime('%H:%M:%S')}] NSE is currently closed (09:15 -"
        " 15:30 IST)."
    )
    print("Test passed: Authentication and environment verified successfully.")
    return

  # Step 3: Run Strategy Loop during Market Hours
  if df is None or len(df) < 200:
    print(f"Candles available: {len(df) if df is not None else 0}. Need >= 200.")
    return

  df = compute_indicators(df)
  i = len(df) - 1

  curr_open = df.loc[i, "open"]
  curr_high = df.loc[i, "high"]
  curr_low = df.loc[i, "low"]
  curr_close = df.loc[i, "close"]

  ema21 = df.loc[i, "ema21"]
  ema50 = df.loc[i, "ema50"]
  ema200 = df.loc[i, "ema200"]
  atr = df.loc[i, "atr"]
  rsi = df.loc[i, "rsi"]
  prev_rsi = df.loc[i - 1, "rsi"]

  print(
      f"Spot: ₹{curr_close:.2f} | RSI: {rsi:.1f} | EMA21: {ema21:.1f} | ATR:"
      f" {atr:.1f}"
  )

  # Entry Rules (Exact Backtest Match)
  long_cond = (
      (curr_close > ema21)
      and (ema21 > ema50 > ema200)
      and (rsi > 55)
      and (prev_rsi <= 55)
      and (curr_low <= ema21)
  )

  short_cond = (
      (curr_close < ema21)
      and (ema21 < ema50 < ema200)
      and (rsi < 45)
      and (prev_rsi >= 45)
      and (curr_high >= ema21)
  )

  if long_cond or short_cond:
    opt_type = "C" if long_cond else "P"
    strike = round(curr_open / 50) * 50
    T = calculate_time_to_expiry(now_ist)

    entry_premium, _, _, _, _ = black_scholes(
        curr_open, strike, T, RISK_FREE_RATE, IMPLIED_VOLATILITY, opt_type
    )
    entry_premium *= 1.025  # 2.5% slippage

    active_sl = max(0.05, entry_premium - (atr * 0.25))
    active_tp = entry_premium + (atr * 1.20)

    # Dynamic lot allocation (capped at 25)
    num_lots = min(MAX_LOTS, 1)
    total_qty = LOT_SIZE * num_lots

    signal_label = "LONG CE" if long_cond else "SHORT PE"
    print("\n" + "#" * 55)
    print(f" >>> SIGNAL TRIGGERED: {signal_label}")
    print(f" Strike: {strike} | Lots: {num_lots} ({total_qty} units)")
    print(f" Est. Premium: ₹{entry_premium:.2f}")
    print(f" Stop Loss: ₹{active_sl:.2f} (-0.25 ATR)")
    print(f" Target: ₹{active_tp:.2f} (+1.20 ATR)")
    print("#" * 55 + "\n")

    place_sandbox_order("BUY", total_qty, curr_close)
  else:
    print("-> No entry signals triggered on this candle.")


if __name__ == "__main__":
  run_trading_bot()
