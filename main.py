from fastapi import FastAPI, Header, HTTPException, Request
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
from typing import Dict, Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
from dotenv import load_dotenv
import time
import json


# Global in-memory cache for yfinance downloads
yf_cache = {}          # key: "ticker_period_interval" → {'data': DataFrame, 'time': timestamp}
CACHE_TIMEOUT = 900    # 15 minutes - adjust if needed (300 = 5 min for more freshness)

load_dotenv()

app = FastAPI(title="BTC Macro & KPI API", version="1.0")

# Rate limiter: 100 requests per minute per IP (adjust as needed)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ================== CONFIG ==================
FRED_KEY = os.getenv("FRED_KEY") or "YOUR_FRED_API_KEY_HERE"
API_KEY = os.getenv("API_KEY") or "mysecretkey123"  # Change this! Or set in .env

# ================== FRED HELPERS (unchanged) ==================
def get_fred_observations(series_id: str, limit: int = 20):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_KEY}&file_type=json&limit={limit}&sort_order=desc"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["observations"]

def get_latest_fred(series_id: str) -> tuple[float, str]:
    obs = get_fred_observations(series_id, 5)
    for o in obs:
        if o["value"] != ".":
            return float(o["value"]), o["date"]
    raise ValueError(f"No data for {series_id}")

def get_yoy_fred(series_id: str) -> float:
    obs = get_fred_observations(series_id, 15)
    values = [float(o["value"]) for o in obs if o["value"] != "."]
    if len(values) >= 13:
        return round((values[0] - values[12]) / values[12] * 100, 2)
    return 0.0

# ================== yfinance cache download ==================
def cached_yf_download(ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """
    Cached wrapper around yf.download to avoid repeated API calls.
    Returns cached DataFrame if fresh, otherwise fetches and caches.
    """
    cache_key = f"{ticker}_{period}_{interval}"
    now = time.time()

    # Cache hit?
    if cache_key in yf_cache and now - yf_cache[cache_key]['time'] < CACHE_TIMEOUT:
        print(f"[CACHE HIT] {cache_key}")
        return yf_cache[cache_key]['data'].copy()  # return copy to avoid mutation issues

    # Cache miss or expired → fetch live
    print(f"[CACHE MISS] Fetching {cache_key}")
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=20)
        if data.empty:
            print(f"[DEBUG] yfinance returned empty for {cache_key}")
            return pd.DataFrame()

        # Store in cache
        yf_cache[cache_key] = {'data': data.copy(), 'time': now}
        return data.copy()
    except Exception as e:
        print(f"[DEBUG] yfinance fetch error for {cache_key}: {str(e)}")
        # Optional: return last known good data if available
        if cache_key in yf_cache:
            print(f"[FALLBACK] Using stale cache for {cache_key}")
            return yf_cache[cache_key]['data'].copy()
        return pd.DataFrame()  # empty fallback

# ================== yfinance PRICES ==================
def get_yf_price(ticker: str) -> float:
    try:
        data = yf.Ticker(ticker).history(period="5d")
        return round(float(data['Close'].iloc[-1]), 2)
    except:
        return 0.0

# ================== 200WMA ==================
def get_200wma() -> float:
    print("[DEBUG] Starting 200WMA...")
    try:
        btc_data = cached_yf_download('BTC-USD', period='max', interval='1d')
        if btc_data.empty:
            print("[DEBUG] yfinance empty")
            return 0.0

        weekly_closes = btc_data['Close'].resample('W').last().dropna()
        print(f"[DEBUG] Weekly points: {len(weekly_closes)}")

        if len(weekly_closes) < 100:
            print("[DEBUG] Not enough weeks")
            return 0.0

        last_200 = weekly_closes.tail(200)
        wma_series = last_200.mean()  # this is a Series (single value but labeled)
        wma = float(wma_series.item())  # extract scalar float safely
        # Alternative: wma = round(float(last_200.mean()), 2)

        print(f"[DEBUG] True 200WMA: {wma}")
        return wma
    except Exception as e:
        print(f"[DEBUG] yfinance error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0

# ================== REAL MAX PAIN FROM DERIBIT ==================
def calculate_max_pain() -> Dict:
    try:
        url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
        resp = requests.get(url, timeout=10).json()
        if 'result' not in resp or not resp['result']:
            return {"error": "No data from Deribit"}

        data = resp['result']
        btc_price = get_yf_price("BTC-USD")  # current price for USD conversion

        # 1. Group by expiry and strike
        by_expiry = {}
        for item in data:
            name = item['instrument_name']  # e.g. BTC-28FEB25-90000-C
            parts = name.split('-')
            if len(parts) != 4:
                continue
            expiry_str = parts[1]           # 28FEB25
            strike_str = parts[2]
            opt_type = parts[3]             # C or P
            strike = float(strike_str)
            oi = item.get('open_interest', 0)

            # Parse expiry to datetime (Deribit format: DDMMMYY)
            try:
                expiry_date = datetime.strptime(expiry_str, "%d%b%y")
                if expiry_date < datetime.now():
                    expiry_date += timedelta(days=365*100)  # very old → assume future century, but rare
            except:
                continue

            if expiry_date not in by_expiry:
                by_expiry[expiry_date] = {}
            if strike not in by_expiry[expiry_date]:
                by_expiry[expiry_date][strike] = {'call_oi': 0, 'put_oi': 0}
            if opt_type == 'C':
                by_expiry[expiry_date][strike]['call_oi'] += oi
            elif opt_type == 'P':
                by_expiry[expiry_date][strike]['put_oi'] += oi

        if not by_expiry:
            return {"error": "No valid options data"}

        # 2. Find the nearest future expiry
        future_expiries = [d for d in by_expiry if d > datetime.now()]
        if not future_expiries:
            return {"error": "No future expiries found"}
        next_expiry_date = min(future_expiries)
        next_expiry_str = next_expiry_date.strftime("%d %b %Y").upper().replace(" ", "")

        strikes = sorted(by_expiry[next_expiry_date].keys())
        if not strikes:
            return {"error": "No strikes for next expiry"}

        # 3. Compute true max pain: find price that minimizes total pain to writers
        min_pain = float('inf')
        max_pain_price = None

        for assumed_price in strikes:  # sample at strike levels (common & sufficient)
            call_pain = 0
            put_pain = 0
            for strike, ois in by_expiry[next_expiry_date].items():
                # Pain for writers: intrinsic value * OI * contract multiplier (Deribit BTC = 1 BTC)
                # Convert to USD pain
                if assumed_price > strike:  # call ITM
                    call_pain += (assumed_price - strike) * ois['call_oi']
                if assumed_price < strike:  # put ITM
                    put_pain += (strike - assumed_price) * ois['put_oi']

            total_pain_btc = call_pain + put_pain
            total_pain_usd = total_pain_btc * btc_price

            if total_pain_usd < min_pain:
                min_pain = total_pain_usd
                max_pain_price = assumed_price

        # Also compute highest OI strike for comparison
        max_oi_strike = max(by_expiry[next_expiry_date], key=lambda s: by_expiry[next_expiry_date][s]['call_oi'] + by_expiry[next_expiry_date][s]['put_oi'])
        total_oi_at_max = by_expiry[next_expiry_date][max_oi_strike]['call_oi'] + by_expiry[next_expiry_date][max_oi_strike]['put_oi']

        return {
            "max_pain_price": max_pain_price,
            "max_pain_usd_pain": round(min_pain, 2),
            "highest_oi_strike": max_oi_strike,
            "total_oi_at_highest": total_oi_at_max,
            "next_expiry": next_expiry_date.strftime("%Y-%m-%d"),
            "next_expiry_readable": next_expiry_date.strftime("%d %b %Y"),
            "note": "True max pain (min USD pain to writers) calculated over strikes"
        }

    except Exception as e:
        print(f"[DEBUG] Max pain error: {str(e)}")
        return {
            "error": str(e),
            "note": "Deribit fetch or calculation failed"
        }

# RSI (14-period) from daily closes
def get_rsi(period: int = 14) -> float:
    try:
        btc_data = cached_yf_download('BTC-USD', period='max', interval='1d')
        if btc_data.empty or len(btc_data) < period + 1:
            return None
        
        delta = btc_data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi.iloc[-1], 2)
    except Exception as e:
        print(f"[DEBUG] RSI error: {str(e)}")
        return None

# MACD (12,26,9) - returns dict with line, signal, histogram
def get_macd(fast=12, slow=26, signal=9) -> Dict:
    try:
        btc_data = cached_yf_download('BTC-USD', period='max', interval='1d')
        if btc_data.empty or len(btc_data) < slow + signal:
            return {"error": "Insufficient data"}
        
        ema_fast = btc_data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = btc_data['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd_line": round(macd_line.iloc[-1], 4),
            "signal_line": round(signal_line.iloc[-1], 4),
            "histogram": round(histogram.iloc[-1], 4)
        }
    except Exception as e:
        print(f"[DEBUG] MACD error: {str(e)}")
        return {"error": str(e)}

# Funding Rate (latest from Binance BTCUSDT perpetual)

def get_funding_rate() -> Dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    # Primary: OKX (reliable for US, public)
    try:
        url = "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP"
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()

        data = resp.json()
        if data.get('code') == "0" and data.get('data'):
            item = data['data'][0]
            rate_pct = float(item['fundingRate']) * 100
            ts_ms = int(item['fundingTime'])
            return {
                "funding_rate_percent": round(rate_pct, 4),
                "funding_rate_timestamp": datetime.fromtimestamp(ts_ms / 1000).isoformat(),
                "source": "OKX",
                "note": "Last known funding rate (BTC-USDT-SWAP perpetual)"
            }
        else:
            print(f"[DEBUG] OKX failed: {data.get('msg', 'No data')}")
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] OKX request error: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"[DEBUG] OKX JSON decode error: {str(e)} - Response: {resp.text[:200]}")
    except Exception as e:
        print(f"[DEBUG] OKX unexpected error: {str(e)}")

    # Optional: Bybit fallback
    try:
        bybit_url = "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=1"
        bybit_resp = requests.get(bybit_url, headers=headers, timeout=8)
        bybit_resp.raise_for_status()

        bybit_data = bybit_resp.json()
        if bybit_data.get('retCode') == 0 and bybit_data.get('result', {}).get('list'):
            item = bybit_data['result']['list'][0]
            rate_pct = float(item['fundingRate']) * 100
            ts_ms = int(item['fundingRateTimestamp'])
            return {
                "funding_rate_percent": round(rate_pct, 4),
                "funding_rate_timestamp": datetime.fromtimestamp(ts_ms / 1000).isoformat(),
                "source": "Bybit",
                "note": "Bybit fallback"
            }
    except Exception as e:
        print(f"[DEBUG] Bybit fallback error: {str(e)}")

    return {
        "funding_rate_percent": 0.0,
        "note": "No live funding data available"
    }

# Fear & Greed Index (from alternative.me)
def get_fear_greed() -> Dict:
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        resp = requests.get(url, timeout=5).json()
        if 'data' not in resp or not resp['data']:
            return {"error": "No F&G data"}
        
        item = resp['data'][0]
        return {
            "value": int(item['value']),
            "classification": item['value_classification'],
            "timestamp": item['timestamp'],
            "note": "Alternative.me Crypto Fear & Greed Index"
        }
    except Exception as e:
        print(f"[DEBUG] Fear & Greed error: {str(e)}")
        return {"error": str(e)}

# ====================== CACHE CLEANER ======================
def clean_cache():
    """
    Removes cache entries older than 4 × CACHE_TIMEOUT (e.g., 1 hour if timeout=900).
    Prevents memory growth over long uptime.
    """
    now = time.time()
    to_delete = [
        k for k, v in yf_cache.items()
        if now - v['time'] > CACHE_TIMEOUT * 4
    ]
    for k in to_delete:
        del yf_cache[k]
    if to_delete:
        print(f"[CACHE] Cleaned {len(to_delete)} expired entries")

# ================== AUTH MIDDLEWARE ==================
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ================== MAIN ENDPOINT ==================
@app.get("/full-dashboard")
@limiter.limit("100/minute")
async def full_dashboard(request: Request, x_api_key: str = Header(None)):
    # Clean cache at the start of each request (cheap & safe)
    clean_cache()

    await verify_api_key(x_api_key)  # require key

    # BTC
    btc_price = get_yf_price("BTC-USD")
    wma = get_200wma()
    distance_pct = round((btc_price - float(wma)) / float(wma) * 100, 2) if float(wma) > 0 else 0.0

    #macro
    cpi_yoy = get_yoy_fred("CPIAUCSL")
    unemp, _ = get_latest_fred("UNRATE")
    m1_yoy = get_yoy_fred("M1SL")
    m2_yoy = get_yoy_fred("M2SL")
    ten_y, _ = get_latest_fred("DGS10")

    #commodities
    gold = get_yf_price("GC=F")
    silver = get_yf_price("SI=F")
    oil = get_yf_price("CL=F")

    # Technicals
    rsi = get_rsi()
    macd = get_macd()

    # Sentiment / Derivatives
    funding = get_funding_rate()
    fear_greed = get_fear_greed()

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "btc": {
            "price": btc_price,
            "200wma": wma,
            "distance_to_200wma_percent": distance_pct,
            "max_pain": calculate_max_pain()
        },
        "macro": {
            "cpi_yoy_percent": cpi_yoy,
            "unemployment_rate": unemp,
            "m1_yoy_percent": m1_yoy,
            "m2_yoy_percent": m2_yoy,
            "10y_treasury_yield": ten_y
        },
        "commodities": {
            "gold_price_usd": gold,
            "silver_price_usd": silver,
            "oil_wti_usd_per_barrel": oil,
            "btc_gold_ratio": round(btc_price / gold, 4) if gold > 0 else 0.0
        },
        "technicals": {
            "rsi_14": rsi if rsi is not None else "N/A (fetch failed)",
            "macd": macd if "error" not in macd else {"error": macd.get("error", "Unknown")}
        },
        "sentiment": {
            "funding_rate": funding if "error" not in funding else {"error": funding["error"]},
            "fear_and_greed": fear_greed
        }
    }
    print(f"[DEBUG] Binance response: {resp}")

@app.get("/health")
def health():
    return {"status": "live"}
