# Bitcoin Macro Technicals API

Real-time Bitcoin macro and technical indicators API — one endpoint for everything you need to track BTC in a macro context.

## Features

- **Current BTC price**
- **200WMA** + percentage distance from current price
- **True max pain** (calculated from Deribit options open interest)
- **RSI (14-period)**
- **MACD (12,26,9)** — line, signal, histogram
- **Funding rate** (OKX perpetual futures)
- **Crypto Fear & Greed Index**
- **Macro indicators**: CPI YoY, unemployment rate, M1 YoY, M2 YoY, 10-year Treasury yield
- **Commodities**: gold price, silver price, WTI oil price + BTC/gold ratio

All data fetched from public sources and updated in real-time/near real-time.

## Live API

- **Base URL**: https://btcmacroapi.onrender.com
- **Main endpoint**: GET `/full-dashboard`
- **Authentication**: Header `X-API-Key` (required)
- **Public listing**: https://rapidapi.com/plumtreegroupllc/api/bitcoin-macro-technicals

### Example Request (cURL)

```
```bash
curl -X GET "https://btcmacroapi.onrender.com/full-dashboard" \
  -H "X-API-Key: 21satoshi"
```
###Example Response (shortened)

```
JSON{
  "timestamp": "2026-02-16T19:30:02.842901Z",
  "btc": {
    "price": 68004.19,
    "200wma": 58374.399033203124,
    "distance_to_200wma_percent": 16.5,
    "max_pain": {
      "max_pain_price": 69000,
      "highest_oi_strike": 72000,
      "next_expiry": "2026-02-17"
    }
  },
  "macro": {
    "cpi_yoy_percent": 2.83,
    "unemployment_rate": 4.3,
    "m1_yoy_percent": 4.2,
    "m2_yoy_percent": 4.6,
    "10y_treasury_yield": 4.09
  },
  "commodities": {
    "gold_price_usd": 5012,
    "silver_price_usd": 76.4,
    "oil_wti_usd_per_barrel": 63.8,
    "btc_gold_ratio": 13.5683
  },
  "technicals": {
    "rsi_14": 34.93,
    "macd": {
      "macd_line": -4947.3614,
      "signal_line": -5178.5996,
      "histogram": 231.2381
    }
  },
  "sentiment": {
    "funding_rate": {
      "funding_rate_percent": -0.0055,
      "source": "OKX"
    },
    "fear_and_greed": {
      "value": 12,
      "classification": "Extreme Fear"
    }
  }
}
```
