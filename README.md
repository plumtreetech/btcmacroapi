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

```bash
curl -X GET "https://btcmacroapi.onrender.com/full-dashboard" \
  -H "X-API-Key: YOUR_MASTER_KEY_HERE"