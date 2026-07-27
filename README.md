# Local FRED Agent

A Windows desktop application that uses a model hosted by LM Studio to answer economic and market-data questions with FRED and Twelve Data.

## Components

- **Desktop UI**: MLflow-inspired run feed with formatted responses and inline chart artifacts for retrieved FRED observations.
- **Request telemetry**: compact session and prompt token totals, plus an animated working state with live agent activity, estimated progress, live token estimate, and token throughput.
- **FRED tools**: search economic series and retrieve series metadata and observations.
- **Twelve Data tools**: search market symbols, retrieve latest quotes, and retrieve price history for stocks, ETFs, mutual funds, indices, forex, and crypto.
- **LM Studio**: provides an OpenAI-compatible local API at `http://localhost:1234/v1` using the `qwen3.6-27b` model.

## Prerequisites

1. Python 3.12 or a compatible version with Tk support.
2. [LM Studio](https://lmstudio.ai/) running its local server on port `1234`.
3. The `qwen3.6-27b` model loaded in LM Studio.
4. A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).
5. An optional [Twelve Data API key](https://twelvedata.com/account/api-keys) for market-data requests.

## Setup

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

In LM Studio, load `qwen3.6-27b` and start its local server. The server must be available at `http://localhost:1234/v1`.

## Run

From this directory, start the application:

```powershell
python app.py
```

Enter your FRED API key when prompted. Enter the Twelve Data key when prompted, or set `TWELVE_DATA_API_KEY` in your environment before starting the app. Keys are used only by the local application and are not included in model prompts or chat history.

## Use

Ask for an indicator by name or series ID. The agent uses FRED for macroeconomic series, and Twelve Data for market questions such as `AAPL`, `SPY`, `EUR/USD`, and `BTC/USD`. For a macroeconomic analysis where market pricing provides useful context, it can use both sources and identify the source and observation date in its report.

When it retrieves observations, the application draws a colorful time-series chart directly below the associated response and adds a data synopsis with the latest readings and changes. Series render in separate panels, so different units and scales remain readable. Common FRED indicators include unemployment (`UNRATE`), consumer prices (`CPIAUCSL`), personal consumption expenditures (`PCEPI`), and real GDP (`GDPC1`).

While the model is working, the small status strip shows the active agent phase in yellow beside animated dots, plus estimated completion, a live token estimate, and tokens per second. The header's prompt total refreshes on a short sampled cadence during active work, while the session total changes only when LM Studio returns completion usage metadata. When usage metadata is unavailable, the application uses its request-size estimate.

## Troubleshooting

- If the app cannot connect to the model, start LM Studio's local server and confirm port `1234` is available.
- If the model does not respond, load `qwen3.6-27b` in LM Studio and confirm its OpenAI-compatible API is enabled.
- If FRED requests fail, confirm that the API key is valid and has been entered correctly.
- If a market-data request reports a missing Twelve Data key, enter it at startup or set `TWELVE_DATA_API_KEY` before launching the app.
