# Local FRED Agent

A Windows desktop application that uses a model hosted by LM Studio to answer economic-data questions with the Federal Reserve Economic Data (FRED) API.

## Components

- **Desktop UI**: Tkinter chat window with professionally formatted responses and a chart tab for retrieved FRED observations.
- **FRED agent**: searches FRED series and retrieves series metadata and observations.
- **LM Studio**: provides an OpenAI-compatible local API at `http://localhost:1234/v1` using the `qwen3.6-27b` model.

## Prerequisites

1. Python 3.12 or a compatible version with Tk support.
2. [LM Studio](https://lmstudio.ai/) running its local server on port `1234`.
3. The `qwen3.6-27b` model loaded in LM Studio.
4. A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).

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

Enter your FRED API key when prompted. The key is used only for the current application session.

## Use

Ask for an indicator by name or series ID. The agent can search the FRED catalog, retrieve series details, and retrieve recent or date-bounded observations. When it retrieves observations, the application draws a colorful time-series chart in the **Charts** tab. Common US indicators include unemployment (`UNRATE`), consumer prices (`CPIAUCSL`), personal consumption expenditures (`PCEPI`), and real GDP (`GDPC1`).

## Troubleshooting

- If the app cannot connect to the model, start LM Studio's local server and confirm port `1234` is available.
- If the model does not respond, load `qwen3.6-27b` in LM Studio and confirm its OpenAI-compatible API is enabled.
- If FRED requests fail, confirm that the API key is valid and has been entered correctly.
