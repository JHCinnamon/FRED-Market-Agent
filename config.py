"""Configuration management for the FRED agent."""

import os
from typing import Optional

# FRED API Configuration
FRED_API_KEY: Optional[str] = os.getenv("FRED_API_KEY", "")
FRED_API_BASE_URL: str = "https://api.stlouisfed.org/fred"

# LM Studio Configuration
LM_STUDIO_MODEL: str = "qwen3.6-27b"
LM_STUDIO_HOST: str = "http://localhost:1234"

# Application Configuration
APP_NAME: str = "FRED Economic Data Agent"
APP_VERSION: str = "1.0.0"