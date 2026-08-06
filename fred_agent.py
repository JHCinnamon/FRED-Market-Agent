"""Local FRED Agent implementation for querying economic data."""

import asyncio
import json
import os
import re
import requests
from typing import Any, Dict, List, Optional, cast
from openai import OpenAI

# API endpoints, model limits, and budgets shared by every agent run.
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # Get from environment variable
TWELVE_DATA_API_BASE_URL = "https://api.twelvedata.com"
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
MODEL_CONTEXT_TOKENS = 8_192
MAX_COMPLETION_TOKENS = 3_072
REQUEST_TOKEN_BUDGET = 4_800
HISTORY_TOKEN_BUDGET = 900
TOOL_RESULT_CHAR_LIMIT = 8_000
DEFAULT_OBSERVATION_LIMIT = 12
MAX_TOOL_CALL_ROUNDS = 8
MARKET_SYMBOL_PATTERN = re.compile(
    r"\b(?:S&P\s*500|NASDAQ(?:\s+COMPOSITE)?|DOW(?:\s+JONES)?|CSI\s*300)\b"
    r"|\$[A-Za-z]{1,5}\b|\b[A-Za-z]{3}/[A-Za-z]{3}\b",
    re.IGNORECASE,
)
# Avoid turning a broad market question into an unbounded sequence of symbol searches.
MAX_PREFLIGHT_SYMBOL_SEARCHES = 2
QWEN_MODEL_NAME = "google/gemma-4-12b-qat"  ##"qwen/qwen3.6-27b" ##(too large for 32 GB SYSRAM)

class FredAPIError(Exception):
    """Custom exception for FRED API errors."""
    pass

class TwelveDataAPIError(Exception):
    """Custom exception for Twelve Data API errors."""
    pass

class LocalFREDAgent:
    """Local FRED Agent that can query economic data using the FRED API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        twelve_data_api_key: Optional[str] = None,
        activity_callback=None,
        chart_callback=None,
        token_callback=None,
    ):
        self.api_key = api_key or FRED_API_KEY
        self.twelve_data_api_key = twelve_data_api_key or TWELVE_DATA_API_KEY
        self.activity_callback = activity_callback or (lambda _: None)
        self.chart_callback = chart_callback or (lambda _series_id, _observations: None)
        self.token_callback = token_callback or (lambda _usage: None)
        # LM Studio exposes the locally loaded Qwen model through an OpenAI-compatible endpoint.
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.tool_map: Dict[str, Any] = {}
        self.openai_tools = []
        
        # Initialize tools
        self._setup_tools()
    
    def _setup_tools(self) -> None:
        """Setup available tools for the agent."""
        # Map model-visible function names to the concrete async implementations.
        self.tool_map = {
            "search_fred_series": self.search_fred_series,
            "get_fred_series_data": self.get_fred_series_data,
            "get_fred_series_info": self.get_fred_series_info,
            "search_market_symbols": self.search_market_symbols,
            "get_market_quote": self.get_market_quote,
            "get_market_time_series": self.get_market_time_series,
        }
        
        # Send JSON schemas to LM Studio so Qwen can issue structured tool calls.
        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_fred_series",
                    "description": "Search for FRED economic series by keyword",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for economic series"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 10)"
                            }
                        },
                        "required": ["query"]
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_fred_series_data",
                    "description": "Get time series data for a FRED economic series",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "series_id": {
                                "type": "string",
                                "description": "The FRED series ID"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format (optional)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date in YYYY-MM-DD format (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum recent observations to return (1-1000, default: 12)"
                            }
                        },
                        "required": ["series_id"]
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_fred_series_info",
                    "description": "Get detailed information about a FRED economic series",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "series_id": {
                                "type": "string",
                                "description": "The FRED series ID"
                            }
                        },
                        "required": ["series_id"]
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_market_symbols",
                    "description": "Search Twelve Data symbols for stocks, ETFs, mutual funds, indices, forex, or crypto.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Company name, fund name, or market symbol to search for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of matches to return, from 1 to 10"
                            }
                        },
                        "required": ["query"]
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_quote",
                    "description": "Get the latest available Twelve Data market quote for a symbol such as AAPL, SPY, EUR/USD, or BTC/USD.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Twelve Data symbol, such as AAPL, SPY, EUR/USD, or BTC/USD"
                            }
                        },
                        "required": ["symbol"]
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_time_series",
                    "description": "Get historical daily prices from Twelve Data for a stock, ETF, mutual fund, index, forex pair, or crypto asset. Use this when a chart, trend, or comparison is needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Twelve Data symbol, such as AAPL, SPY, EUR/USD, or BTC/USD"
                            },
                            "interval": {
                                "type": "string",
                                "description": "Time interval; use 1day unless the request needs another interval"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum observations to return, from 2 to 60"
                            }
                        },
                        "required": ["symbol"]
                    },
                }
            }
        ]

    def _make_fred_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the FRED API."""
        # FRED expects its credentials and JSON response type on every request.
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        try:
            response = requests.get(f"{FRED_API_BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise FredAPIError(f"Error querying FRED API: {str(e)}")

    def _make_twelve_data_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an authenticated request to Twelve Data and surface API errors clearly."""
        # Fail before a network call when the optional market-data credential is unavailable.
        if not self.twelve_data_api_key:
            raise TwelveDataAPIError(
                "A Twelve Data API key is required. Set TWELVE_DATA_API_KEY or enter it at startup."
            )
        params["apikey"] = self.twelve_data_api_key
        try:
            response = requests.get(
                f"{TWELVE_DATA_API_BASE_URL}/{endpoint}",
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as error:
            raise TwelveDataAPIError(f"Error querying Twelve Data API: {error}") from error
        if data.get("status") == "error" or data.get("code"):
            raise TwelveDataAPIError(data.get("message", "Twelve Data returned an API error."))
        return data

    async def search_market_symbols(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Twelve Data's market symbol catalog."""
        # Return only fields useful for choosing an unambiguous market instrument.
        self.activity_callback(f"Searching market symbols: {query}")
        data = self._make_twelve_data_request(
            "symbol_search", {"symbol": query, "outputsize": max(1, min(int(limit), 10))}
        )
        return [
            {
                field: item.get(field)
                for field in ("symbol", "instrument_name", "exchange", "mic_code", "type", "currency")
                if item.get(field) is not None
            }
            for item in data.get("data", [])
        ]

    def _market_symbol_queries(self, question: str) -> List[str]:
        """Extract explicit market identifiers suitable for a symbol lookup."""
        # Deduplicate case-insensitively because a prompt may repeat the same identifier.
        queries: List[str] = []
        for match in MARKET_SYMBOL_PATTERN.finditer(question):
            query = match.group(0).lstrip("$")
            if query.casefold() not in {item.casefold() for item in queries}:
                queries.append(query)
        return queries[:MAX_PREFLIGHT_SYMBOL_SEARCHES]

    async def _prefetch_market_symbols(self, question: str) -> Dict[str, List[Dict[str, Any]]]:
        """Resolve explicit market identifiers before the model plans its tool calls."""
        # A failed preflight must not prevent the model from using its normal tools later.
        matches: Dict[str, List[Dict[str, Any]]] = {}
        for query in self._market_symbol_queries(question):
            try:
                results = await self.search_market_symbols(query, limit=3)
            except TwelveDataAPIError:
                continue
            if results:
                matches[query] = results
        return matches

    async def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        """Get the latest market quote from Twelve Data."""
        self.activity_callback(f"Getting market quote: {symbol}")
        data = self._make_twelve_data_request("quote", {"symbol": symbol.upper()})
        fields = (
            "symbol", "name", "exchange", "currency", "datetime", "close", "previous_close",
            "change", "percent_change", "open", "high", "low", "volume", "is_market_open",
        )
        return {field: data.get(field) for field in fields if data.get(field) is not None}

    async def get_market_time_series(
        self,
        symbol: str,
        interval: str = "1day",
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """Get normalized market observations and send them to the chart callback."""
        self.activity_callback(f"Getting market time series: {symbol}")
        data = self._make_twelve_data_request(
            "time_series",
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "outputsize": max(2, min(int(limit), 60)),
            },
        )
        # Normalize Twelve Data's timestamped closes into the same chart callback shape as FRED.
        observations = [
            {"date": item.get("datetime", "")[:10], "value": item.get("close")}
            for item in data.get("values", [])
        ]
        chart_id = f"Market: {data.get('meta', {}).get('symbol', symbol.upper())}"
        self.chart_callback(chart_id, observations)
        return observations
    
    async def search_fred_series(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for FRED economic series by keyword."""
        # Restrict search output to metadata the model can use to select a series.
        self.activity_callback(f"Searching for FRED series: {query}")
        
        try:
            # Descending order makes bounded requests return the newest observations first.
            params = {
                "search_text": query,
                "limit": max(1, min(int(limit), 5)),
            }
            data = self._make_fred_request("series/search", params)
            
            if 'seriess' in data:
                fields = (
                    "id",
                    "title",
                    "frequency",
                    "units",
                    "seasonal_adjustment",
                    "observation_start",
                    "observation_end",
                )
                return [
                    {field: series.get(field) for field in fields if field in series}
                    for series in data['seriess']
                ]
            else:
                return []
        except Exception as e:
            raise FredAPIError(f"Failed to search FRED series: {str(e)}")
    
    async def get_fred_series_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = DEFAULT_OBSERVATION_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Get time series data for a FRED economic series."""
        self.activity_callback(f"Getting data for FRED series: {series_id}")
        
        try:
            params = {
                "series_id": series_id,
                "limit": max(1, min(int(limit), 1000)),
                "sort_order": "desc",
            }
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            
            data = self._make_fred_request("series/observations", params)
            
            # Extract the observations from the response
            if 'observations' in data:
                observations = data['observations']
                self.chart_callback(series_id, observations)
                return observations
            else:
                return []
        except Exception as e:
            raise FredAPIError(f"Failed to get FRED series data: {str(e)}")
    
    async def get_fred_series_info(self, series_id: str) -> Dict[str, Any]:
        """Get detailed information about a FRED economic series."""
        self.activity_callback(f"Getting info for FRED series: {series_id}")
        
        try:
            params = {
                "series_id": series_id
            }
            data = self._make_fred_request("series", params)
            
            # Extract the series information from the response
            if 'seriess' in data and len(data['seriess']) > 0:
                return data['seriess'][0]
            else:
                return {}
        except Exception as e:
            raise FredAPIError(f"Failed to get FRED series info: {str(e)}")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name with given arguments."""
        # Reject invented tool names rather than exposing arbitrary methods to the model.
        if name not in self.tool_map:
            raise ValueError(f"Unknown tool: {name}")
        
        tool_func = self.tool_map[name]
        return await tool_func(**arguments)
    
    def _token_count(self, value: Any) -> int:
        """Conservatively estimate tokens in serialized OpenAI request data."""
        # Usage metadata is not always returned by local servers, so use a stable character estimate.
        if not isinstance(value, str):
            value = json.dumps(value, default=str, ensure_ascii=True, separators=(",", ":"))
        return (len(value) + 2) // 3

    def _message_token_count(self, message: Dict[str, Any]) -> int:
        return self._token_count(message) + 8

    def _conversation_units(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group tool calls with their required tool responses."""
        units: List[List[Dict[str, Any]]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            unit = [message]
            index += 1
            if message.get("role") == "assistant" and message.get("tool_calls"):
                while index < len(messages) and messages[index].get("role") == "tool":
                    unit.append(messages[index])
                    index += 1
            units.append(unit)
        return units

    def _truncate_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = REQUEST_TOKEN_BUDGET,
    ) -> List[Dict[str, Any]]:
        """Keep a system prompt, the latest user request, and recent complete turns."""
        system_message = next(
            (message for message in messages if message.get("role") == "system"),
            None,
        )
        if system_message is None:
            raise ValueError("A system message is required for the FRED agent request.")

        non_system_messages = [
            message for message in messages if message.get("role") != "system"
        ]
        # Reserve space for tool schemas and a completion so requests fit the loaded context.
        request_overhead = self._token_count(self.openai_tools) + 1_024
        base_tokens = self._message_token_count(system_message) + request_overhead
        latest_user_index = max(
            (
                index
                for index, message in enumerate(non_system_messages)
                if message.get("role") == "user"
            ),
            default=-1,
        )
        if latest_user_index < 0:
            raise ValueError("A user message is required for the FRED agent request.")

        latest_user = non_system_messages[latest_user_index]
        trailing_exchange = non_system_messages[latest_user_index + 1 :]
        anchor = [latest_user, *trailing_exchange]
        anchor_tokens = sum(self._message_token_count(message) for message in anchor)

        # A tool-call exchange is only valid with its initiating user request. If the
        # exchange cannot fit, keep the request and let the model issue fresh tool calls.
        if base_tokens + anchor_tokens > max_tokens:
            anchor = [latest_user]
            anchor_tokens = self._message_token_count(latest_user)

        used_tokens = base_tokens + anchor_tokens
        previous_messages = non_system_messages[:latest_user_index]
        prior_turns: List[List[Dict[str, Any]]] = []
        current_turn: List[Dict[str, Any]] = []
        for message in previous_messages:
            if message.get("role") == "user" and current_turn:
                prior_turns.append(current_turn)
                current_turn = []
            current_turn.append(message)
        if current_turn:
            prior_turns.append(current_turn)

        # Retain only a small recent-history window; long reports should not slow future requests.
        kept_turns: List[List[Dict[str, Any]]] = []
        history_tokens = 0
        for turn in reversed(prior_turns):
            turn_tokens = sum(self._message_token_count(message) for message in turn)
            if (
                history_tokens + turn_tokens > HISTORY_TOKEN_BUDGET
                or used_tokens + turn_tokens > max_tokens
            ):
                continue
            kept_turns.append(turn)
            used_tokens += turn_tokens
            history_tokens += turn_tokens

        return [system_message] + [
            message for turn in reversed(kept_turns) for message in turn
        ] + anchor

    def _serialize_tool_result(self, result: Any) -> str:
        """Serialize a bounded tool result so one API response cannot fill context."""
        # Tool payloads become messages in the next completion, so cap their serialized size.
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        content = json.dumps(result, default=str, ensure_ascii=True, separators=(",", ":"))
        if len(content) <= TOOL_RESULT_CHAR_LIMIT:
            return content
        return json.dumps(
            {
                "truncated": True,
                "message": "Tool result exceeded the response budget.",
                "preview": content[:TOOL_RESULT_CHAR_LIMIT],
            },
            separators=(",", ":"),
        )

    def _create_completion(self, messages: List[Dict[str, Any]]):
        """Create a concise response compatible with LM Studio's Qwen template."""
        # Disable Qwen's hidden reasoning channel so the UI receives concise visible answers.
        return self.client.chat.completions.create(
            model=QWEN_MODEL_NAME,
            messages=cast(Any, messages),
            tools=cast(Any, self.openai_tools),
            max_tokens=MAX_COMPLETION_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

    def _create_final_completion(self, messages: List[Dict[str, Any]]):
        """Request a final response without offering additional tools."""
        return self.client.chat.completions.create(
            model=QWEN_MODEL_NAME,
            messages=cast(Any, messages),
            max_tokens=MAX_COMPLETION_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

    def _report_token_usage(self, response: Any, messages: List[Dict[str, Any]]) -> None:
        """Report server usage when available, otherwise a request-size estimate."""
        # Fall back to local estimates when LM Studio does not include usage in its response.
        estimated_prompt_tokens = (
            sum(self._message_token_count(message) for message in messages)
            + self._token_count(self.openai_tools)
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        has_reported_usage = any(
            value is not None for value in (prompt_tokens, completion_tokens, total_tokens)
        )
        prompt_tokens = prompt_tokens if prompt_tokens is not None else estimated_prompt_tokens
        completion_tokens = completion_tokens if completion_tokens is not None else 0
        total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        self.token_callback(
            {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "estimated": not has_reported_usage,
            }
        )

    async def run(self, conversation: List[Dict[str, str]]) -> str:
        """Run the agent with a conversation."""
        # Resolve explicit symbols before planning so the model sees candidate market identifiers.
        latest_question = next(
            (message["content"] for message in reversed(conversation) if message["role"] == "user"),
            "",
        )
        market_symbol_matches = await self._prefetch_market_symbols(latest_question)
        preflight_context = ""
        if market_symbol_matches:
            # Add matches to the initial instruction without changing the user conversation.
            preflight_context = (
                " Twelve Data preflight market-symbol matches for this request: "
                f"{json.dumps(market_symbol_matches, ensure_ascii=True)}. "
                "Use these matches when planning market-data retrieval."
            )
        # Build a single leading system message because Qwen's chat template requires that placement.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful economic data assistant. You can use the "
                    "available FRED and Twelve Data tools to retrieve economic and market data. "
                    "Use FRED for macroeconomic indicators, economic releases, and historical "
                    "US economic series. Use Twelve Data for stocks, ETFs, mutual funds, indices, "
                    "forex, crypto, quotes, and market-price history. For an economic question that "
                    "would be clearer with market context, you may use both sources, but do not "
                    "fetch market data merely for decoration. Treat company, stock, and stocks as "
                    "Twelve Data requests, even when the question also includes macroeconomic context. "
                    "Use a tool whenever it is necessary "
                    "and identify the source and observation date for material claims. "
                    "For common US indicators, use UNRATE, CPIAUCSL, PCEPI, and GDPC1 directly "
                    "instead of searching. For a multi-indicator economic report, retrieve the "
                    "four relevant series in one bounded batch with limit 12, then write the "
                    "report from those results. Treat forecasts as uncertain estimates, state "
                    "the data-release lag, and do not make additional searches unless a series "
                    "is unavailable. When a user requests a chart, graph, plot, or visualization, "
                    "retrieve the relevant series observations so the desktop application can "
                    "render the chart. For multi-indicator reports, give a complete narrative "
                    "with these Markdown sections: Executive summary, Latest readings, What the "
                    "data suggests, Outlook, and Caveats. Explain the direction and significance "
                    "of every retrieved series, distinguish level changes from inflation rates, "
                    "and turn every reported figure or chart into a natural-language explanation "
                    "of what it means for the user's question; never return only headings, raw "
                    "tool output, or a chart label. Forecasts must include a "
                    "range, assumptions, uncertainty, and the data-release lag. Format final "
                    "answers with concise Markdown headings, bullet points, bold emphasis, and "
                    f"inline code where helpful.{preflight_context}"
                ),
            },
            *conversation,
        ]
        
        # Enforce the request budget before the first local-model completion.
        messages = self._truncate_messages(messages)
        self.activity_callback("Preparing the model request")
        
        for tool_call_round in range(MAX_TOOL_CALL_ROUNDS):
            # Each iteration is either a final answer or one structured tool-call round.
            try:
                response = self._create_completion(messages)
            except Exception as e:
                # If we get a context overflow error, truncate the conversation and retry
                if "context" in str(e).lower() or "exceeds" in str(e).lower():
                    self.activity_callback("Context window exceeded, truncating conversation")
                    messages = self._truncate_messages(messages, max_tokens=4_500)
                    response = self._create_completion(messages)
                else:
                    raise e

            self._report_token_usage(response, messages)
            
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls
            if not tool_calls:
                if assistant_message.content:
                    return assistant_message.content
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    return (
                        "The local model exhausted its response budget before producing an "
                        "answer. Reduce LM Studio's context usage or increase its loaded "
                        "context window, then submit the request again."
                    )
                return "The local model returned no visible answer. Please submit the request again."

            self.activity_callback("Reviewing tool requests")
            
            # Preserve the assistant tool-call envelope so following tool results retain their call IDs.
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": cast(Any, tc).function.name,
                            "arguments": cast(Any, tc).function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })
            
            for call in tool_calls:
                # Return tool errors to Qwen as data, allowing it to recover in its final response.
                tool_name = cast(Any, call).function.name
                arguments = json.loads(cast(Any, call).function.arguments)
                self.activity_callback(f"Retrieving data with {tool_name}")
                try:
                    tool_result = await self.call_tool(tool_name, arguments)
                    content = self._serialize_tool_result(tool_result)
                except Exception as error:
                    content = json.dumps({"error": str(error)})
                
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content,
                    }
                )
                
            messages = self._truncate_messages(messages)
            self.activity_callback("Synthesizing the final response")

        self.activity_callback("Tool-call limit reached, writing final response")
        response = self._create_final_completion(messages)
        answer = response.choices[0].message.content
        if answer:
            return answer
        return "The local model did not produce a final response. Please submit the request again."
