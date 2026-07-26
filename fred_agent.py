"""Local FRED Agent implementation for querying economic data."""

import asyncio
import json
import os
import requests
from typing import Any, Dict, List
from openai import OpenAI

# FRED API Configuration
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # Get from environment variable
MODEL_CONTEXT_TOKENS = 8_192
MAX_COMPLETION_TOKENS = 3_072
REQUEST_TOKEN_BUDGET = 4_800
TOOL_RESULT_CHAR_LIMIT = 8_000
DEFAULT_OBSERVATION_LIMIT = 12

class FredAPIError(Exception):
    """Custom exception for FRED API errors."""
    pass

class LocalFREDAgent:
    """Local FRED Agent that can query economic data using the FRED API."""
    
    def __init__(self, api_key: str = None, activity_callback=None, chart_callback=None):
        self.api_key = api_key or FRED_API_KEY
        self.activity_callback = activity_callback or (lambda _: None)
        self.chart_callback = chart_callback or (lambda _series_id, _observations: None)
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.tool_map: Dict[str, Any] = {}
        self.openai_tools = []
        
        # Initialize tools
        self._setup_tools()
    
    def _setup_tools(self) -> None:
        """Setup available tools for the agent."""
        # Define FRED API tools
        self.tool_map = {
            "search_fred_series": self.search_fred_series,
            "get_fred_series_data": self.get_fred_series_data,
            "get_fred_series_info": self.get_fred_series_info,
        }
        
        # Create OpenAI-compatible tool definitions
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
            }
        ]
    
    def _make_fred_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the FRED API."""
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        try:
            response = requests.get(f"{FRED_API_BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise FredAPIError(f"Error querying FRED API: {str(e)}")
    
    async def search_fred_series(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for FRED economic series by keyword."""
        self.activity_callback(f"Searching for FRED series: {query}")
        
        try:
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
        start_date: str = None,
        end_date: str = None,
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
        if name not in self.tool_map:
            raise ValueError(f"Unknown tool: {name}")
        
        tool_func = self.tool_map[name]
        return await tool_func(**arguments)
    
    def _token_count(self, value: Any) -> int:
        """Conservatively estimate tokens in serialized OpenAI request data."""
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
        """Keep one leading system prompt and recent complete conversation turns."""
        system_message = next(
            (message for message in messages if message.get("role") == "system"),
            None,
        )
        if system_message is None:
            raise ValueError("A system message is required for the FRED agent request.")

        non_system_messages = [
            message for message in messages if message.get("role") != "system"
        ]
        request_overhead = self._token_count(self.openai_tools) + 1_024
        used_tokens = self._message_token_count(system_message) + request_overhead
        kept_units: List[List[Dict[str, Any]]] = []

        for unit in reversed(self._conversation_units(non_system_messages)):
            unit_tokens = sum(self._message_token_count(message) for message in unit)
            if used_tokens + unit_tokens > max_tokens:
                continue
            kept_units.append(unit)
            used_tokens += unit_tokens

        return [system_message] + [
            message for unit in reversed(kept_units) for message in unit
        ]

    def _serialize_tool_result(self, result: Any) -> str:
        """Serialize a bounded tool result so one API response cannot fill context."""
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
        return self.client.chat.completions.create(
            model="qwen3.6-27b",
            messages=messages,
            tools=self.openai_tools,
            max_tokens=MAX_COMPLETION_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    
    async def run(self, conversation: List[Dict[str, str]]) -> str:
        """Run the agent with a conversation."""
        # Create initial messages with system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful economic data assistant. You can use the "
                    "available FRED API tools to search for and retrieve economic "
                    "data from the Federal Reserve Economic Database (FRED). "
                    "Use a tool whenever it is necessary and explain important results clearly. "
                    "For common US indicators, use UNRATE, CPIAUCSL, PCEPI, and GDPC1 directly "
                    "instead of searching. For a multi-indicator economic report, retrieve the "
                    "four relevant series in one bounded batch with limit 12, then write the "
                    "report from those results. Treat forecasts as uncertain estimates, state "
                    "the data-release lag, and do not make additional searches unless a series "
                    "is unavailable. When a user requests a chart, graph, plot, or visualization, "
                    "retrieve the relevant series observations so the desktop application can "
                    "render the chart. Format final answers with concise Markdown headings, "
                    "bullet points, bold emphasis, and inline code where helpful."
                ),
            },
            *conversation,
        ]
        
        # Truncate messages to prevent context overflow
        messages = self._truncate_messages(messages)
        
        while True:
            try:
                response = self._create_completion(messages)
            except Exception as e:
                # If we get a context overflow error, truncate the conversation and retry
                if "context" in str(e).lower() or "exceeds" in str(e).lower():
                    self.activity_callback("Context window exceeded, truncating conversation...")
                    messages = self._truncate_messages(messages, max_tokens=4_500)
                    response = self._create_completion(messages)
                else:
                    raise e
            
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
            
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })
            
            for call in tool_calls:
                tool_name = call.function.name
                arguments = json.loads(call.function.arguments)
                self.activity_callback(f"Calling FRED tool: {tool_name}")
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
