"""Tests for the LM Studio Qwen model-loading configuration."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from evals.run_evals import run_case
from fred_agent import LocalFREDAgent, QWEN_MODEL_CONFIG, QWEN_MODEL_NAME


class QwenModelConfigurationTests(unittest.TestCase):
    @patch("fred_agent.lms.Client")
    def test_agent_loads_qwen_without_kv_cache_quantization(self, client: object) -> None:
        LocalFREDAgent()

        client.return_value.llm.model.assert_called_once_with(
            QWEN_MODEL_NAME, config=QWEN_MODEL_CONFIG
        )
        config = client.return_value.llm.model.call_args.kwargs["config"]
        self.assertNotIn("llamaKCacheQuantizationType", config)
        self.assertNotIn("llamaVCacheQuantizationType", config)

    @patch("fred_agent.lms.Client")
    def test_evaluation_initialization_loads_qwen_without_kv_cache_quantization(
        self, client: object
    ) -> None:
        with patch.object(LocalFREDAgent, "run", new=AsyncMock(return_value="complete")):
            result = asyncio.run(run_case({"id": "model-config", "prompt": "Test prompt."}))

        self.assertEqual("passed", result["status"])
        client.return_value.llm.model.assert_called_once_with(
            QWEN_MODEL_NAME, config=QWEN_MODEL_CONFIG
        )


if __name__ == "__main__":
    unittest.main()