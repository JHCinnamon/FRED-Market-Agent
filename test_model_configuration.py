"""Tests for the LM Studio Qwen model-loading configuration."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from evals.run_evals import run_case
from fred_agent import LocalFREDAgent, QWEN_MODEL_CONFIG, QWEN_MODEL_NAME


class QwenModelConfigurationTests(unittest.TestCase):
    @patch("fred_agent.LMStudio")
    def test_agent_loads_qwen_with_q8_kv_cache_quantization(self, lm_studio: object) -> None:
        LocalFREDAgent()

        model = lm_studio.return_value.models.get.return_value
        lm_studio.return_value.models.get.assert_called_once_with(QWEN_MODEL_NAME)
        model.load.assert_called_once_with(config=QWEN_MODEL_CONFIG)
        config = model.load.call_args.kwargs["config"]
        self.assertEqual("q8_0", config["llamaKCacheQuantizationType"])
        self.assertEqual("q8_0", config["llamaVCacheQuantizationType"])

    @patch("fred_agent.LMStudio")
    def test_evaluation_initialization_loads_qwen_with_q8_kv_cache_quantization(
        self, lm_studio: object
    ) -> None:
        with patch.object(LocalFREDAgent, "run", new=AsyncMock(return_value="complete")):
            result = asyncio.run(run_case({"id": "model-config", "prompt": "Test prompt."}))

        self.assertEqual("passed", result["status"])
        model = lm_studio.return_value.models.get.return_value
        lm_studio.return_value.models.get.assert_called_once_with(QWEN_MODEL_NAME)
        model.load.assert_called_once_with(config=QWEN_MODEL_CONFIG)


if __name__ == "__main__":
    unittest.main()