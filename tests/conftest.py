("""Pytest configuration for test collection.

By default we skip integration tests that require external services (LLM providers,
local Ollama server, networked MCP servers). To run them set the environment
variable RUN_INTEGRATION=1.
""")
import os
import pytest
from pathlib import Path


RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "") == "1"


def pytest_collection_modifyitems(config, items):
	if RUN_INTEGRATION:
		return

	skip_reason = "Integration tests skipped (set RUN_INTEGRATION=1 to run)"
	for item in items:
		p = Path(item.fspath)
		# Skip the gemini integration tests and known external-provider tests
		if "tests/gemini" in str(p) or p.name in {
			"test_providers.py",
			"test_ollama_model_tool_support.py",
		}:
			item.add_marker(pytest.mark.skip(reason=skip_reason))

