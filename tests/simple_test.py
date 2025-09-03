
import asyncio
import pytest
from backend.agents.financial_agent import get_financial_agent, FinancialAgent

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="module")
async def agent() -> FinancialAgent:
    """Fixture to get a fully initialized financial agent."""
    return await get_financial_agent()

async def run_tool_test(agent: FinancialAgent, query: str):
    """Helper function to run a tool test and print results."""
    print(f"--- Query ---")
    print(query)
    
    response = await agent.process_message(query)
    
    print(f"--- Response ---")
    # The response can be an async generator, so we need to handle it
    response_text = ''
    try:
        if hasattr(response, '__aiter__') and not isinstance(response, str):
            async for part in response:
                response_text += str(part)
        else:
            response_text = str(response)
    except Exception:
        response_text = str(response)
        
    print(response_text)
    print("\n" + "="*80 + "\n")
    
    # Basic assertion to ensure the agent ran
    assert response_text is not None
    assert "error" not in response_text.lower()

async def test_tool_get_database_schema(agent: FinancialAgent):
    await run_tool_test(agent, "What is the database schema?")
