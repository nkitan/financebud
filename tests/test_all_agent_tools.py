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

async def test_tool_get_account_summary(agent: FinancialAgent):
    await run_tool_test(agent, "Give me a summary of my account.")

async def test_tool_get_recent_transactions(agent: FinancialAgent):
    await run_tool_test(agent, "Show me my last 5 transactions.")

async def test_tool_search_transactions(agent: FinancialAgent):
    await run_tool_test(agent, "Search for transactions with 'UPI'.")

async def test_tool_get_transactions_by_date_range(agent: FinancialAgent):
    query = """
    Get all transactions from January 1, 2024 to January 31, 2024.
    """
    await run_tool_test(agent, query)

async def test_tool_get_monthly_summary(agent: FinancialAgent):
    await run_tool_test(agent, "Get the monthly summary for year 2024 and month 1.")

async def test_tool_get_spending_by_category(agent: FinancialAgent):
    await run_tool_test(agent, "Get spending by category for the last 30 days.")

async def test_tool_find_recurring_payments(agent: FinancialAgent):
    await run_tool_test(agent, "Find my recurring payments.")

async def test_tool_analyze_spending_trends(agent: FinancialAgent):
    await run_tool_test(agent, "Analyze my spending trends for the last 6 months.")

async def test_tool_execute_custom_query(agent: FinancialAgent):
    query = """
    Execute the following SQL query:
    SELECT transaction_date, description, debit_amount 
    FROM transactions 
    WHERE debit_amount > 1000 
    ORDER BY debit_amount DESC 
    LIMIT 5;
    """
    await run_tool_test(agent, query)

async def test_tool_get_database_schema(agent: FinancialAgent):
    await run_tool_test(agent, "What is the database schema?")