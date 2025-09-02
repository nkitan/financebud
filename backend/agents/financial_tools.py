"""
Financial Tools
===============

Production-ready financial analysis tools with real data integration,
advanced caching, error handling, and comprehensive monitoring.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from backend.mcp.client import get_mcp_manager
from backend.database.db import get_db_manager
from backend.logging_config import get_logger_with_context

logger = get_logger_with_context(__name__)


async def get_account_summary_tool(tool_input: str) -> str:
    """Get comprehensive account summary with error handling."""
    try:
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_account_summary", {})
        
        # Validate and enhance the result
        if isinstance(result, dict) and "data" in result:
            data = result["data"]
            
            # Add derived insights
            result = {
                **result,
                "insights": {
                    "account_health": _calculate_account_health(data),
                    "cash_flow_status": _analyze_cash_flow(data),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Account summary tool error: {e}")
        return json.dumps({
            "error": f"Failed to get account summary: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def get_recent_transactions_tool(tool_input: str) -> str:
    """Get recent transactions with intelligent parsing and validation."""
    limit = 10  # Initialize limit with default value
    try:
        # input parsing
        try:
            if tool_input.strip():
                parsed_input = json.loads(tool_input)
                limit = parsed_input.get("limit", 10)
        except json.JSONDecodeError:
            # Try to parse as integer
            try:
                limit = int(tool_input.strip())
            except (ValueError, AttributeError):
                limit = 10
        
        # Validate and constrain limit
        limit = max(1, min(limit, 100))
        
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr", 
            "get_recent_transactions", 
            {"limit": limit}
        )
        
        # Enhance with transaction insights
        if isinstance(result, dict) and "data" in result:
            transactions = result["data"].get("transactions", [])
            
            result = {
                **result,
                "insights": {
                    "transaction_patterns": _analyze_transaction_patterns(transactions),
                    "spending_categories": _categorize_recent_transactions(transactions),
                    "unusual_activity": _detect_unusual_activity(transactions)
                }
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Recent transactions tool error: {e}")
        return json.dumps({
            "error": f"Failed to get recent transactions: {str(e)}",
            "limit_attempted": limit,
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def search_transactions_tool(tool_input: str) -> str:
    """Search transactions with intelligent pattern matching."""
    pattern = ""
    limit = 20
    
    try:
        try:
            parsed_input = json.loads(tool_input)
            pattern = parsed_input.get("pattern", "")
            limit = parsed_input.get("limit", 20)
        except json.JSONDecodeError:
            pattern = tool_input.strip()
        
        if not pattern:
            return json.dumps({
                "error": "Search pattern is required",
                "example": "Use a keyword like 'UPI', 'grocery', or merchant name",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
        
        # Validate limit
        limit = max(1, min(limit, 100))
        
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "search_transactions",
            {"pattern": pattern, "limit": limit}
        )
        
        # Enhance with search insights
        if isinstance(result, dict) and "data" in result:
            transactions = result["data"].get("transactions", [])
            
            result = {
                **result,
                "search_insights": {
                    "pattern_effectiveness": len(transactions) / limit if limit > 0 else 0,
                    "amount_distribution": _analyze_amount_distribution(transactions),
                    "time_distribution": _analyze_time_distribution(transactions),
                    "suggested_filters": _suggest_search_refinements(pattern, transactions)
                }
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Search transactions tool error: {e}")
        return json.dumps({
            "error": f"Failed to search transactions: {str(e)}",
            "pattern_attempted": pattern,
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def get_transactions_by_date_range_tool(tool_input: str) -> str:
    """Get transactions by date range with validation and insights."""
    try:
        parsed_input = json.loads(tool_input)
        start_date = parsed_input.get("start_date")
        end_date = parsed_input.get("end_date")
        
        if not start_date or not end_date:
            return json.dumps({
                "error": "Both start_date and end_date are required",
                "format": "YYYY-MM-DD",
                "example": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31"
                },
                "timestamp": datetime.now().isoformat()
            }, indent=2)
        
        # Validate date format
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_dt > end_dt:
                return json.dumps({
                    "error": "start_date must be before end_date",
                    "provided": {"start_date": start_date, "end_date": end_date},
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
                
        except ValueError as e:
            return json.dumps({
                "error": f"Invalid date format: {str(e)}",
                "required_format": "YYYY-MM-DD",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
        
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "get_transactions_by_date_range",
            {"start_date": start_date, "end_date": end_date}
        )
        
        # Enhance with period analysis
        if isinstance(result, dict) and "data" in result:
            transactions = result["data"].get("transactions", [])
            summary = result["data"].get("summary", {})
            
            result = {
                **result,
                "period_analysis": {
                    "daily_averages": _calculate_daily_averages(transactions, start_dt, end_dt),
                    "weekday_patterns": _analyze_weekday_patterns(transactions),
                    "spending_velocity": _calculate_spending_velocity(transactions),
                    "period_comparison": _compare_with_previous_period(
                        start_dt, end_dt, summary
                    )
                }
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Date range transactions tool error: {e}")
        return json.dumps({
            "error": f"Failed to get transactions by date range: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def get_monthly_summary_tool(tool_input: str) -> str:
    """Get monthly summary with analytics."""
    try:
        year = None
        month = None
        
        if tool_input.strip():
            try:
                parsed_input = json.loads(tool_input)
                year = parsed_input.get("year")
                month = parsed_input.get("month")
            except json.JSONDecodeError:
                pass
        
        arguments = {}
        if year:
            arguments["year"] = year
        if month:
            arguments["month"] = month
        
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "get_monthly_summary",
            arguments
        )
        
        # Enhance with monthly insights
        if isinstance(result, dict) and "data" in result:
            summary = result["data"].get("summary", {})
            daily_breakdown = result["data"].get("daily_breakdown", [])
            
            result = {
                **result,
                "monthly_insights": {
                    "spending_consistency": _analyze_spending_consistency(daily_breakdown),
                    "peak_spending_days": _find_peak_spending_days(daily_breakdown),
                    "budget_recommendations": _generate_budget_recommendations(summary),
                    "month_over_month": _compare_month_over_month(year, month, summary)
                }
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Monthly summary tool error: {e}")
        return json.dumps({
            "error": f"Failed to get monthly summary: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def get_spending_by_category_tool(tool_input: str) -> str:
    """Analyze spending by category with advanced insights."""
    days = 30  # Initialize with default value
    
    try:
        if tool_input.strip():
            try:
                parsed_input = json.loads(tool_input)
                days = parsed_input.get("days", 30)
            except json.JSONDecodeError:
                try:
                    days = int(tool_input.strip())
                except (ValueError, AttributeError):
                    days = 30
        
        days = max(1, min(days, 365))
        
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "get_spending_by_category",
            {"days": days}
        )
        
        # Enhance with category insights
        if isinstance(result, dict) and "data" in result:
            categories = result["data"].get("categories", [])
            
            result = {
                **result,
                "category_insights": {
                    "top_categories": _identify_top_spending_categories(categories),
                    "optimization_opportunities": _identify_optimization_opportunities(categories),
                    "spending_distribution": _analyze_spending_distribution(categories),
                    "category_trends": _analyze_category_trends(categories, days)
                }
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Spending by category tool error: {e}")
        return json.dumps({
            "error": f"Failed to analyze spending by category: {str(e)}",
            "days_attempted": days,
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def find_recurring_payments_tool(tool_input: str) -> str:
    """Find recurring payments with pattern detection."""
    try:
        min_occurrences = 3
        days_back = 90
        top_n = 20

        if tool_input.strip():
            try:
                parsed_input = json.loads(tool_input)
                min_occurrences = parsed_input.get("min_occurrences", 3)
                days_back = parsed_input.get("days_back", 90)
                top_n = parsed_input.get("top_n", 20)
            except json.JSONDecodeError:
                # Support simple space-separated fallback: "min_occurrences days_back top_n"
                try:
                    parts = tool_input.strip().split()
                    if len(parts) >= 1:
                        min_occurrences = int(parts[0])
                    if len(parts) >= 2:
                        days_back = int(parts[1])
                    if len(parts) >= 3:
                        top_n = int(parts[2])
                except Exception:
                    pass

        # Constrain inputs
        min_occurrences = max(2, min(min_occurrences, 10))
        days_back = max(7, min(days_back, 365))
        top_n = max(1, min(top_n, 100))

        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "find_recurring_payments",
            {"min_occurrences": min_occurrences, "days_back": days_back, "top_n": top_n}
        )

        # Enhance with recurring payment insights
        if isinstance(result, dict) and "data" in result:
            recurring_payments = result["data"].get("recurring_payments", [])
            
            max_payments_to_return = 10
            truncated_message = ""
            if len(recurring_payments) > max_payments_to_return:
                truncated_message = f" (showing top {max_payments_to_return} out of {len(recurring_payments)} found)"
                recurring_payments = recurring_payments[:max_payments_to_return]

            result = {
                **result,
                "recurring_insights": {
                    "subscription_management": _analyze_subscription_health(recurring_payments),
                    "cost_optimization": _identify_cost_optimization(recurring_payments),
                    "payment_reliability": _analyze_payment_reliability(recurring_payments),
                    "upcoming_payments": _predict_upcoming_payments(recurring_payments)
                },
                "recurring_payments": recurring_payments, # Add the (potentially truncated) list back
                "message": f"Successfully found recurring payments{truncated_message}."
            }

            return json.dumps(result, indent=2, ensure_ascii=False)

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Recurring payments tool error: {e}")
        return json.dumps({
            "error": f"Failed to find recurring payments: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def analyze_spending_trends_tool(tool_input: str) -> str:
    """Analyze spending trends with predictive insights."""
    months_back = 6  # Initialize with default value
    
    try:
        if tool_input.strip():
            try:
                parsed_input = json.loads(tool_input)
                months_back = parsed_input.get("months", 6)
            except json.JSONDecodeError:
                try:
                    months_back = int(tool_input.strip())
                except (ValueError, AttributeError):
                    months_back = 6
        
        months_back = max(1, min(months_back, 24))
        
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "analyze_spending_trends",
            {"months_back": months_back}
        )
        
        # Enhance with trend insights
        if isinstance(result, dict) and "data" in result:
            monthly_trends = result["data"].get("monthly_trends", [])
            
            result = {
                **result,
                "trend_insights": {
                    "trend_direction": _determine_overall_trend(monthly_trends),
                    "volatility_analysis": _analyze_spending_volatility(monthly_trends),
                    "seasonal_patterns": _detect_seasonal_patterns(monthly_trends),
                    "forecasting": _generate_spending_forecast(monthly_trends)
                }
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Spending trends tool error: {e}")
        return json.dumps({
            "error": f"Failed to analyze spending trends: {str(e)}",
            "months_attempted": months_back,
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def execute_custom_query_tool(tool_input: str) -> str:
    """Execute custom SQL query with validation and safety checks."""
    sql_query = ""
    
    try:
        try:
            parsed_input = json.loads(tool_input)
            sql_query = parsed_input.get("query", "")
        except json.JSONDecodeError:
            sql_query = tool_input.strip()
        
        if not sql_query:
            return json.dumps({
                "error": "SQL query is required",
                "example": "SELECT * FROM transactions WHERE description LIKE '%UPI%' LIMIT 10",
                "safety_note": "Only SELECT queries are allowed for security",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
        
        # Safety validation
        query_upper = sql_query.upper().strip()
        if not query_upper.startswith('SELECT'):
            return json.dumps({
                "error": "Only SELECT queries are allowed for security reasons",
                "provided_query": sql_query[:100] + "..." if len(sql_query) > 100 else sql_query,
                "timestamp": datetime.now().isoformat()
            }, indent=2)
        
        # Check for dangerous patterns
        dangerous_patterns = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for pattern in dangerous_patterns:
            if pattern in query_upper:
                return json.dumps({
                    "error": f"Query contains potentially dangerous operation: {pattern}",
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
        
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "execute_custom_query",
            {"sql_query": sql_query}
        )
        
        # Enhance with query insights
        if isinstance(result, dict) and "data" in result:
            results = result["data"].get("results", [])
            
            result = {
                **result,
                "query_insights": {
                    "result_summary": _analyze_query_results(results),
                    "data_quality": _assess_data_quality(results),
                    "suggested_improvements": _suggest_query_improvements(sql_query, results)
                }
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Custom query tool error: {e}")
        return json.dumps({
            "error": f"Failed to execute custom query: {str(e)}",
            "query_attempted": sql_query[:100] + "..." if len(sql_query) > 100 else sql_query,
            "timestamp": datetime.now().isoformat()
        }, indent=2)


async def get_database_schema_tool(tool_input: str) -> str:
    """Get database schema with documentation."""
    try:
        mcp_manager = await get_mcp_manager()
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "get_database_schema",
            {}
        )
        
        # Enhance with schema insights
        if isinstance(result, dict) and "data" in result:
            data_freshness = _assess_data_freshness()
            result = {
                **result,
                "schema_insights": {
                    "usage_examples": _generate_usage_examples(),
                    "data_relationships": _explain_data_relationships(),
                    "query_recommendations": _provide_query_recommendations(),
                    "data_freshness": data_freshness
                }
            }

        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Database schema tool error: {e}")
        return json.dumps({
            "error": f"Failed to get database schema: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, indent=2)


# Helper functions for insights

def _calculate_account_health(data: Dict[str, Any]) -> str:
    """Calculate account health score."""
    try:
        balance = data.get("current_balance_raw", 0)
        total_credits = data.get("total_credits_raw", 0)
        total_debits = data.get("total_debits_raw", 0)
        
        if balance > 50000:
            return "Excellent"
        elif balance > 20000:
            return "Good"
        elif balance > 5000:
            return "Fair"
        else:
            return "Needs Attention"
    except:
        return "Unknown"


def _analyze_cash_flow(data: Dict[str, Any]) -> str:
    """Analyze cash flow status."""
    try:
        total_credits = data.get("total_credits_raw", 0)
        total_debits = data.get("total_debits_raw", 0)
        
        if total_credits > total_debits * 1.2:
            return "Positive - Strong savings"
        elif total_credits > total_debits:
            return "Positive - Moderate savings"
        elif total_credits > total_debits * 0.9:
            return "Balanced"
        else:
            return "Negative - Spending exceeds income"
    except:
        return "Unable to analyze"


def _analyze_transaction_patterns(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns in recent transactions."""
    if not transactions:
        return {"pattern": "No transactions to analyze"}
    
    # Analyze transaction timing
    transaction_hours = []
    upi_count = 0
    atm_count = 0
    
    for transaction in transactions:
        description = transaction.get("description", "").lower()
        if "upi" in description:
            upi_count += 1
        if "atm" in description:
            atm_count += 1
    
    return {
        "total_analyzed": len(transactions),
        "upi_transactions": upi_count,
        "atm_withdrawals": atm_count,
        "digital_payment_ratio": upi_count / len(transactions) if transactions else 0
    }


def _categorize_recent_transactions(transactions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Categorize recent transactions."""
    categories = {
        "UPI/Digital": 0,
        "ATM/Cash": 0,
        "Transfers": 0,
        "Bills": 0,
        "Shopping": 0,
        "Other": 0
    }
    
    for transaction in transactions:
        description = transaction.get("description", "").lower()
        
        if any(term in description for term in ["upi", "paytm", "gpay", "phonepe"]):
            categories["UPI/Digital"] += 1
        elif any(term in description for term in ["atm", "cash"]):
            categories["ATM/Cash"] += 1
        elif any(term in description for term in ["transfer", "neft", "imps"]):
            categories["Transfers"] += 1
        elif any(term in description for term in ["bill", "electricity", "mobile"]):
            categories["Bills"] += 1
        elif any(term in description for term in ["amazon", "flipkart", "shop"]):
            categories["Shopping"] += 1
        else:
            categories["Other"] += 1
    
    return categories


def _detect_unusual_activity(transactions: List[Dict[str, Any]]) -> List[str]:
    """Detect unusual activity in transactions."""
    alerts = []
    
    if not transactions:
        return alerts
    
    # Check for large transactions
    amounts = [
        abs(t.get("debit_amount", 0) or t.get("credit_amount", 0))
        for t in transactions
    ]
    
    if amounts:
        avg_amount = sum(amounts) / len(amounts)
        max_amount = max(amounts)
        
        if max_amount > avg_amount * 5:
            alerts.append(f"Large transaction detected: ₹{max_amount:,.2f}")
    
    # Check for multiple ATM withdrawals
    atm_count = sum(1 for t in transactions if "atm" in t.get("description", "").lower())
    if atm_count > 3:
        alerts.append(f"Multiple ATM withdrawals: {atm_count} transactions")
    
    return alerts


def _analyze_amount_distribution(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze amount distribution in transactions."""
    if not transactions:
        return {"message": "No transactions to analyze"}
    
    amounts = []
    for transaction in transactions:
        amount = transaction.get("debit_amount") or transaction.get("credit_amount")
        if amount:
            amounts.append(abs(amount))
    
    if not amounts:
        return {"message": "No valid amounts found"}
    
    amounts.sort()
    n = len(amounts)
    
    return {
        "min_amount": min(amounts),
        "max_amount": max(amounts),
        "avg_amount": sum(amounts) / n,
        "median_amount": amounts[n // 2] if n % 2 == 1 else (amounts[n // 2 - 1] + amounts[n // 2]) / 2,
        "total_transactions": n
    }


def _analyze_time_distribution(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze time distribution of transactions."""
    if not transactions:
        return {"message": "No transactions to analyze"}
    
    dates = []
    for transaction in transactions:
        date_str = transaction.get("transaction_date")
        if date_str:
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date_obj)
            except ValueError:
                continue
    
    if not dates:
        return {"message": "No valid dates found"}
    
    dates.sort()
    
    return {
        "earliest_date": dates[0].strftime("%Y-%m-%d"),
        "latest_date": dates[-1].strftime("%Y-%m-%d"),
        "date_range_days": (dates[-1] - dates[0]).days,
        "transactions_with_dates": len(dates)
    }


def _suggest_search_refinements(pattern: str, transactions: List[Dict[str, Any]]) -> List[str]:
    """Suggest search refinements based on results."""
    suggestions = []
    
    if len(transactions) == 0:
        suggestions.append(f"Try a broader search term instead of '{pattern}'")
        suggestions.append("Common terms: UPI, ATM, transfer, bill, grocery")
    elif len(transactions) > 50:
        suggestions.append(f"'{pattern}' returned many results. Try adding date filters")
        suggestions.append("Add specific merchant names or amount ranges")
    
    return suggestions


def _calculate_daily_averages(transactions: List[Dict[str, Any]], start_dt: datetime, end_dt: datetime) -> Dict[str, float]:
    """Calculate daily averages for the period."""
    period_days = (end_dt - start_dt).days + 1
    
    total_debits = sum(t.get("debit_amount", 0) or 0 for t in transactions)
    total_credits = sum(t.get("credit_amount", 0) or 0 for t in transactions)
    
    return {
        "avg_daily_spending": total_debits / period_days if period_days > 0 else 0,
        "avg_daily_income": total_credits / period_days if period_days > 0 else 0,
        "avg_daily_transactions": len(transactions) / period_days if period_days > 0 else 0
    }


def _analyze_weekday_patterns(transactions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze spending patterns by weekday."""
    weekday_counts = {
        "Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0,
        "Friday": 0, "Saturday": 0, "Sunday": 0
    }

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for transaction in transactions:
        date_str = transaction.get("transaction_date")
        if not date_str:
            continue
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            weekday = weekday_names[date_obj.weekday()]
            weekday_counts[weekday] += 1
        except ValueError:
            continue

    return weekday_counts


def _calculate_spending_velocity(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate spending velocity metrics."""
    debits = [t for t in transactions if t.get("debit_amount")]
    
    if len(debits) < 2:
        return {"message": "Insufficient data for velocity calculation"}
    
    # Sort by date
    sorted_debits = sorted(debits, key=lambda x: x.get("transaction_date", ""))
    
    total_amount = sum(t.get("debit_amount", 0) for t in sorted_debits)
    
    # Calculate time span
    try:
        first_date = datetime.strptime(sorted_debits[0]["transaction_date"], "%Y-%m-%d")
        last_date = datetime.strptime(sorted_debits[-1]["transaction_date"], "%Y-%m-%d")
        days_span = (last_date - first_date).days + 1
    except:
        days_span = 1
    
    return {
        "total_spending": total_amount,
        "spending_per_day": total_amount / days_span if days_span > 0 else 0,
        "transaction_frequency": len(sorted_debits) / days_span if days_span > 0 else 0,
        "period_days": days_span
    }


def _compare_with_previous_period(start_dt: datetime, end_dt: datetime, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Compare current period with previous period (placeholder)."""
    period_days = (end_dt - start_dt).days + 1
    
    # This would require additional database queries in a real implementation
    return {
        "comparison_available": False,
        "message": f"Comparison with previous {period_days}-day period would require additional data",
        "current_period_days": period_days
    }


def _analyze_spending_consistency(daily_breakdown: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze spending consistency across days."""
    if not daily_breakdown:
        return {"message": "No daily data available"}
    
    daily_spending = [day.get("daily_debits", 0) for day in daily_breakdown]
    
    if not daily_spending:
        return {"message": "No spending data available"}
    
    avg_spending = sum(daily_spending) / len(daily_spending)
    
    # Calculate variance
    variance = sum((x - avg_spending) ** 2 for x in daily_spending) / len(daily_spending)
    std_deviation = variance ** 0.5
    
    consistency_score = 1 - (std_deviation / avg_spending) if avg_spending > 0 else 0
    
    return {
        "consistency_score": max(0, min(1, consistency_score)),
        "avg_daily_spending": avg_spending,
        "spending_volatility": std_deviation,
        "interpretation": "High" if consistency_score > 0.7 else "Medium" if consistency_score > 0.4 else "Low"
    }


def _find_peak_spending_days(daily_breakdown: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find days with highest spending."""
    if not daily_breakdown:
        return []
    
    # Sort by spending amount
    sorted_days = sorted(
        daily_breakdown,
        key=lambda x: x.get("daily_debits", 0),
        reverse=True
    )
    
    return sorted_days[:3]  # Top 3 spending days


def _generate_budget_recommendations(summary: Dict[str, Any]) -> List[str]:
    """Generate budget recommendations based on summary."""
    recommendations = []
    
    total_debits = summary.get("total_debits_raw", 0)
    total_credits = summary.get("total_credits_raw", 0)
    
    if total_debits > total_credits * 0.9:
        recommendations.append("Consider reducing discretionary spending")
    
    if total_credits > 0:
        savings_rate = (total_credits - total_debits) / total_credits
        if savings_rate < 0.2:
            recommendations.append("Aim to save at least 20% of income")
    
    recommendations.append("Track spending categories to identify optimization opportunities")
    
    return recommendations


def _compare_month_over_month(year: Optional[int], month: Optional[int], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Compare with previous month (placeholder)."""
    return {
        "comparison_available": False,
        "message": "Month-over-month comparison requires historical data analysis",
        "current_month": f"{year}-{month:02d}" if year and month else "Current"
    }


def _identify_top_spending_categories(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify top spending categories."""
    if not categories:
        return []
    
    # Sort by total spending
    sorted_categories = sorted(
        categories,
        key=lambda x: x.get("total_spent", 0),
        reverse=True
    )
    
    return sorted_categories[:5]  # Top 5 categories


def _identify_optimization_opportunities(categories: List[Dict[str, Any]]) -> List[str]:
    """Identify spending optimization opportunities."""
    opportunities = []
    
    if not categories:
        return opportunities
    
    total_spending = sum(cat.get("total_spent", 0) for cat in categories)
    
    for category in categories:
        category_name = category.get("category", "Unknown")
        category_spending = category.get("total_spent", 0)
        percentage = (category_spending / total_spending * 100) if total_spending > 0 else 0
        
        if percentage > 30:
            opportunities.append(f"{category_name} represents {percentage:.1f}% of spending - consider reviewing")
        elif percentage > 20 and "dining" in category_name.lower():
            opportunities.append(f"Dining expenses are high - consider cooking more at home")
        elif percentage > 15 and "online" in category_name.lower():
            opportunities.append(f"Online shopping is significant - review subscription services")
    
    return opportunities


def _analyze_spending_distribution(categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze spending distribution across categories."""
    if not categories:
        return {"message": "No categories to analyze"}
    
    total_spending = sum(cat.get("total_spent", 0) for cat in categories)
    
    # Calculate concentration
    category_percentages = [
        (cat.get("total_spent", 0) / total_spending * 100) if total_spending > 0 else 0
        for cat in categories
    ]
    
    # Gini coefficient approximation for concentration
    sorted_percentages = sorted(category_percentages)
    n = len(sorted_percentages)
    cumulative = sum((i + 1) * value for i, value in enumerate(sorted_percentages))
    concentration = (2 * cumulative) / (n * sum(sorted_percentages)) - (n + 1) / n if sorted_percentages else 0
    
    return {
        "total_categories": len(categories),
        "concentration_index": concentration,
        "distribution_type": "Concentrated" if concentration > 0.5 else "Diversified",
        "largest_category_percentage": max(category_percentages) if category_percentages else 0
    }


def _analyze_category_trends(categories: List[Dict[str, Any]], days: int) -> Dict[str, Any]:
    """Analyze category trends over the period."""
    # This would require historical data for proper trend analysis
    return {
        "analysis_period_days": days,
        "trend_analysis": "Requires historical data for trend comparison",
        "categories_analyzed": len(categories)
    }


def _analyze_subscription_health(recurring_payments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze health of recurring subscriptions."""
    if not recurring_payments:
        return {"message": "No recurring payments found"}
    
    total_monthly_cost = sum(
        payment.get("amount", 0) * (30 / payment.get("avg_days_between", 30))
        for payment in recurring_payments
    )
    
    return {
        "total_subscriptions": len(recurring_payments),
        "estimated_monthly_cost": total_monthly_cost,
        "cost_per_subscription": total_monthly_cost / len(recurring_payments) if recurring_payments else 0
    }


def _identify_cost_optimization(recurring_payments: List[Dict[str, Any]]) -> List[str]:
    """Identify cost optimization opportunities in recurring payments."""
    opportunities = []
    
    for payment in recurring_payments:
        payee = payment.get("payee", "Unknown")
        amount = payment.get("amount", 0)
        frequency = payment.get("frequency", 0)
        
        if amount > 1000 and frequency > 10:
            opportunities.append(f"Review {payee} - high frequency, high amount subscription")
        elif amount > 5000:
            opportunities.append(f"Consider negotiating {payee} - high-value subscription")
    
    return opportunities


def _analyze_payment_reliability(recurring_payments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze reliability of recurring payment patterns."""
    if not recurring_payments:
        return {"message": "No recurring payments to analyze"}
    
    reliable_count = sum(
        1 for payment in recurring_payments
        if payment.get("pattern_type") in ["Monthly", "Weekly", "Bi-weekly"]
    )
    
    return {
        "total_payments": len(recurring_payments),
        "reliable_patterns": reliable_count,
        "reliability_percentage": (reliable_count / len(recurring_payments) * 100) if recurring_payments else 0
    }


def _predict_upcoming_payments(recurring_payments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Predict upcoming payments based on patterns."""
    upcoming = []
    
    for payment in recurring_payments:
        last_seen = payment.get("last_seen")
        avg_days = payment.get("avg_days_between", 30)
        
        if last_seen:
            try:
                last_date = datetime.strptime(last_seen, "%Y-%m-%d")
                next_expected = last_date + timedelta(days=avg_days)
                
                if next_expected > datetime.now():
                    upcoming.append({
                        "payee": payment.get("payee"),
                        "amount": payment.get("amount"),
                        "expected_date": next_expected.strftime("%Y-%m-%d"),
                        "confidence": "High" if payment.get("pattern_type") in ["Monthly", "Weekly"] else "Medium"
                    })
            except ValueError:
                continue
    
    return sorted(upcoming, key=lambda x: x["expected_date"])[:5]  # Next 5 payments


def _determine_overall_trend(monthly_trends: List[Dict[str, Any]]) -> str:
    """Determine overall spending trend."""
    if len(monthly_trends) < 2:
        return "Insufficient data for trend analysis"
    
    # Compare first and last months
    first_month = monthly_trends[-1].get("total_debits", 0)  # Oldest
    last_month = monthly_trends[0].get("total_debits", 0)    # Most recent
    
    if last_month > first_month * 1.1:
        return "Increasing"
    elif last_month < first_month * 0.9:
        return "Decreasing"
    else:
        return "Stable"


def _analyze_spending_volatility(monthly_trends: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze volatility in monthly spending."""
    if len(monthly_trends) < 3:
        return {"message": "Insufficient data for volatility analysis"}
    
    spending_amounts = [month.get("total_debits", 0) for month in monthly_trends]
    avg_spending = sum(spending_amounts) / len(spending_amounts)
    
    variance = sum((x - avg_spending) ** 2 for x in spending_amounts) / len(spending_amounts)
    std_deviation = variance ** 0.5
    
    coefficient_of_variation = (std_deviation / avg_spending) if avg_spending > 0 else 0
    
    return {
        "volatility_coefficient": coefficient_of_variation,
        "volatility_level": "High" if coefficient_of_variation > 0.3 else "Medium" if coefficient_of_variation > 0.15 else "Low",
        "avg_monthly_spending": avg_spending,
        "spending_standard_deviation": std_deviation
    }


def _detect_seasonal_patterns(monthly_trends: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect seasonal patterns in spending."""
    if len(monthly_trends) < 12:
        return {"message": "Need at least 12 months of data for seasonal analysis"}
    
    # Group by month number
    month_spending = {}
    for trend in monthly_trends:
        month_str = trend.get("month", "")
        if month_str and len(month_str) >= 7:  # Format: YYYY-MM
            month_num = int(month_str.split("-")[1])
            spending = trend.get("total_debits", 0)
            
            if month_num not in month_spending:
                month_spending[month_num] = []
            month_spending[month_num].append(spending)
    
    # Calculate averages
    month_averages = {
        month: sum(amounts) / len(amounts)
        for month, amounts in month_spending.items()
        if amounts
    }
    
    if len(month_averages) < 12:
        return {"message": "Incomplete seasonal data"}
    
    # Find peak and low spending months
    peak_month = max(month_averages.keys(), key=lambda k: month_averages[k])
    low_month = min(month_averages.keys(), key=lambda k: month_averages[k])
    
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    return {
        "peak_spending_month": month_names[peak_month - 1],
        "lowest_spending_month": month_names[low_month - 1],
        "seasonal_variation": (month_averages[peak_month] - month_averages[low_month]) / month_averages[low_month] * 100
    }


def _generate_spending_forecast(monthly_trends: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate simple spending forecast."""
    if len(monthly_trends) < 3:
        return {"message": "Insufficient data for forecasting"}
    
    # Simple linear trend
    spending_amounts = [month.get("total_debits", 0) for month in monthly_trends[-6:]]  # Last 6 months
    
    if len(spending_amounts) < 3:
        return {"message": "Need at least 3 months for forecast"}
    
    # Calculate simple trend
    recent_avg = sum(spending_amounts[-3:]) / 3
    older_avg = sum(spending_amounts[-6:-3]) / 3 if len(spending_amounts) >= 6 else recent_avg
    
    trend = (recent_avg - older_avg) / older_avg * 100 if older_avg > 0 else 0
    
    next_month_forecast = recent_avg * (1 + trend / 100)
    
    return {
        "next_month_forecast": next_month_forecast,
        "trend_percentage": trend,
        "confidence": "Low - Simple linear projection",
        "based_on_months": len(spending_amounts)
    }


def _analyze_query_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze custom query results."""
    if not results:
        return {"message": "No results returned"}
    
    return {
        "row_count": len(results),
        "columns": list(results[0].keys()) if results else [],
        "has_amount_columns": any("amount" in col.lower() for col in (results[0].keys() if results else [])),
        "has_date_columns": any("date" in col.lower() for col in (results[0].keys() if results else []))
    }


def _assess_data_quality(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess data quality of query results."""
    if not results:
        return {"message": "No data to assess"}
    
    total_rows = len(results)
    columns = list(results[0].keys()) if results else []
    
    null_counts = {}
    for col in columns:
        null_count = sum(1 for row in results if row.get(col) is None or row.get(col) == "")
        null_counts[col] = null_count
    
    return {
        "total_rows": total_rows,
        "completeness": {
            col: (total_rows - null_count) / total_rows * 100
            for col, null_count in null_counts.items()
        },
        "overall_completeness": sum(
            (total_rows - null_count) / total_rows
            for null_count in null_counts.values()
        ) / len(columns) * 100 if columns else 0
    }


def _suggest_query_improvements(sql_query: str, results: List[Dict[str, Any]]) -> List[str]:
    """Suggest query improvements."""
    suggestions = []
    
    if len(results) > 1000:
        suggestions.append("Consider adding LIMIT clause to improve performance")
    
    if "ORDER BY" not in sql_query.upper():
        suggestions.append("Add ORDER BY clause for consistent result ordering")
    
    if "WHERE" not in sql_query.upper():
        suggestions.append("Consider adding WHERE clause to filter results")
    
    return suggestions


def _generate_usage_examples() -> List[Dict[str, str]]:
    """Generate usage examples for the database schema."""
    return [
        {
            "description": "Get recent UPI transactions",
            "query": "SELECT * FROM transactions WHERE description LIKE '%UPI%' ORDER BY transaction_date DESC LIMIT 10"
        },
        {
            "description": "Calculate monthly spending",
            "query": "SELECT strftime('%Y-%m', transaction_date) as month, SUM(debit_amount) as total_spending FROM transactions WHERE debit_amount IS NOT NULL GROUP BY month ORDER BY month DESC"
        },
        {
            "description": "Find largest transactions",
            "query": "SELECT transaction_date, description, debit_amount, credit_amount FROM transactions WHERE debit_amount > 5000 OR credit_amount > 5000 ORDER BY COALESCE(debit_amount, credit_amount) DESC LIMIT 10"
        }
    ]


def _explain_data_relationships() -> Dict[str, str]:
    """Explain data relationships in the schema."""
    return {
        "transactions": "Main table containing all bank transactions",
        "statement_id": "Links transactions to bank statement files",
        "transaction_date": "Date when transaction occurred",
        "value_date": "Date when transaction was processed by bank",
        "debit_amount": "Money going out of account (expenses)",
        "credit_amount": "Money coming into account (income)",
        "balance": "Account balance after transaction",
        "reference_number": "Bank reference for transaction",
        "beneficiary_name": "Name of transaction recipient/sender",
        "upi_id": "UPI ID for digital payments"
    }


def _provide_query_recommendations() -> List[str]:
    """Provide query recommendations."""
    return [
        "Always use LIMIT to avoid large result sets",
        "Filter by date ranges for better performance",
        "Use indexes on frequently queried columns",
        "Consider using aggregation functions for summaries",
        "Join with lookup tables for categorization"
    ]


def _assess_data_freshness() -> Dict[str, Any]:
    """Assess data freshness (placeholder)."""
    # In a real implementation, this would query the database for the latest transaction date
    return {
        "status": "unknown",
        "last_checked": datetime.now().isoformat(),
        "message": "Data freshness assessment is a placeholder."
    }


def get_financial_tools() -> List[Dict[str, Any]]:
    """Get configuration for all financial tools."""
    return [
        {
            "name": "get_account_summary",
            "description": "Get comprehensive account balance and transaction summary with health insights. Provides overview of total transactions, date range, current balance in INR, and financial health assessment.",
            "func": get_account_summary_tool,
            "cache_ttl": 30,
            "timeout": 15.0,
            "rate_limit": 20
        },
        {
            "name": "get_recent_transactions",
            "description": "Get the most recent N transactions with pattern analysis. Input: {\"limit\": 10} or just a number. Shows transaction details including amounts in INR with spending insights.",
            "func": get_recent_transactions_tool,
            "cache_ttl": 10,
            "timeout": 15.0,
            "rate_limit": 30
        },
        {
            "name": "search_transactions",
            "description": "Search transactions by description pattern with intelligent matching. Input: {\"pattern\": \"search_term\", \"limit\": 20} or just the search term. All amounts shown in INR with search optimization suggestions.",
            "func": search_transactions_tool,
            "cache_ttl": 60,
            "timeout": 20.0,
            "rate_limit": 25
        },
        {
            "name": "get_transactions_by_date_range",
            "description": "Get transactions within a date range with period analysis. Input: {\"start_date\": \"YYYY-MM-DD\", \"end_date\": \"YYYY-MM-DD\"}. Returns transactions with INR amounts and period insights.",
            "func": get_transactions_by_date_range_tool,
            "cache_ttl": 300,
            "timeout": 25.0,
            "rate_limit": 15
        },
        {
            "name": "get_monthly_summary",
            "description": "Get monthly spending summary with analytics. Input: {\"year\": 2024, \"month\": 1} (optional). Shows monthly totals in INR with spending consistency analysis.",
            "func": get_monthly_summary_tool,
            "cache_ttl": 300,
            "timeout": 20.0,
            "rate_limit": 20
        },
        {
            "name": "get_spending_by_category",
            "description": "Analyze spending by category with optimization insights for the last N days. Input: {\"days\": 30} or empty for 30 days. Categories based on transaction descriptions, amounts in INR with recommendations.",
            "func": get_spending_by_category_tool,
            "cache_ttl": 180,
            "timeout": 25.0,
            "rate_limit": 20
        },
        {
            "name": "find_recurring_payments",
            "description": "Find recurring payments and subscriptions with management insights. Input: {\"min_occurrences\": 3, \"days_back\": 90} (optional). Identifies potential recurring transactions with INR amounts and optimization opportunities.",
            "func": find_recurring_payments_tool,
            "cache_ttl": 600,
            "timeout": 30.0,
            "rate_limit": 10
        },
        {
            "name": "analyze_spending_trends",
            "description": "Analyze spending trends over time with forecasting. Input: {\"months\": 6} or empty for 6 months. Shows spending patterns with INR amounts, volatility analysis, and predictions.",
            "func": analyze_spending_trends_tool,
            "cache_ttl": 600,
            "timeout": 30.0,
            "rate_limit": 10
        },
        {
            "name": "execute_custom_query",
            "description": "Execute a custom SQL query with safety validation (SELECT only). Input: {\"query\": \"SELECT * FROM transactions LIMIT 5\"} or just the SQL. Results show INR amounts with data quality assessment.",
            "func": execute_custom_query_tool,
            "cache_ttl": 0,
            "timeout": 20.0,
            "rate_limit": 5
        },
        {
            "name": "get_database_schema",
            "description": "Get database schema and table structure information with usage examples. No input required. Shows table definitions, sample data, and query recommendations.",
            "func": get_database_schema_tool,
            "cache_ttl": 3600,
            "timeout": 10.0,
            "rate_limit": 10
        }
    ]
