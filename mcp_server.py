#!/usr/bin/env python3
"""
FastMCP server for querying consolidated financial database.
Provides standardized tools for LLMs to access bank statement data with
high-performance features including connection pooling, query caching,
and optimized database operations.

Key features:
- Connection pooling for database operations
- Intelligent query result caching with TTL
- Prepared statements for better performance
- Batch processing capabilities
- Performance monitoring and statistics
- Optimized JSON serialization

All amounts are in INR (Indian Rupees).
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP

# Import database manager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database.db import DatabaseManager, QueryCache, get_db_manager

# Initialize the MCP server
mcp = FastMCP("Financial Data Server")

# Get database manager
db_manager = get_db_manager()

def format_inr(amount: float) -> str:
    """Format amount as INR currency."""
    if amount is None:
        return "₹0.00"
    return f"₹{amount:,.2f}"

def serialize_result(data: Any, include_meta: bool = True) -> str:
    """Optimized JSON serialization with metadata."""
    if include_meta:
        result = {
            "data": data,
            "currency": "INR",
            "generated_at": datetime.now().isoformat(),
            "server": "optimized-financial-mcp"
        }
        return json.dumps(result, indent=2, default=str)
    else:
        return json.dumps(data, indent=2, default=str)

@mcp.tool()
def get_account_summary() -> str:
    """Get a summary of all bank accounts and their current status. All amounts in INR."""
    try:
        # Use cached queries for better performance
        total_result = db_manager.execute_query(
            "SELECT COUNT(*) as count FROM transactions",
            cache_ttl=300  # Cache for 30 seconds
        )
        total_transactions = total_result.data[0]['count']
        
        date_range_result = db_manager.execute_query(
            "SELECT MIN(transaction_date) as earliest_date, MAX(transaction_date) as latest_date FROM transactions",
            cache_ttl=300  # Cache for 5 minutes
        )
        date_range = date_range_result.data[0]
        
        latest_balance_result = db_manager.execute_query(
            "SELECT balance FROM transactions ORDER BY transaction_date DESC, transaction_id DESC LIMIT 1",
            cache_ttl=300  # Cache for 5 minutes
        )
        latest_balance = latest_balance_result.data[0]['balance']
        
        totals_result = db_manager.execute_query(
            "SELECT SUM(debit_amount) as total_debits, SUM(credit_amount) as total_credits FROM transactions WHERE debit_amount IS NOT NULL OR credit_amount IS NOT NULL",
            cache_ttl=300  # Cache for 5 minutes
        )
        totals = totals_result.data[0]
        
        summary = {
            "total_transactions": total_transactions,
            "date_range": {
                "earliest": date_range['earliest_date'], 
                "latest": date_range['latest_date']
            },
            "current_balance_inr": format_inr(latest_balance),
            "current_balance_raw": latest_balance,
            "total_debits_inr": format_inr(totals['total_debits']) if totals['total_debits'] else "₹0.00",
            "total_credits_inr": format_inr(totals['total_credits']) if totals['total_credits'] else "₹0.00",
            "total_debits_raw": totals['total_debits'] or 0,
            "total_credits_raw": totals['total_credits'] or 0,
            "performance": {
                "queries_executed": 4,
                "cache_hits": sum(1 for r in [total_result, date_range_result, latest_balance_result, totals_result] if r.cached),
                "total_execution_time": sum(r.execution_time for r in [total_result, date_range_result, latest_balance_result, totals_result])
            }
        }
        
        return serialize_result(summary)
    except Exception as e:
        return serialize_result({"error": f"Error getting account summary: {str(e)}"})

@mcp.tool()
def get_recent_transactions(limit: int = 10) -> str:
    """Get the most recent N transactions. Defaults to 10 transactions. All amounts in INR."""
    try:
        # Validate limit
        limit = max(1, min(limit, 100))  # Limit between 1 and 100
        
        result = db_manager.execute_query(
            """
            SELECT transaction_id, transaction_date, description, transaction_type,
                   debit_amount, credit_amount, balance, reference_number, beneficiary_name, upi_id
            FROM transactions 
            ORDER BY transaction_date DESC, transaction_id DESC 
            LIMIT ?
            """,
            (limit,),
            cache_ttl=10  # Cache for 10 seconds
        )
        
        transactions = result.data
        
        # Add INR formatting
        for transaction in transactions:
            if transaction['debit_amount']:
                transaction['debit_amount_inr'] = format_inr(transaction['debit_amount'])
            if transaction['credit_amount']:
                transaction['credit_amount_inr'] = format_inr(transaction['credit_amount'])
            if transaction['balance']:
                transaction['balance_inr'] = format_inr(transaction['balance'])
        
        response = {
            "transactions": transactions,
            "count": len(transactions),
            "limit_applied": limit,
            "performance": {
                "execution_time": result.execution_time,
                "cached": result.cached
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error getting recent transactions: {str(e)}"})

@mcp.tool()
def search_transactions(pattern: str, limit: int = 20) -> str:
    """Search transactions by description pattern. Case-insensitive search. All amounts in INR."""
    try:
        if not pattern:
            return serialize_result({"error": "Search pattern cannot be empty"})
        
        # Validate limit
        limit = max(1, min(limit, 100))
        
        result = db_manager.execute_query(
            """
            SELECT transaction_id, transaction_date, description, transaction_type,
                   debit_amount, credit_amount, balance, reference_number, beneficiary_name, upi_id
            FROM transactions 
            WHERE description LIKE ? 
            ORDER BY transaction_date DESC 
            LIMIT ?
            """,
            (f"%{pattern}%", limit),
            cache_ttl=60  # Cache for 1 minute
        )
        
        transactions = result.data
        
        # Add INR formatting
        for transaction in transactions:
            if transaction['debit_amount']:
                transaction['debit_amount_inr'] = format_inr(transaction['debit_amount'])
            if transaction['credit_amount']:
                transaction['credit_amount_inr'] = format_inr(transaction['credit_amount'])
            if transaction['balance']:
                transaction['balance_inr'] = format_inr(transaction['balance'])
        
        response = {
            "transactions": transactions,
            "search_pattern": pattern,
            "matches_found": len(transactions),
            "limit_applied": limit,
            "performance": {
                "execution_time": result.execution_time,
                "cached": result.cached
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error searching transactions: {str(e)}"})

@mcp.tool()
def get_transactions_by_date_range(start_date: str, end_date: str) -> str:
    """Get transactions within a specific date range (YYYY-MM-DD format). All amounts in INR."""
    try:
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return serialize_result({"error": "Invalid date format. Use YYYY-MM-DD"})
        
        result = db_manager.execute_query(
            """
            SELECT transaction_id, transaction_date, description, transaction_type,
                   debit_amount, credit_amount, balance, reference_number, beneficiary_name, upi_id
            FROM transactions 
            WHERE transaction_date BETWEEN ? AND ?
            ORDER BY transaction_date DESC
            """,
            (start_date, end_date),
            cache_ttl=300  # Cache for 5 minutes
        )
        
        transactions = result.data
        
        # Calculate totals and add INR formatting
        total_debits = 0
        total_credits = 0
        
        for transaction in transactions:
            if transaction['debit_amount']:
                transaction['debit_amount_inr'] = format_inr(transaction['debit_amount'])
                total_debits += transaction['debit_amount']
            if transaction['credit_amount']:
                transaction['credit_amount_inr'] = format_inr(transaction['credit_amount'])
                total_credits += transaction['credit_amount']
            if transaction['balance']:
                transaction['balance_inr'] = format_inr(transaction['balance'])
        
        response = {
            "transactions": transactions,
            "date_range": {"start": start_date, "end": end_date},
            "summary": {
                "total_transactions": len(transactions),
                "total_debits_inr": format_inr(total_debits),
                "total_credits_inr": format_inr(total_credits),
                "net_amount_inr": format_inr(total_credits - total_debits),
                "total_debits_raw": total_debits,
                "total_credits_raw": total_credits,
                "net_amount_raw": total_credits - total_debits
            },
            "performance": {
                "execution_time": result.execution_time,
                "cached": result.cached
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error getting transactions by date range: {str(e)}"})

@mcp.tool()
def get_monthly_summary(year: Optional[int] = None, month: Optional[int] = None) -> str:
    """Get monthly spending summary. If year/month not provided, uses current month. All amounts in INR."""
    try:
        if not year or not month:
            now = datetime.now()
            year = year or now.year
            month = month or now.month
        
        # Validate month
        if not (1 <= month <= 12):
            return serialize_result({"error": "Month must be between 1 and 12"})
        
        start_date = f"{year}-{month:02d}-01"
        # Calculate end date
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"
        
        result = db_manager.execute_query(
            """
            SELECT 
                COUNT(*) as transaction_count,
                SUM(CASE WHEN debit_amount IS NOT NULL THEN debit_amount ELSE 0 END) as total_debits,
                SUM(CASE WHEN credit_amount IS NOT NULL THEN credit_amount ELSE 0 END) as total_credits,
                AVG(CASE WHEN debit_amount IS NOT NULL THEN debit_amount ELSE 0 END) as avg_debit,
                AVG(CASE WHEN credit_amount IS NOT NULL THEN credit_amount ELSE 0 END) as avg_credit
            FROM transactions 
            WHERE transaction_date >= ? AND transaction_date < ?
            """,
            (start_date, end_date),
            cache_ttl=300  # Cache for 5 minutes
        )
        
        summary = result.data[0]
        
        # Get daily breakdown
        daily_result = db_manager.execute_query(
            """
            SELECT 
                transaction_date,
                COUNT(*) as daily_transactions,
                SUM(CASE WHEN debit_amount IS NOT NULL THEN debit_amount ELSE 0 END) as daily_debits,
                SUM(CASE WHEN credit_amount IS NOT NULL THEN credit_amount ELSE 0 END) as daily_credits
            FROM transactions 
            WHERE transaction_date >= ? AND transaction_date < ?
            GROUP BY transaction_date
            ORDER BY transaction_date
            """,
            (start_date, end_date),
            cache_ttl=300
        )
        
        daily_breakdown = daily_result.data
        
        # Add INR formatting to daily breakdown
        for day in daily_breakdown:
            day['daily_debits_inr'] = format_inr(day['daily_debits'])
            day['daily_credits_inr'] = format_inr(day['daily_credits'])
            day['daily_net_inr'] = format_inr(day['daily_credits'] - day['daily_debits'])
        
        response = {
            "period": f"{year}-{month:02d}",
            "summary": {
                "transaction_count": summary['transaction_count'],
                "total_debits_inr": format_inr(summary['total_debits']),
                "total_credits_inr": format_inr(summary['total_credits']),
                "net_amount_inr": format_inr(summary['total_credits'] - summary['total_debits']),
                "avg_debit_inr": format_inr(summary['avg_debit']),
                "avg_credit_inr": format_inr(summary['avg_credit']),
                "total_debits_raw": summary['total_debits'],
                "total_credits_raw": summary['total_credits'],
                "net_amount_raw": summary['total_credits'] - summary['total_debits']
            },
            "daily_breakdown": daily_breakdown,
            "performance": {
                "execution_time": result.execution_time + daily_result.execution_time,
                "cached": result.cached and daily_result.cached
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error getting monthly summary: {str(e)}"})

@mcp.tool()
def get_spending_by_category(days: int = 30) -> str:
    """Analyze spending by category for the last N days. Categories based on transaction descriptions. All amounts in INR."""
    try:
        # Validate days
        days = max(1, min(days, 365))
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        result = db_manager.execute_query(
            """
            SELECT 
                CASE 
                    WHEN LOWER(description) LIKE '%upi%' OR LOWER(description) LIKE '%paytm%' OR LOWER(description) LIKE '%gpay%' OR LOWER(description) LIKE '%phonepe%' THEN 'UPI/Digital Payments'
                    WHEN LOWER(description) LIKE '%atm%' OR LOWER(description) LIKE '%cash%' THEN 'ATM/Cash Withdrawal'
                    WHEN LOWER(description) LIKE '%grocery%' OR LOWER(description) LIKE '%supermarket%' OR LOWER(description) LIKE '%mart%' THEN 'Groceries'
                    WHEN LOWER(description) LIKE '%fuel%' OR LOWER(description) LIKE '%petrol%' OR LOWER(description) LIKE '%diesel%' THEN 'Fuel'
                    WHEN LOWER(description) LIKE '%restaurant%' OR LOWER(description) LIKE '%food%' OR LOWER(description) LIKE '%cafe%' THEN 'Food & Dining'
                    WHEN LOWER(description) LIKE '%transfer%' OR LOWER(description) LIKE '%neft%' OR LOWER(description) LIKE '%imps%' THEN 'Transfers'
                    WHEN LOWER(description) LIKE '%bill%' OR LOWER(description) LIKE '%electricity%' OR LOWER(description) LIKE '%mobile%' THEN 'Bill Payments'
                    WHEN LOWER(description) LIKE '%amazon%' OR LOWER(description) LIKE '%flipkart%' OR LOWER(description) LIKE '%online%' THEN 'Online Shopping'
                    WHEN LOWER(description) LIKE '%medical%' OR LOWER(description) LIKE '%hospital%' OR LOWER(description) LIKE '%pharmacy%' THEN 'Healthcare'
                    WHEN LOWER(description) LIKE '%salary%' OR LOWER(description) LIKE '%income%' THEN 'Income'
                    ELSE 'Other'
                END as category,
                COUNT(*) as transaction_count,
                SUM(CASE WHEN debit_amount IS NOT NULL THEN debit_amount ELSE 0 END) as total_spent,
                AVG(CASE WHEN debit_amount IS NOT NULL THEN debit_amount ELSE 0 END) as avg_transaction
            FROM transactions 
            WHERE transaction_date >= ?
            GROUP BY category
            ORDER BY total_spent DESC
            """,
            (start_date,),
            cache_ttl=180  # Cache for 3 minutes
        )
        
        categories = result.data
        
        # Calculate total for percentages and add INR formatting
        total_spent = sum(cat['total_spent'] for cat in categories)
        
        for category in categories:
            category['total_spent_inr'] = format_inr(category['total_spent'])
            category['avg_transaction_inr'] = format_inr(category['avg_transaction'])
            category['percentage'] = (category['total_spent'] / total_spent * 100) if total_spent > 0 else 0
        
        response = {
            "analysis_period_days": days,
            "total_spent_inr": format_inr(total_spent),
            "total_spent_raw": total_spent,
            "categories": categories,
            "performance": {
                "execution_time": result.execution_time,
                "cached": result.cached
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error analyzing spending by category: {str(e)}"})

@mcp.tool()
def execute_custom_query(sql_query: str) -> str:
    """Execute a custom SQL query (SELECT only for security). Results show amounts in INR."""
    try:
        if not sql_query.strip().upper().startswith('SELECT'):
            return serialize_result({"error": "Only SELECT queries are allowed for security reasons"})
        
        result = db_manager.execute_query(
            sql_query,
            cache_ttl=0  # No caching for custom queries
        )
        
        results = result.data
        
        # Add INR formatting to any amount columns
        for row in results:
            # Create a list of keys to avoid "dictionary changed size during iteration"
            original_keys = list(row.keys())
            for key in original_keys:
                value = row[key]
                if 'amount' in key.lower() or 'balance' in key.lower():
                    if isinstance(value, (int, float)) and value is not None:
                        row[f"{key}_inr"] = format_inr(value)
        
        response = {
            "query": sql_query,
            "result_count": len(results),
            "results": results[:100],  # Limit results to prevent massive responses
            "performance": {
                "execution_time": result.execution_time,
                "cached": result.cached
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error executing query: {str(e)}"})

@mcp.tool()
def get_database_schema() -> str:
    """Get the database schema information including table structure and sample data."""
    try:
        # Get table schema
        schema_result = db_manager.execute_query(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='transactions'",
            cache_ttl=3600  # Cache for 1 hour
        )
        
        # Get sample data
        sample_result = db_manager.execute_query(
            "SELECT * FROM transactions ORDER BY transaction_date DESC LIMIT 3",
            cache_ttl=60  # Cache for 1 minute
        )
        
        # Get column info
        columns_result = db_manager.execute_query(
            "PRAGMA table_info(transactions)",
            cache_ttl=3600  # Cache for 1 hour
        )
        
        response = {
            "table_schema": schema_result.data[0]['sql'] if schema_result.data else "No schema found",
            "columns": columns_result.data,
            "sample_records": sample_result.data,
            "description": "Table contains Indian bank transaction data with columns: transaction_id, statement_id, transaction_date, value_date, description, transaction_type, debit_amount (INR), credit_amount (INR), balance (INR), reference_number, beneficiary_name, upi_id, bank_code, created_at",
            "performance": {
                "total_execution_time": schema_result.execution_time + sample_result.execution_time + columns_result.execution_time,
                "cached_queries": sum(1 for r in [schema_result, sample_result, columns_result] if r.cached)
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error getting database schema: {str(e)}"})

@mcp.tool()
def find_recurring_payments(min_occurrences: int = 3, days_back: int = 90) -> str:
    """Find potential recurring payments by analyzing transaction patterns. All amounts in INR."""
    try:
        # Validate parameters
        min_occurrences = max(2, min(min_occurrences, 10))
        days_back = max(7, min(days_back, 365))
        
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        result = db_manager.execute_query(
            """
            WITH transaction_intervals AS (
                SELECT 
                    CASE 
                        WHEN beneficiary_name IS NOT NULL AND LENGTH(beneficiary_name) > 3 
                        THEN beneficiary_name
                        ELSE SUBSTR(description, 1, 30)
                    END as payee,
                    ROUND(debit_amount, 2) as amount,
                    transaction_date,
                    julianday(transaction_date) - julianday(LAG(transaction_date) OVER (
                        PARTITION BY 
                            CASE 
                                WHEN beneficiary_name IS NOT NULL AND LENGTH(beneficiary_name) > 3 
                                THEN beneficiary_name
                                ELSE SUBSTR(description, 1, 30)
                            END, 
                            ROUND(debit_amount, 2) 
                        ORDER BY transaction_date
                    )) as days_between
                FROM transactions 
                WHERE debit_amount IS NOT NULL 
                    AND debit_amount > 100  -- Filter out small transactions
                    AND transaction_date >= ?
            ),
            recurring_candidates AS (
                SELECT 
                    payee,
                    amount,
                    COUNT(*) as frequency,
                    MIN(transaction_date) as first_seen,
                    MAX(transaction_date) as last_seen,
                    AVG(days_between) as avg_days_between
                FROM transaction_intervals
                GROUP BY payee, amount
                HAVING COUNT(*) >= ?
                    AND julianday(MAX(transaction_date)) - julianday(MIN(transaction_date)) >= 14  -- At least 2 weeks apart
            )
            SELECT 
                payee,
                amount,
                frequency,
                first_seen,
                last_seen,
                ROUND(avg_days_between, 1) as avg_days_between,
                CASE 
                    WHEN avg_days_between BETWEEN 28 AND 32 THEN 'Monthly'
                    WHEN avg_days_between BETWEEN 6 AND 8 THEN 'Weekly'
                    WHEN avg_days_between BETWEEN 13 AND 16 THEN 'Bi-weekly'
                    WHEN avg_days_between BETWEEN 89 AND 93 THEN 'Quarterly'
                    ELSE 'Irregular'
                END as pattern_type,
                amount * frequency as total_spent
            FROM recurring_candidates
            ORDER BY frequency DESC, total_spent DESC
            LIMIT 20
            """,
            (start_date, min_occurrences),
            cache_ttl=300  # Cache for 5 minutes
        )
        
        recurring_payments = result.data
        
        # Add INR formatting and calculate statistics
        total_recurring_amount = 0
        for payment in recurring_payments:
            payment['amount_inr'] = format_inr(payment['amount'])
            payment['total_spent_inr'] = format_inr(payment['total_spent'])
            total_recurring_amount += payment['total_spent']
        
        response = {
            "analysis_period_days": days_back,
            "minimum_occurrences": min_occurrences,
            "recurring_payments": recurring_payments,
            "summary": {
                "total_recurring_payees": len(recurring_payments),
                "total_recurring_amount_inr": format_inr(total_recurring_amount),
                "total_recurring_amount_raw": total_recurring_amount,
                "pattern_distribution": {
                    pattern: len([p for p in recurring_payments if p['pattern_type'] == pattern])
                    for pattern in ['Monthly', 'Weekly', 'Bi-weekly', 'Quarterly', 'Irregular']
                }
            },
            "performance": {
                "execution_time": result.execution_time,
                "cached": result.cached
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error finding recurring payments: {str(e)}"})

@mcp.tool()
def analyze_spending_trends(months_back: int = 6) -> str:
    """Analyze spending trends over the last N months. All amounts in INR."""
    try:
        # Validate months_back
        months_back = max(1, min(months_back, 24))
        
        # Calculate start date
        start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
        
        result = db_manager.execute_query(
            """
            SELECT 
                strftime('%Y-%m', transaction_date) as month,
                COUNT(*) as transaction_count,
                SUM(CASE WHEN debit_amount IS NOT NULL THEN debit_amount ELSE 0 END) as total_debits,
                SUM(CASE WHEN credit_amount IS NOT NULL THEN credit_amount ELSE 0 END) as total_credits,
                AVG(CASE WHEN debit_amount IS NOT NULL THEN debit_amount ELSE 0 END) as avg_debit,
                AVG(CASE WHEN credit_amount IS NOT NULL THEN credit_amount ELSE 0 END) as avg_credit,
                
                -- Category breakdowns
                SUM(CASE WHEN debit_amount IS NOT NULL AND (LOWER(description) LIKE '%upi%' OR LOWER(description) LIKE '%paytm%' OR LOWER(description) LIKE '%gpay%') THEN debit_amount ELSE 0 END) as upi_spending,
                SUM(CASE WHEN debit_amount IS NOT NULL AND (LOWER(description) LIKE '%atm%' OR LOWER(description) LIKE '%cash%') THEN debit_amount ELSE 0 END) as cash_withdrawals,
                SUM(CASE WHEN debit_amount IS NOT NULL AND (LOWER(description) LIKE '%grocery%' OR LOWER(description) LIKE '%supermarket%') THEN debit_amount ELSE 0 END) as grocery_spending,
                SUM(CASE WHEN debit_amount IS NOT NULL AND (LOWER(description) LIKE '%fuel%' OR LOWER(description) LIKE '%petrol%') THEN debit_amount ELSE 0 END) as fuel_spending
                
            FROM transactions 
            WHERE transaction_date >= ?
            GROUP BY strftime('%Y-%m', transaction_date)
            ORDER BY month DESC
            """,
            (start_date,),
            cache_ttl=300  # Cache for 5 minutes
        )
        
        monthly_data = result.data
        
        # Calculate trends and add INR formatting
        if len(monthly_data) > 1:
            # Calculate month-over-month changes
            for i in range(len(monthly_data) - 1):
                current = monthly_data[i]
                previous = monthly_data[i + 1]
                
                # Add INR formatting
                current['total_debits_inr'] = format_inr(current['total_debits'])
                current['total_credits_inr'] = format_inr(current['total_credits'])
                current['avg_debit_inr'] = format_inr(current['avg_debit'])
                current['avg_credit_inr'] = format_inr(current['avg_credit'])
                current['upi_spending_inr'] = format_inr(current['upi_spending'])
                current['cash_withdrawals_inr'] = format_inr(current['cash_withdrawals'])
                current['grocery_spending_inr'] = format_inr(current['grocery_spending'])
                current['fuel_spending_inr'] = format_inr(current['fuel_spending'])
                current['net_amount_inr'] = format_inr(current['total_credits'] - current['total_debits'])
                
                # Calculate percentage changes
                if previous['total_debits'] > 0:
                    current['spending_change_pct'] = ((current['total_debits'] - previous['total_debits']) / previous['total_debits']) * 100
                else:
                    current['spending_change_pct'] = 0
                    
                if previous['total_credits'] > 0:
                    current['income_change_pct'] = ((current['total_credits'] - previous['total_credits']) / previous['total_credits']) * 100
                else:
                    current['income_change_pct'] = 0
            
            # Format the last month too
            if monthly_data:
                last_month = monthly_data[-1]
                last_month['total_debits_inr'] = format_inr(last_month['total_debits'])
                last_month['total_credits_inr'] = format_inr(last_month['total_credits'])
                last_month['avg_debit_inr'] = format_inr(last_month['avg_debit'])
                last_month['avg_credit_inr'] = format_inr(last_month['avg_credit'])
                last_month['upi_spending_inr'] = format_inr(last_month['upi_spending'])
                last_month['cash_withdrawals_inr'] = format_inr(last_month['cash_withdrawals'])
                last_month['grocery_spending_inr'] = format_inr(last_month['grocery_spending'])
                last_month['fuel_spending_inr'] = format_inr(last_month['fuel_spending'])
                last_month['net_amount_inr'] = format_inr(last_month['total_credits'] - last_month['total_debits'])
                last_month['spending_change_pct'] = 0
                last_month['income_change_pct'] = 0
        
        # Calculate overall statistics
        total_months = len(monthly_data)
        avg_monthly_spending = sum(month['total_debits'] for month in monthly_data) / total_months if total_months > 0 else 0
        avg_monthly_income = sum(month['total_credits'] for month in monthly_data) / total_months if total_months > 0 else 0
        
        response = {
            "analysis_period_months": months_back,
            "monthly_trends": monthly_data,
            "summary": {
                "total_months_analyzed": total_months,
                "avg_monthly_spending_inr": format_inr(avg_monthly_spending),
                "avg_monthly_income_inr": format_inr(avg_monthly_income),
                "avg_monthly_net_inr": format_inr(avg_monthly_income - avg_monthly_spending),
                "avg_monthly_spending_raw": avg_monthly_spending,
                "avg_monthly_income_raw": avg_monthly_income,
                "trend_direction": "increasing" if len(monthly_data) >= 2 and monthly_data[0]['total_debits'] > monthly_data[1]['total_debits'] else "decreasing" if len(monthly_data) >= 2 else "stable"
            },
            "performance": {
                "execution_time": result.execution_time,
                "cached": result.cached
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error analyzing spending trends: {str(e)}"})

@mcp.tool()
def get_performance_stats() -> str:
    """Get performance statistics for the optimized MCP server."""
    try:
        db_stats = db_manager.get_stats()
        
        response = {
            "database_performance": db_stats,
            "server_info": {
                "server_type": "optimized-financial-mcp",
                "features": [
                    "Connection pooling",
                    "Query result caching",
                    "Prepared statements",
                    "Batch processing",
                    "Optimized JSON serialization"
                ],
                "cache_enabled": True,
                "pool_enabled": True
            }
        }
        
        return serialize_result(response)
    except Exception as e:
        return serialize_result({"error": f"Error getting performance stats: {str(e)}"})

if __name__ == "__main__":
    import sys
    try:
        # Print startup messages to stderr to avoid interfering with JSON-RPC on stdout
        print("🚀 Starting Optimized Financial MCP Server...", file=sys.stderr)
        print(f"📊 Database: {db_manager.database_path}", file=sys.stderr)
        print(f"🔄 Connection pool size: {db_manager.connection_pool.max_connections}", file=sys.stderr)
        print(f"💾 Cache size: {db_manager.query_cache.max_size}", file=sys.stderr)
        print("✅ Optimized Financial MCP Server ready for connections!", file=sys.stderr)
        mcp.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Optimized Financial MCP Server...", file=sys.stderr)
        db_manager.close()
        print("✅ Server stopped gracefully", file=sys.stderr)
    except Exception as e:
        print(f"❌ Server error: {e}", file=sys.stderr)
        db_manager.close()
        exit(1)
