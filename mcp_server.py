#!/usr/bin/env python3
"""
FastMCP server for querying consolidated financial database.
Provides standardized tools for LLMs to access bank statement data.
All amounts are in INR (Indian Rupees).
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("Financial Data Server")

# Database path
DB_PATH = "/home/notroot/Work/financebud/financial_data.db"

def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)

def execute_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a SQL query and return results as list of dictionaries."""
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

def format_inr(amount: float) -> str:
    """Format amount as INR currency."""
    if amount is None:
        return "₹0.00"
    return f"₹{amount:,.2f}"

@mcp.tool()
def get_account_summary() -> str:
    """Get a summary of all bank accounts and their current status. All amounts in INR."""
    try:
        total_transactions = execute_query("SELECT COUNT(*) as count FROM transactions")[0]['count']
        date_range = execute_query("SELECT MIN(transaction_date) as earliest_date, MAX(transaction_date) as latest_date FROM transactions")[0]
        latest_balance = execute_query("SELECT balance FROM transactions ORDER BY transaction_date DESC, transaction_id DESC LIMIT 1")[0]['balance']
        totals = execute_query("SELECT SUM(debit_amount) as total_debits, SUM(credit_amount) as total_credits FROM transactions WHERE debit_amount IS NOT NULL OR credit_amount IS NOT NULL")[0]
        
        summary = {
            "total_transactions": total_transactions,
            "date_range": {"earliest": date_range['earliest_date'], "latest": date_range['latest_date']},
            "current_balance_inr": format_inr(latest_balance),
            "current_balance_raw": latest_balance,
            "total_debits_inr": format_inr(totals['total_debits'] or 0),
            "total_debits_raw": totals['total_debits'] or 0,
            "total_credits_inr": format_inr(totals['total_credits'] or 0),
            "total_credits_raw": totals['total_credits'] or 0,
            "net_change_inr": format_inr((totals['total_credits'] or 0) - (totals['total_debits'] or 0)),
            "net_change_raw": (totals['total_credits'] or 0) - (totals['total_debits'] or 0),
            "currency": "INR"
        }
        return json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error getting account summary: {str(e)}"

@mcp.tool()
def search_transactions(description_pattern: str, limit: int = 50) -> str:
    """Search for transactions matching a description pattern. All amounts in INR."""
    try:
        query = "SELECT transaction_date, description, debit_amount, credit_amount, balance FROM transactions WHERE description LIKE ? COLLATE NOCASE ORDER BY transaction_date DESC LIMIT ?"
        results = execute_query(query, (f"%{description_pattern}%", limit))
        
        if not results:
            return f"No transactions found matching '{description_pattern}'"
        
        # Add formatted INR amounts
        for result in results:
            result['debit_amount_inr'] = format_inr(result['debit_amount']) if result['debit_amount'] else None
            result['credit_amount_inr'] = format_inr(result['credit_amount']) if result['credit_amount'] else None
            result['balance_inr'] = format_inr(result['balance'])
        
        return json.dumps({
            "search_pattern": description_pattern,
            "total_results": len(results),
            "currency": "INR",
            "transactions": results
        }, indent=2)
    except Exception as e:
        return f"Error searching transactions: {str(e)}"

@mcp.tool()
def get_transactions_by_date_range(start_date: str, end_date: str, limit: int = 100) -> str:
    """Get transactions within a specific date range (YYYY-MM-DD format). All amounts in INR."""
    try:
        query = "SELECT transaction_date, description, debit_amount, credit_amount, balance FROM transactions WHERE transaction_date BETWEEN ? AND ? ORDER BY transaction_date DESC LIMIT ?"
        results = execute_query(query, (start_date, end_date, limit))
        
        if results:
            total_debits = sum(float(t['debit_amount'] or 0) for t in results)
            total_credits = sum(float(t['credit_amount'] or 0) for t in results)
            net_change = total_credits - total_debits
            
            # Add formatted INR amounts to each transaction
            for result in results:
                result['debit_amount_inr'] = format_inr(result['debit_amount']) if result['debit_amount'] else None
                result['credit_amount_inr'] = format_inr(result['credit_amount']) if result['credit_amount'] else None
                result['balance_inr'] = format_inr(result['balance'])
        else:
            total_debits = total_credits = net_change = 0
        
        return json.dumps({
            "date_range": {"start": start_date, "end": end_date},
            "total_transactions": len(results),
            "currency": "INR",
            "summary": {
                "total_debits_inr": format_inr(total_debits),
                "total_debits_raw": total_debits,
                "total_credits_inr": format_inr(total_credits),
                "total_credits_raw": total_credits,
                "net_change_inr": format_inr(net_change),
                "net_change_raw": net_change
            },
            "transactions": results
        }, indent=2)
    except Exception as e:
        return f"Error getting transactions by date range: {str(e)}"

@mcp.tool()
def get_monthly_summary(year: int, month: int) -> str:
    """Get a monthly summary of transactions and spending patterns. All amounts in INR."""
    try:
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month + 1:02d}-01" if month < 12 else f"{year + 1}-01-01"
        
        query = "SELECT transaction_date, description, debit_amount, credit_amount, balance, transaction_type FROM transactions WHERE transaction_date >= ? AND transaction_date < ? ORDER BY transaction_date ASC"
        transactions = execute_query(query, (start_date, end_date))
        
        if not transactions:
            return f"No transactions found for {year}-{month:02d}"
        
        total_debits = sum(float(t['debit_amount'] or 0) for t in transactions)
        total_credits = sum(float(t['credit_amount'] or 0) for t in transactions)
        net_change = total_credits - total_debits
        
        start_balance = transactions[0]['balance'] - (float(transactions[0]['credit_amount'] or 0) - float(transactions[0]['debit_amount'] or 0))
        end_balance = transactions[-1]['balance']
        
        # Analyze spending categories based on UPI transactions and descriptions
        categories = {}
        for t in transactions:
            desc = t['description'].lower()
            if t['debit_amount']:
                amount = float(t['debit_amount'])
                if 'upi' in desc and any(word in desc for word in ['grocery', 'supermarket', 'swiggy', 'zomato', 'food']):
                    categories['Food & Groceries'] = categories.get('Food & Groceries', 0) + amount
                elif 'upi' in desc and any(word in desc for word in ['petrol', 'fuel', 'gas']):
                    categories['Transportation'] = categories.get('Transportation', 0) + amount
                elif 'upi' in desc and any(word in desc for word in ['electric', 'water', 'utility']):
                    categories['Utilities'] = categories.get('Utilities', 0) + amount
                elif 'upi' in desc:
                    categories['UPI Payments'] = categories.get('UPI Payments', 0) + amount
                elif 'atm' in desc or 'withdrawal' in desc:
                    categories['ATM/Cash'] = categories.get('ATM/Cash', 0) + amount
                else:
                    categories['Other'] = categories.get('Other', 0) + amount
        
        # Format categories with INR
        formatted_categories = {k: format_inr(v) for k, v in categories.items()}
        
        # Add formatted amounts to sample transactions
        sample_transactions = transactions[:20]
        for t in sample_transactions:
            t['debit_amount_inr'] = format_inr(t['debit_amount']) if t['debit_amount'] else None
            t['credit_amount_inr'] = format_inr(t['credit_amount']) if t['credit_amount'] else None
            t['balance_inr'] = format_inr(t['balance'])
        
        return json.dumps({
            "month": f"{year}-{month:02d}",
            "transaction_count": len(transactions),
            "currency": "INR",
            "summary": {
                "starting_balance_inr": format_inr(start_balance),
                "starting_balance_raw": start_balance,
                "ending_balance_inr": format_inr(end_balance),
                "ending_balance_raw": end_balance,
                "total_debits_inr": format_inr(total_debits),
                "total_debits_raw": total_debits,
                "total_credits_inr": format_inr(total_credits),
                "total_credits_raw": total_credits,
                "net_change_inr": format_inr(net_change),
                "net_change_raw": net_change
            },
            "spending_categories": formatted_categories,
            "spending_categories_raw": categories,
            "sample_transactions": sample_transactions
        }, indent=2)
    except Exception as e:
        return f"Error getting monthly summary: {str(e)}"

@mcp.tool()
def find_recurring_payments(min_occurrences: int = 3) -> str:
    """Find recurring payments or deposits based on beneficiary names and amounts. All amounts in INR."""
    try:
        query = """
            SELECT 
                beneficiary_name,
                debit_amount,
                credit_amount,
                COUNT(*) as occurrence_count,
                AVG(COALESCE(debit_amount, 0)) as avg_debit,
                AVG(COALESCE(credit_amount, 0)) as avg_credit,
                MIN(transaction_date) as first_occurrence,
                MAX(transaction_date) as last_occurrence,
                transaction_type,
                CASE 
                    WHEN COUNT(*) > 1 THEN 
                        ROUND(
                            (julianday(MAX(transaction_date)) - julianday(MIN(transaction_date))) / 
                            (COUNT(*) - 1), 1
                        )
                    ELSE NULL
                END as avg_days_between
            FROM transactions
            WHERE beneficiary_name IS NOT NULL 
                AND (debit_amount > 0 OR credit_amount > 0)
            GROUP BY beneficiary_name, COALESCE(debit_amount, 0), COALESCE(credit_amount, 0)
            HAVING COUNT(*) >= ?
            ORDER BY occurrence_count DESC, beneficiary_name
        """
        results = execute_query(query, (min_occurrences,))
        
        # Add formatted INR amounts and additional analysis
        for result in results:
            result['avg_debit_inr'] = format_inr(result['avg_debit'])
            result['avg_credit_inr'] = format_inr(result['avg_credit'])
            if result['debit_amount']:
                result['debit_amount_inr'] = format_inr(result['debit_amount'])
            if result['credit_amount']:
                result['credit_amount_inr'] = format_inr(result['credit_amount'])
            
            # Calculate total transaction volume
            total_volume = (result['debit_amount'] or 0) * result['occurrence_count']
            result['total_volume_inr'] = format_inr(total_volume)
            result['total_volume_raw'] = total_volume
            
            # Determine payment frequency category
            if result['avg_days_between']:
                if result['avg_days_between'] <= 7:
                    result['frequency_category'] = 'Weekly'
                elif result['avg_days_between'] <= 31:
                    result['frequency_category'] = 'Monthly'
                elif result['avg_days_between'] <= 92:
                    result['frequency_category'] = 'Quarterly'
                else:
                    result['frequency_category'] = 'Irregular'
            else:
                result['frequency_category'] = 'Single Amount'
        
        # Group by beneficiary for summary (combining different amounts for same beneficiary)
        beneficiary_summary = {}
        for result in results:
            name = result['beneficiary_name']
            if name not in beneficiary_summary:
                beneficiary_summary[name] = {
                    'beneficiary_name': name,
                    'total_transactions': 0,
                    'total_amount_spent': 0,
                    'transaction_types': set(),
                    'amount_patterns': []
                }
            
            beneficiary_summary[name]['total_transactions'] += result['occurrence_count']
            beneficiary_summary[name]['total_amount_spent'] += result['total_volume_raw']
            beneficiary_summary[name]['transaction_types'].add(result['transaction_type'])
            beneficiary_summary[name]['amount_patterns'].append({
                'amount': result['debit_amount'] or result['credit_amount'],
                'count': result['occurrence_count'],
                'avg_days_between': result['avg_days_between']
            })
        
        # Convert sets to lists and format amounts for JSON serialization
        beneficiary_list = []
        for name, data in sorted(beneficiary_summary.items(), 
                                key=lambda x: x[1]['total_transactions'], reverse=True):
            data['transaction_types'] = list(data['transaction_types'])
            data['total_amount_spent_inr'] = format_inr(data['total_amount_spent'])
            beneficiary_list.append(data)
        
        return json.dumps({
            "recurring_payments_by_amount": results[:20],  # Top 20 specific amount patterns
            "recurring_beneficiaries_summary": beneficiary_list[:15],  # Top 15 beneficiaries
            "analysis_criteria": f"Minimum {min_occurrences} occurrences",
            "total_patterns_found": len(results),
            "total_beneficiaries": len(beneficiary_list),
            "currency": "INR"
        }, indent=2)
    except Exception as e:
        return f"Error finding recurring payments: {str(e)}"

@mcp.tool()
def analyze_spending_trends(months_back: int = 6) -> str:
    """Analyze spending trends over the last N months. All amounts in INR."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        query = """
            SELECT 
                strftime('%Y-%m', transaction_date) as month,
                SUM(debit_amount) as total_debits,
                SUM(credit_amount) as total_credits,
                COUNT(*) as transaction_count
            FROM transactions
            WHERE transaction_date >= ? AND transaction_date <= ?
            GROUP BY strftime('%Y-%m', transaction_date)
            ORDER BY month
        """
        results = execute_query(query, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        
        # Add formatted amounts
        for result in results:
            result['total_debits_inr'] = format_inr(result['total_debits'] or 0)
            result['total_credits_inr'] = format_inr(result['total_credits'] or 0)
        
        if len(results) > 1:
            avg_spending = sum(float(r['total_debits'] or 0) for r in results) / len(results)
            recent_spending = float(results[-1]['total_debits'] or 0) if results else 0
            trend = "increasing" if recent_spending > avg_spending else "decreasing"
        else:
            avg_spending = recent_spending = 0
            trend = "insufficient data"
        
        return json.dumps({
            "analysis_period": f"{months_back} months",
            "currency": "INR",
            "monthly_data": results,
            "trends": {
                "average_monthly_spending_inr": format_inr(avg_spending),
                "average_monthly_spending_raw": avg_spending,
                "recent_monthly_spending_inr": format_inr(recent_spending),
                "recent_monthly_spending_raw": recent_spending,
                "trend_direction": trend
            }
        }, indent=2)
    except Exception as e:
        return f"Error analyzing spending trends: {str(e)}"

@mcp.tool()
def execute_custom_query(sql_query: str) -> str:
    """Execute a custom SQL query (SELECT only for security). Results show amounts in INR."""
    try:
        if not sql_query.strip().upper().startswith('SELECT'):
            return "Error: Only SELECT queries are allowed for security reasons"
        
        results = execute_query(sql_query)
        
        # Add INR formatting to any amount columns
        for result in results:
            # Create a list of keys to avoid "dictionary changed size during iteration"
            original_keys = list(result.keys())
            for key in original_keys:
                value = result[key]
                if 'amount' in key.lower() or 'balance' in key.lower():
                    if isinstance(value, (int, float)) and value is not None:
                        result[f"{key}_inr"] = format_inr(value)
        
        return json.dumps({
            "query": sql_query,
            "result_count": len(results),
            "currency": "INR",
            "results": results[:100]
        }, indent=2)
    except Exception as e:
        return f"Error executing query: {str(e)}"

@mcp.tool()
def get_database_schema() -> str:
    """Get the database schema information including table structure and sample data."""
    try:
        schema = execute_query("PRAGMA table_info(transactions)")
        sample_data = execute_query("SELECT * FROM transactions LIMIT 5")
        
        # Add INR formatting to sample data
        for row in sample_data:
            if row['debit_amount']:
                row['debit_amount_inr'] = format_inr(row['debit_amount'])
            if row['credit_amount']:
                row['credit_amount_inr'] = format_inr(row['credit_amount'])
            if row['balance']:
                row['balance_inr'] = format_inr(row['balance'])
        
        return json.dumps({
            "table_name": "transactions",
            "currency": "INR (Indian Rupees)",
            "schema": schema,
            "sample_data": sample_data,
            "description": "Table contains Indian bank transaction data with columns: transaction_id, statement_id, transaction_date, value_date, description, transaction_type, debit_amount (INR), credit_amount (INR), balance (INR), reference_number, beneficiary_name, upi_id, bank_code, created_at"
        }, indent=2)
    except Exception as e:
        return f"Error getting database schema: {str(e)}"

@mcp.tool()
def get_database_columns() -> str:
    """Get detailed information about all database columns including types, constraints, and sample values."""
    try:
        # Get table schema information
        schema_query = "PRAGMA table_info(transactions)"
        schema_info = execute_query(schema_query)
        
        # Get total count first
        total_count = execute_query("SELECT COUNT(*) as total FROM transactions")[0]['total']
        
        # Get some statistics about each column
        column_stats = {}
        for column in schema_info:
            col_name = column['name']
            
            # Get non-null count
            non_null_query = f"SELECT COUNT({col_name}) as non_null_count FROM transactions WHERE {col_name} IS NOT NULL"
            non_null_count = execute_query(non_null_query)[0]['non_null_count']
            
            # Get sample values (limit to avoid too much data)
            if col_name in ['debit_amount', 'credit_amount', 'balance']:
                sample_query = f"SELECT DISTINCT {col_name} FROM transactions WHERE {col_name} IS NOT NULL ORDER BY {col_name} DESC LIMIT 5"
            else:
                sample_query = f"SELECT DISTINCT {col_name} FROM transactions WHERE {col_name} IS NOT NULL LIMIT 5"
            
            sample_values = execute_query(sample_query)
            
            column_stats[col_name] = {
                "type": column['type'],
                "not_null": bool(column['notnull']),
                "primary_key": bool(column['pk']),
                "default_value": column['dflt_value'],
                "non_null_count": non_null_count,
                "total_count": total_count,
                "completeness_percentage": round((non_null_count / total_count * 100), 2) if total_count > 0 else 0,
                "sample_values": [row[col_name] for row in sample_values[:5]]
            }
            
            # Add formatted INR values for amount columns
            if col_name in ['debit_amount', 'credit_amount', 'balance']:
                column_stats[col_name]["sample_values_inr"] = [
                    format_inr(val) for val in column_stats[col_name]["sample_values"] if val is not None
                ]
        
        return json.dumps({
            "table_name": "transactions",
            "currency": "INR (Indian Rupees)",
            "total_records": total_count,
            "column_details": column_stats,
            "description": "Detailed information about each column including data types, completeness, and sample values"
        }, indent=2)
    except Exception as e:
        return f"Error getting database columns: {str(e)}"

@mcp.tool()
def get_upi_transaction_analysis() -> str:
    """Analyze UPI transactions specifically for Indian banking patterns. All amounts in INR."""
    try:
        query = """
            SELECT 
                COUNT(*) as total_upi_transactions,
                SUM(debit_amount) as total_upi_debits,
                SUM(credit_amount) as total_upi_credits,
                AVG(debit_amount) as avg_upi_debit,
                MIN(transaction_date) as first_upi_date,
                MAX(transaction_date) as last_upi_date
            FROM transactions 
            WHERE transaction_type = 'UPI' OR description LIKE '%UPI%'
        """
        summary = execute_query(query)[0]
        
        # Get top UPI beneficiaries
        beneficiary_query = """
            SELECT 
                beneficiary_name,
                COUNT(*) as transaction_count,
                SUM(debit_amount) as total_amount
            FROM transactions 
            WHERE transaction_type = 'UPI' AND beneficiary_name IS NOT NULL
            GROUP BY beneficiary_name
            ORDER BY total_amount DESC
            LIMIT 10
        """
        top_beneficiaries = execute_query(beneficiary_query)
        
        # Format amounts
        summary['total_upi_debits_inr'] = format_inr(summary['total_upi_debits'] or 0)
        summary['total_upi_credits_inr'] = format_inr(summary['total_upi_credits'] or 0)
        summary['avg_upi_debit_inr'] = format_inr(summary['avg_upi_debit'] or 0)
        
        for beneficiary in top_beneficiaries:
            beneficiary['total_amount_inr'] = format_inr(beneficiary['total_amount'])
        
        return json.dumps({
            "upi_analysis": summary,
            "top_beneficiaries": top_beneficiaries,
            "currency": "INR",
            "note": "Analysis specific to UPI transactions in Indian banking system"
        }, indent=2)
    except Exception as e:
        return f"Error analyzing UPI transactions: {str(e)}"

if __name__ == "__main__":
    mcp.run()
