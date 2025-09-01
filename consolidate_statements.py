#!/usr/bin/env python3
"""
Bank Statement Consolidator
===========================

This script consolidates bank statement transactions from multiple .xls files
(which are actually tab-separated text files) into a SQLite database for easy
querying by LLMs and other applications.

The script:
1. Reads all bank statement files in the Bank-Statements directory
2. Extracts account information and transactions
3. Stores everything in a normalized SQLite database
4. Provides utilities for querying the data

Author: Generated for FinanceBud
Date: August 2025
"""

import os
import sqlite3
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BankStatementConsolidator:
    def __init__(self, statements_dir: str, db_path: str = "financial_data.db"):
        """
        Initialize the consolidator.
        
        Args:
            statements_dir: Path to directory containing bank statement files
            db_path: Path to SQLite database file
        """
        self.statements_dir = statements_dir
        self.db_path = db_path
        self.conn = None
        
    def __enter__(self):
        """Context manager entry."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.conn.close()
            
    def create_tables(self):
        """Create the database schema."""
        cursor = self.conn.cursor()
        
        # Account information table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_number TEXT UNIQUE NOT NULL,
                account_name TEXT,
                account_description TEXT,
                branch TEXT,
                ifsc_code TEXT,
                micr_code TEXT,
                cif_number TEXT,
                ckycr_number TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Statements table (for tracking which files have been processed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statements (
                statement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER,
                file_name TEXT NOT NULL,
                start_date DATE,
                end_date DATE,
                opening_balance DECIMAL(15,2),
                closing_balance DECIMAL(15,2),
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES accounts (account_id)
            )
        """)
        
        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                statement_id INTEGER,
                transaction_date DATE NOT NULL,
                value_date DATE,
                description TEXT,
                transaction_type TEXT,
                debit_amount DECIMAL(15,2),
                credit_amount DECIMAL(15,2),
                balance DECIMAL(15,2),
                reference_number TEXT,
                beneficiary_name TEXT,
                upi_id TEXT,
                bank_code TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (statement_id) REFERENCES statements (statement_id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(debit_amount, credit_amount)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_beneficiary ON transactions(beneficiary_name)")
        
        self.conn.commit()
        logger.info("Database schema created successfully")
        
    def parse_account_info(self, lines: List[str]) -> Dict:
        """Parse account information from statement header."""
        account_info = {}
        
        for line in lines[:20]:  # Account info is typically in first 20 lines
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if 'Account Name' in key:
                    account_info['account_name'] = value
                elif 'Account Number' in key:
                    # Remove leading underscore if present
                    account_info['account_number'] = value.lstrip('_')
                elif 'Account Description' in key:
                    account_info['account_description'] = value
                elif 'Branch' in key:
                    account_info['branch'] = value
                elif 'IFS' in key and 'Code' in key:
                    account_info['ifsc_code'] = value.lstrip('_')
                elif 'MICR' in key and 'Code' in key:
                    account_info['micr_code'] = value.lstrip('_')
                elif 'CIF No' in key:
                    account_info['cif_number'] = value.lstrip('_')
                elif 'CKYCR' in key:
                    account_info['ckycr_number'] = value
                elif 'Balance on' in key:
                    # Extract opening balance
                    balance_match = re.search(r'[\d,]+\.?\d*', value)
                    if balance_match:
                        account_info['opening_balance'] = float(balance_match.group().replace(',', ''))
                elif 'Start Date' in key:
                    account_info['start_date'] = value
                elif 'End Date' in key:
                    account_info['end_date'] = value
                    
        return account_info
        
    def extract_transaction_details(self, description: str) -> Dict:
        """Extract structured information from transaction description."""
        details = {'raw_description': description}
        
        # Extract UPI transaction details - improved patterns
        # Pattern 1: TO TRANSFER-UPI/DR/refnum/name/bank/upiid/...
        upi_match = re.search(r'TRANSFER-UPI/(?:DR|CR)/(\d+)/([^/]+)/([^/]+)/([^/]+)/', description)
        if upi_match:
            details['reference_number'] = upi_match.group(1)
            details['beneficiary_name'] = upi_match.group(2).strip()
            details['bank_code'] = upi_match.group(3).strip()
            details['upi_id'] = upi_match.group(4).strip()
            details['transaction_type'] = 'UPI'
            return details
        
        # Pattern 2: Direct UPI/DR/refnum/name/bank/upiid/...
        upi_match2 = re.search(r'UPI/(?:DR|CR)/(\d+)/([^/]+)/([^/]+)/([^/]+)/', description)
        if upi_match2:
            details['reference_number'] = upi_match2.group(1)
            details['beneficiary_name'] = upi_match2.group(2).strip()
            details['bank_code'] = upi_match2.group(3).strip()
            details['upi_id'] = upi_match2.group(4).strip()
            details['transaction_type'] = 'UPI'
            return details
            
        # Pattern 3: UPI REVERSAL transactions
        if 'UPI' in description and ('REVERSAL' in description or 'REV/' in description):
            details['beneficiary_name'] = 'UPI Reversal'
            details['transaction_type'] = 'UPI'
            # Extract reference number if available
            rev_match = re.search(r'UPI/(?:REV/)?(\d+)', description)
            if rev_match:
                details['reference_number'] = rev_match.group(1)
            return details
        
        # Extract NEFT transaction details - improved patterns
        if 'NEFT' in description:
            # Pattern: TRANSFER-NEFT*BANKCODE*REFNUM*BENEFICIARY_NAME
            neft_match = re.search(r'NEFT\*([^*]+)\*([^*]+)\*([^*-]+)', description)
            if neft_match:
                details['bank_code'] = neft_match.group(1).strip()
                details['reference_number'] = neft_match.group(2).strip()
                details['beneficiary_name'] = neft_match.group(3).strip()
                details['transaction_type'] = 'NEFT'
                return details
        
        # Extract IMPS transaction details - improved patterns
        if 'IMPS' in description:
            # Pattern 1: TRANSFER-IMPS/refnum/details/beneficiary/...
            imps_match = re.search(r'IMPS/(\d+)/([^/]+)/([^/]+)', description)
            if imps_match:
                details['reference_number'] = imps_match.group(1)
                bank_info = imps_match.group(2).strip()
                details['beneficiary_name'] = imps_match.group(3).strip()
                details['transaction_type'] = 'IMPS'
                return details
            # Pattern 2: INB IMPS/refnum/bank-info/beneficiary/...
            imps_match2 = re.search(r'INB IMPS/(\d+)/([^/]+)/([^/]+)', description)
            if imps_match2:
                details['reference_number'] = imps_match2.group(1)
                details['beneficiary_name'] = imps_match2.group(3).strip()
                details['transaction_type'] = 'IMPS'
                return details
            # Pattern 3: INB IMPS with reference and account info
            imps_match3 = re.search(r'INB IMPS(\d+)/(\d+)/([^/]+)/([^-]+)', description)
            if imps_match3:
                details['reference_number'] = imps_match3.group(1)
                details['beneficiary_name'] = imps_match3.group(4).strip()
                details['transaction_type'] = 'IMPS'
                return details
            else:
                details['transaction_type'] = 'IMPS'
                return details
        
        # Extract debit card transaction details
        if 'by debit card' in description.lower():
            # Pattern: by debit card-OTHPOS/OTHPG refnum MERCHANT_NAME LOCATION
            card_match = re.search(r'by debit card-\w+\s*\d+\s+([^-]+?)(?:\s+[A-Z\s]+)?--', description, re.IGNORECASE)
            if card_match:
                details['beneficiary_name'] = card_match.group(1).strip()
                details['transaction_type'] = 'CARD'
                return details
            else:
                details['transaction_type'] = 'CARD'
                return details
        
        # Extract ATM transaction details
        if 'ATM' in description.upper():
            # Pattern: ATM WDL-ATM CASH refnum LOCATION
            atm_match = re.search(r'ATM\s+WDL-ATM\s+CASH\s+\d+\s+([^-]+)', description, re.IGNORECASE)
            if atm_match:
                details['beneficiary_name'] = f"ATM - {atm_match.group(1).strip()}"
                details['transaction_type'] = 'ATM'
                return details
            elif 'ATM DECLINE CHARGE' in description:
                details['beneficiary_name'] = 'ATM Decline Charge'
                details['transaction_type'] = 'ATM'
                return details
            else:
                details['transaction_type'] = 'ATM'
                return details
        
        # Extract interest credit details
        if 'CREDIT INTEREST' in description.upper():
            details['beneficiary_name'] = 'Bank Interest'
            details['transaction_type'] = 'INTEREST'
            return details
        
        # Handle INB (Internet Banking) transactions
        if 'TRANSFER-INB' in description:
            # Extract beneficiary name after INB
            inb_patterns = [
                r'TRANSFER-INB\s+([^-]+?)--',  # Basic pattern
                r'TRANSFER-INB\s+([^-]+?)\s+[A-Z]',  # With additional info
                r'TRANSFER-INB\s+(.+?)(?:\s+ICICI_CC|--)',  # Credit card payments
            ]
            
            for pattern in inb_patterns:
                inb_match = re.search(pattern, description)
                if inb_match:
                    beneficiary = inb_match.group(1).strip()
                    if beneficiary:
                        details['beneficiary_name'] = beneficiary
                        details['transaction_type'] = 'INB'
                        return details
            
            # Fallback for INB transactions
            details['transaction_type'] = 'INB'
            return details
        
        # Handle decline charges
        if 'DECLINE CHARGE' in description.upper():
            if 'POS' in description:
                details['beneficiary_name'] = 'POS Decline Charge'
            elif 'ATM' in description:
                details['beneficiary_name'] = 'ATM Decline Charge'
            else:
                details['beneficiary_name'] = 'Decline Charge'
            details['transaction_type'] = 'FEE'
            return details
        
        # Handle miscellaneous service charges
        if 'COMM - OTHER MISC. SERVICES' in description:
            details['beneficiary_name'] = 'Bank Service Charge'
            details['transaction_type'] = 'FEE'
            return details
        
        # Handle remaining UPI transactions that might not match above patterns
        if 'UPI' in description.upper():
            details['transaction_type'] = 'UPI'
            # Try to extract any beneficiary name after UPI markers
            upi_fallback = re.search(r'UPI[^/]*/[^/]*/[^/]*/([^/]+)/', description)
            if upi_fallback:
                details['beneficiary_name'] = upi_fallback.group(1).strip()
            return details
        
        # Handle card transactions (alternative pattern)
        if 'CARD' in description.upper():
            details['transaction_type'] = 'CARD'
            return details
        
        # Handle special transaction references
        if 'SBIYA' in description and 'Transfer to' in description:
            transfer_match = re.search(r'Transfer to\s+(.+?)(?:--|\s*$)', description)
            if transfer_match:
                details['beneficiary_name'] = transfer_match.group(1).strip()
                details['transaction_type'] = 'TRANSFER'
                return details
        
        # Default case
        details['transaction_type'] = 'OTHER'
        return details
        
    def parse_transaction_line(self, line: str) -> Optional[Dict]:
        """Parse a single transaction line."""
        parts = line.strip().split('\t')
        
        # Filter out empty parts and clean up
        parts = [part.strip() for part in parts if part.strip()]
        
        if len(parts) < 6:
            return None
            
        try:
            # Extract date (first part)
            date_str = parts[0].strip()
            transaction_date = datetime.strptime(date_str, '%d %b %Y').date()
            
            # Extract value date (second part)
            value_date_str = parts[1].strip()
            value_date = datetime.strptime(value_date_str, '%d %b %Y').date()
            
            # Extract description (third part)
            description = parts[2].strip()
            
            # Extract transaction type (fourth part)
            transaction_type_raw = parts[3].strip()
            
            # Extract amounts and balance
            # The last part should be balance
            balance_str = parts[-1].replace(',', '')
            balance = float(balance_str)
            
            # Determine debit/credit amounts
            debit_amount = None
            credit_amount = None
            
            if len(parts) >= 7:
                # Format: date, value_date, description, type, debit, credit, balance
                debit_str = parts[4].replace(',', '').strip()
                credit_str = parts[5].replace(',', '').strip()
                
                if debit_str and debit_str != '':
                    debit_amount = float(debit_str)
                if credit_str and credit_str != '':
                    credit_amount = float(credit_str)
            else:
                # Format: date, value_date, description, type, amount, balance
                amount_str = parts[4].replace(',', '').strip()
                if amount_str:
                    amount = float(amount_str)
                    # Determine if it's debit or credit based on description
                    if 'TO TRANSFER' in description or 'DR/' in description:
                        debit_amount = amount
                    else:
                        credit_amount = amount
            
            # Extract additional details from description
            transaction_details = self.extract_transaction_details(description)
            
            return {
                'transaction_date': transaction_date,
                'value_date': value_date,
                'description': description,
                'transaction_type': transaction_details.get('transaction_type', 'OTHER'),
                'debit_amount': debit_amount,
                'credit_amount': credit_amount,
                'balance': balance,
                'reference_number': transaction_details.get('reference_number'),
                'beneficiary_name': transaction_details.get('beneficiary_name'),
                'upi_id': transaction_details.get('upi_id'),
                'bank_code': transaction_details.get('bank_code')
            }
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing transaction line: {line[:100]}... Error: {e}")
            return None
            
    def process_statement_file(self, file_path: str) -> bool:
        """Process a single statement file."""
        try:
            logger.info(f"Processing file: {os.path.basename(file_path)}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse account information
            account_info = self.parse_account_info(lines)
            
            if not account_info.get('account_number'):
                logger.error(f"Could not extract account number from {file_path}")
                return False
            
            # Insert or get account
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO accounts 
                (account_number, account_name, account_description, branch, ifsc_code, micr_code, cif_number, ckycr_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account_info.get('account_number'),
                account_info.get('account_name'),
                account_info.get('account_description'),
                account_info.get('branch'),
                account_info.get('ifsc_code'),
                account_info.get('micr_code'),
                account_info.get('cif_number'),
                account_info.get('ckycr_number')
            ))
            
            # Get account ID
            cursor.execute("SELECT account_id FROM accounts WHERE account_number = ?", 
                         (account_info['account_number'],))
            account_id = cursor.fetchone()[0]
            
            # Insert statement record
            cursor.execute("""
                INSERT INTO statements 
                (account_id, file_name, start_date, end_date, opening_balance)
                VALUES (?, ?, ?, ?, ?)
            """, (
                account_id,
                os.path.basename(file_path),
                account_info.get('start_date'),
                account_info.get('end_date'),
                account_info.get('opening_balance')
            ))
            
            statement_id = cursor.lastrowid
            
            # Parse transactions
            transactions = []
            for line in lines:
                # Skip lines that don't start with a date
                if not re.match(r'^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', line.strip()):
                    continue
                    
                transaction = self.parse_transaction_line(line)
                if transaction:
                    transaction['statement_id'] = statement_id
                    transactions.append(transaction)
            
            # Insert transactions
            if transactions:
                cursor.executemany("""
                    INSERT INTO transactions 
                    (statement_id, transaction_date, value_date, description, transaction_type,
                     debit_amount, credit_amount, balance, reference_number, beneficiary_name, upi_id, bank_code)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (t['statement_id'], t['transaction_date'], t['value_date'], t['description'],
                     t['transaction_type'], t['debit_amount'], t['credit_amount'], t['balance'],
                     t['reference_number'], t['beneficiary_name'], t['upi_id'], t['bank_code'])
                    for t in transactions
                ])
                
                # Update closing balance in statement
                if transactions:
                    closing_balance = transactions[-1]['balance']
                    cursor.execute("""
                        UPDATE statements SET closing_balance = ? WHERE statement_id = ?
                    """, (closing_balance, statement_id))
                
                logger.info(f"Processed {len(transactions)} transactions from {os.path.basename(file_path)}")
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False
    
    def consolidate_all_statements(self):
        """Process all statement files in the directory."""
        if not os.path.exists(self.statements_dir):
            logger.error(f"Statements directory not found: {self.statements_dir}")
            return False
            
        # Get all .xls files
        files = [f for f in os.listdir(self.statements_dir) if f.endswith('.xls')]
        
        if not files:
            logger.error("No .xls files found in statements directory")
            return False
            
        logger.info(f"Found {len(files)} statement files to process")
        
        # Create tables
        self.create_tables()
        
        # Process each file
        successful = 0
        for file_name in sorted(files):
            file_path = os.path.join(self.statements_dir, file_name)
            if self.process_statement_file(file_path):
                successful += 1
                
        logger.info(f"Successfully processed {successful}/{len(files)} files")
        return successful == len(files)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from the database."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total transactions
        cursor.execute("SELECT COUNT(*) FROM transactions")
        stats['total_transactions'] = cursor.fetchone()[0]
        
        # Date range
        cursor.execute("SELECT MIN(transaction_date), MAX(transaction_date) FROM transactions")
        date_range = cursor.fetchone()
        stats['date_range'] = {'start': date_range[0], 'end': date_range[1]}
        
        # Total amounts
        cursor.execute("SELECT SUM(debit_amount), SUM(credit_amount) FROM transactions")
        amounts = cursor.fetchone()
        stats['total_debits'] = amounts[0] or 0
        stats['total_credits'] = amounts[1] or 0
        
        # Transaction types
        cursor.execute("""
            SELECT transaction_type, COUNT(*), SUM(COALESCE(debit_amount, 0)), SUM(COALESCE(credit_amount, 0))
            FROM transactions 
            GROUP BY transaction_type 
            ORDER BY COUNT(*) DESC
        """)
        stats['transaction_types'] = {
            row[0]: {'count': row[1], 'total_debits': row[2], 'total_credits': row[3]}
            for row in cursor.fetchall()
        }
        
        # Top beneficiaries by transaction count
        cursor.execute("""
            SELECT beneficiary_name, COUNT(*), SUM(COALESCE(debit_amount, 0))
            FROM transactions 
            WHERE beneficiary_name IS NOT NULL
            GROUP BY beneficiary_name 
            ORDER BY COUNT(*) DESC 
            LIMIT 10
        """)
        stats['top_beneficiaries'] = [
            {'name': row[0], 'count': row[1], 'total_amount': row[2]}
            for row in cursor.fetchall()
        ]
        
        return stats

def main():
    """Main function to run the consolidation."""
    statements_dir = os.getenv("STATEMENTS_DIR", "Bank-Statements")
    db_path = os.getenv("DATABASE_PATH", "financial_data.db")
    
    with BankStatementConsolidator(statements_dir, db_path) as consolidator:
        # Consolidate all statements
        success = consolidator.consolidate_all_statements()
        
        if success:
            # Print summary
            stats = consolidator.get_summary_stats()
            print("\n" + "="*50)
            print("CONSOLIDATION COMPLETE!")
            print("="*50)
            print(f"Total transactions: {stats['total_transactions']:,}")
            print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            print(f"Total debits: ₹{stats['total_debits']:,.2f}")
            print(f"Total credits: ₹{stats['total_credits']:,.2f}")
            print(f"Net amount: ₹{stats['total_credits'] - stats['total_debits']:,.2f}")
            
            print("\nTransaction Types:")
            for trans_type, data in stats['transaction_types'].items():
                print(f"  {trans_type}: {data['count']:,} transactions")
            
            print(f"\nTop 10 Beneficiaries:")
            for beneficiary in stats['top_beneficiaries']:
                print(f"  {beneficiary['name'][:30]:<30} {beneficiary['count']:>3} transactions ₹{beneficiary['total_amount']:>10,.2f}")
            
            print(f"\nDatabase saved to: {db_path}")
            print("You can now query the database using SQL or the provided utility functions.")
        else:
            print("Consolidation failed. Check the logs for errors.")

if __name__ == "__main__":
    main()
