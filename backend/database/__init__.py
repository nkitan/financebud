"""
Database module for financial data management.
Provides high-performance database operations with connection pooling and caching.
"""

from .db import DatabaseManager, get_db_manager

__all__ = ['DatabaseManager', 'get_db_manager']
