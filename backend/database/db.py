"""
Database Connection Manager
===========================

High-performance connection pooling and caching for SQLite database operations
to provide fast and efficient database access for MCP server operations.

Key features:
- Connection pooling for concurrent access
- Prepared statement caching
- Result caching for frequent queries
- Transaction batching capabilities
- Connection reuse and optimization
"""

import sqlite3
import threading
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import queue
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of a database query."""
    data: List[Dict[str, Any]]
    execution_time: float
    cached: bool = False

class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_connections = 0
        
        # Initialize pool with one connection
        self._create_connection()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new optimized SQLite connection."""
        conn = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            timeout=30.0
        )
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        
        # Optimize SQLite settings for performance
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        # Set row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        self._created_connections += 1
        logger.debug(f"Created connection {self._created_connections}/{self.max_connections}")
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        try:
            # Try to get from pool with timeout
            try:
                conn = self._pool.get(timeout=5.0)
            except queue.Empty:
                # Create new connection if pool is empty and we haven't hit limit
                with self._lock:
                    if self._created_connections < self.max_connections:
                        conn = self._create_connection()
                    else:
                        # Wait longer for a connection to become available
                        conn = self._pool.get(timeout=10.0)
            
            yield conn
            
        finally:
            if conn:
                # Return connection to pool
                try:
                    self._pool.put_nowait(conn)
                except queue.Full:
                    # Pool is full, close this connection
                    conn.close()
                    with self._lock:
                        self._created_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool."""
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        with self._lock:
            self._created_connections = 0

class QueryCache:
    """LRU cache for database query results with TTL."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._expiry_times = {}
        self._lock = threading.RLock()
    
    def _get_cache_key(self, query: str, params: Tuple) -> str:
        """Generate cache key for query and parameters."""
        combined = f"{query}:{json.dumps(params)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, params: Tuple) -> Optional[List[Dict[str, Any]]]:
        """Get cached result if available and not expired."""
        cache_key = self._get_cache_key(query, params)
        
        with self._lock:
            # Check if key exists and not expired
            if cache_key in self._cache:
                current_time = time.time()
                
                if current_time < self._expiry_times[cache_key]:
                    # Update access time for LRU
                    self._access_times[cache_key] = current_time
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return self._cache[cache_key]
                else:
                    # Expired, remove from cache
                    self._remove_key(cache_key)
            
            return None
    
    def put(self, query: str, params: Tuple, result: List[Dict[str, Any]], ttl: Optional[int] = None):
        """Store result in cache with TTL."""
        cache_key = self._get_cache_key(query, params)
        current_time = time.time()
        expiry_time = current_time + (ttl or self.default_ttl)
        
        with self._lock:
            # Check if we need to evict old entries
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_lru()
            
            self._cache[cache_key] = result
            self._access_times[cache_key] = current_time
            self._expiry_times[cache_key] = expiry_time
            
            logger.debug(f"Cached result for query: {query[:50]}...")
    
    def _remove_key(self, cache_key: str):
        """Remove a key from all cache structures."""
        self._cache.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
        self._expiry_times.pop(cache_key, None)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        # Find the key with the oldest access time
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove_key(oldest_key)
        logger.debug("Evicted LRU cache entry")
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._expiry_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for expiry_time in self._expiry_times.values()
                if current_time >= expiry_time
            )
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "expired_entries": expired_count,
                "fill_ratio": len(self._cache) / self.max_size
            }

class DatabaseManager:
    """High-performance database manager with connection pooling and caching."""
    
    def __init__(self, database_path: str, pool_size: int = 10, cache_size: int = 1000):
        self.database_path = database_path
        self.connection_pool = ConnectionPool(database_path, pool_size)
        self.query_cache = QueryCache(cache_size)
        self.prepared_statements = {}
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_query_time": 0.0,
            "start_time": time.time()
        }
        self._stats_lock = threading.Lock()
        
        logger.info(f"Initialized OptimizedDatabaseManager for {database_path}")
    
    def execute_query(self, query: str, params: Tuple = (), cache_ttl: Optional[int] = None) -> QueryResult:
        """Execute a query with caching and connection pooling."""
        start_time = time.time()
        
        # Check cache first (only for SELECT queries)
        if query.strip().upper().startswith('SELECT'):
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                execution_time = time.time() - start_time
                self._update_stats(execution_time, True)
                return QueryResult(cached_result, execution_time, cached=True)
        
        # Execute query
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(query, params)
                result = [dict(row) for row in cursor.fetchall()]
                
                execution_time = time.time() - start_time
                
                # Cache SELECT query results
                if query.strip().upper().startswith('SELECT') and cache_ttl != 0:
                    self.query_cache.put(query, params, result, cache_ttl)
                
                self._update_stats(execution_time, False)
                return QueryResult(result, execution_time, cached=False)
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed: {e}")
            self._update_stats(execution_time, False)
            raise
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> List[QueryResult]:
        """Execute the same query with multiple parameter sets."""
        start_time = time.time()
        results = []
        
        try:
            with self.connection_pool.get_connection() as conn:
                for params in params_list:
                    cursor = conn.execute(query, params)
                    result = [dict(row) for row in cursor.fetchall()]
                    results.append(QueryResult(result, 0.0, cached=False))
                
                execution_time = time.time() - start_time
                
                # Update stats for batch operation
                with self._stats_lock:
                    self._stats["total_queries"] += len(params_list)
                    self._stats["cache_misses"] += len(params_list)
                    
                    # Update average query time
                    avg_time_per_query = execution_time / len(params_list)
                    self._update_avg_time(avg_time_per_query)
                
                return results
                
        except Exception as e:
            logger.error(f"Batch query execution failed: {e}")
            raise
    
    def execute_transaction(self, queries_and_params: List[Tuple[str, Tuple]]) -> List[QueryResult]:
        """Execute multiple queries in a transaction."""
        start_time = time.time()
        results = []
        
        try:
            with self.connection_pool.get_connection() as conn:
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    for query, params in queries_and_params:
                        cursor = conn.execute(query, params)
                        result = [dict(row) for row in cursor.fetchall()]
                        results.append(QueryResult(result, 0.0, cached=False))
                    
                    conn.execute("COMMIT")
                    
                except Exception as e:
                    conn.execute("ROLLBACK")
                    raise
                
                execution_time = time.time() - start_time
                
                # Update stats for transaction
                with self._stats_lock:
                    self._stats["total_queries"] += len(queries_and_params)
                    self._stats["cache_misses"] += len(queries_and_params)
                    
                    avg_time_per_query = execution_time / len(queries_and_params)
                    self._update_avg_time(avg_time_per_query)
                
                return results
                
        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")
            raise
    
    def _update_stats(self, execution_time: float, cache_hit: bool):
        """Update performance statistics."""
        with self._stats_lock:
            self._stats["total_queries"] += 1
            
            if cache_hit:
                self._stats["cache_hits"] += 1
            else:
                self._stats["cache_misses"] += 1
            
            self._update_avg_time(execution_time)
    
    def _update_avg_time(self, execution_time: float):
        """Update average query time using exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self._stats["avg_query_time"] == 0:
            self._stats["avg_query_time"] = execution_time
        else:
            self._stats["avg_query_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self._stats["avg_query_time"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance and cache statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
        
        cache_stats = self.query_cache.get_stats()
        
        # Calculate additional metrics
        total_queries = stats["total_queries"]
        if total_queries > 0:
            cache_hit_ratio = stats["cache_hits"] / total_queries
        else:
            cache_hit_ratio = 0.0
        
        uptime = time.time() - stats["start_time"]
        
        return {
            **stats,
            "cache_hit_ratio": cache_hit_ratio,
            "uptime_seconds": uptime,
            "queries_per_second": total_queries / uptime if uptime > 0 else 0,
            "cache_stats": cache_stats,
            "connection_pool_size": self.connection_pool._created_connections
        }
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def close(self):
        """Close all connections and cleanup."""
        self.connection_pool.close_all()
        self.query_cache.clear()
        logger.info("OptimizedDatabaseManager closed")

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

def get_db_manager(database_path: str = "/home/notroot/Work/financebud/financial_data.db") -> DatabaseManager:
    """Get a singleton instance of the database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_path)
    return _db_manager
