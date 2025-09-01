"""
Database Manager with Advanced Features
=======================================

Production-ready database manager with advanced connection pooling, caching,
monitoring, and transaction management capabilities.
"""

import asyncio
import sqlite3
import threading
import time
import hashlib
import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import psutil
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config.settings import get_settings
from backend.logging_config import get_logger_with_context

logger = get_logger_with_context(__name__)
settings = get_settings()


@dataclass
class QueryResult:
    """Result of a database query with metadata."""
    data: List[Dict[str, Any]]
    execution_time: float
    cached: bool = False
    rows_affected: int = 0
    query_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    total_requests: int = 0
    avg_wait_time: float = 0.0
    peak_connections: int = 0
    last_reset: datetime = field(default_factory=datetime.now)


class QueryMetrics:
    """Query performance metrics collector."""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_execution_time": 0.0,
            "slow_queries": 0,
            "query_types": {},
            "hourly_stats": {},
        }
        self._lock = threading.Lock()
    
    def record_query(self, query: str, execution_time: float, cached: bool, success: bool):
        """Record query execution metrics."""
        with self._lock:
            self.metrics["total_queries"] += 1
            
            if success:
                self.metrics["successful_queries"] += 1
            else:
                self.metrics["failed_queries"] += 1
            
            if cached:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
            
            # Update average execution time
            alpha = 0.1  # Smoothing factor
            current_avg = self.metrics["avg_execution_time"]
            self.metrics["avg_execution_time"] = (
                alpha * execution_time + (1 - alpha) * current_avg
            )
            
            # Track slow queries
            if execution_time > settings.monitoring.slow_query_threshold:
                self.metrics["slow_queries"] += 1
                logger.warning(f"Slow query detected: {execution_time:.3f}s - {query[:100]}")
            
            # Track query types
            query_type = query.strip().split()[0].upper()
            self.metrics["query_types"][query_type] = (
                self.metrics["query_types"].get(query_type, 0) + 1
            )
            
            # Track hourly stats
            hour = datetime.now().strftime("%Y-%m-%d %H:00")
            if hour not in self.metrics["hourly_stats"]:
                self.metrics["hourly_stats"][hour] = {"count": 0, "avg_time": 0.0}
            
            hourly = self.metrics["hourly_stats"][hour]
            hourly["count"] += 1
            hourly["avg_time"] = (
                (hourly["avg_time"] * (hourly["count"] - 1) + execution_time) / hourly["count"]
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "avg_execution_time": 0.0,
                "slow_queries": 0,
                "query_types": {},
                "hourly_stats": {},
            }


class AdvancedConnectionPool:
    """Advanced SQLite connection pool with monitoring and optimization."""
    
    def __init__(self, database_path: str, max_connections: int = 20):
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool = queue.Queue(maxsize=max_connections)
        self._all_connections = set()  # Use regular set instead of WeakSet
        self._lock = threading.RLock()
        self._stats = ConnectionStats()
        self._monitor_thread = None
        self._shutdown_event = threading.Event()
        
        # Initialize pool
        self._initialize_pool()
        self._start_monitoring()
    
    def _initialize_pool(self):
        """Initialize the connection pool with optimal connections."""
        initial_size = min(5, self.max_connections)
        for _ in range(initial_size):
            conn = self._create_optimized_connection()
            self._pool.put(conn)
            self._stats.total_connections += 1
            self._stats.idle_connections += 1
    
    def _create_optimized_connection(self) -> sqlite3.Connection:
        """Create an optimized SQLite connection."""
        conn = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            timeout=settings.database.pool_timeout,
            isolation_level=None  # Autocommit mode
        )
        
        # Apply database-specific optimizations
        optimizations = [
            f"PRAGMA journal_mode={settings.database.pragma_journal_mode}",
            f"PRAGMA synchronous={settings.database.pragma_synchronous}",
            f"PRAGMA cache_size={settings.database.pragma_cache_size}",
            f"PRAGMA temp_store={settings.database.pragma_temp_store}",
            f"PRAGMA mmap_size={settings.database.pragma_mmap_size}",
            "PRAGMA foreign_keys=ON",
            "PRAGMA page_size=4096",
            "PRAGMA locking_mode=NORMAL",
        ]
        
        for pragma in optimizations:
            try:
                conn.execute(pragma)
            except sqlite3.Error as e:
                logger.warning(f"Failed to apply optimization '{pragma}': {e}")
        
        # Set row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        # Add to tracking
        self._all_connections.add(conn)
        
        logger.debug(f"Created optimized connection to {self.database_path}")
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with comprehensive monitoring."""
        start_time = time.time()
        conn = None
        wait_time = 0.0
        
        try:
            with self._lock:
                self._stats.total_requests += 1
            
            # Try to get from pool
            try:
                conn = self._pool.get(timeout=2.0)
                wait_time = time.time() - start_time
                
                with self._lock:
                    self._stats.pool_hits += 1
                    self._stats.idle_connections -= 1
                    self._stats.active_connections += 1
                    
                    # Update average wait time
                    current_avg = self._stats.avg_wait_time
                    self._stats.avg_wait_time = (
                        0.1 * wait_time + 0.9 * current_avg
                    )
                
            except queue.Empty:
                # Create new connection if under limit
                with self._lock:
                    if self._stats.total_connections < self.max_connections:
                        conn = self._create_optimized_connection()
                        self._stats.total_connections += 1
                        self._stats.active_connections += 1
                        self._stats.pool_misses += 1
                        wait_time = time.time() - start_time
                    else:
                        # Wait longer for connection
                        pass
                
                if conn is None:
                    conn = self._pool.get(timeout=10.0)
                    wait_time = time.time() - start_time
                    
                    with self._lock:
                        self._stats.pool_hits += 1
                        self._stats.idle_connections -= 1
                        self._stats.active_connections += 1
            
            # Validate connection
            try:
                conn.execute("SELECT 1").fetchone()
            except sqlite3.Error:
                # Connection is bad, create a new one
                logger.warning("Bad connection detected, creating new one")
                conn.close()
                conn = self._create_optimized_connection()
            
            yield conn
            
        except Exception as e:
            logger.error(f"Connection pool error: {e}")
            if conn:
                try:
                    conn.close()
                except:
                    pass
                conn = None
            raise
        
        finally:
            # Return connection to pool
            if conn:
                try:
                    # Return to pool if space available
                    self._pool.put_nowait(conn)
                    with self._lock:
                        self._stats.active_connections -= 1
                        self._stats.idle_connections += 1
                except queue.Full:
                    # Pool is full, close connection
                    conn.close()
                    with self._lock:
                        self._stats.total_connections -= 1
                        self._stats.active_connections -= 1
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    conn.close()
                    with self._lock:
                        self._stats.total_connections -= 1
                        self._stats.active_connections -= 1
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(
                target=self._monitor_connections,
                daemon=True,
                name="ConnectionPoolMonitor"
            )
            self._monitor_thread.start()
    
    def _monitor_connections(self):
        """Monitor connection pool health and performance."""
        while not self._shutdown_event.is_set():
            try:
                # Update peak connections
                with self._lock:
                    current_total = self._stats.total_connections
                    if current_total > self._stats.peak_connections:
                        self._stats.peak_connections = current_total
                
                # Clean up dead connections
                self._cleanup_dead_connections()
                
                # Log pool status periodically
                if logger.isEnabledFor(logging.DEBUG):
                    stats = self.get_stats()
                    logger.debug(f"Pool stats: {stats}")
                
                # Wait before next check
                self._shutdown_event.wait(30.0)
                
            except Exception as e:
                logger.error(f"Connection pool monitoring error: {e}")
                self._shutdown_event.wait(5.0)
    
    def _cleanup_dead_connections(self):
        """Remove closed connections from tracking."""
        dead_connections = []
        for conn_ref in list(self._all_connections):
            try:
                # Test if connection is alive
                conn_ref.execute("SELECT 1")
            except (sqlite3.Error, AttributeError):
                dead_connections.append(conn_ref)
        
        if dead_connections:
            logger.debug(f"Cleaned up {len(dead_connections)} dead connections")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "total_connections": self._stats.total_connections,
                "active_connections": self._stats.active_connections,
                "idle_connections": self._stats.idle_connections,
                "pool_hits": self._stats.pool_hits,
                "pool_misses": self._stats.pool_misses,
                "total_requests": self._stats.total_requests,
                "avg_wait_time": self._stats.avg_wait_time,
                "peak_connections": self._stats.peak_connections,
                "hit_ratio": (
                    self._stats.pool_hits / max(1, self._stats.total_requests)
                ),
                "max_connections": self.max_connections,
                "pool_utilization": (
                    self._stats.total_connections / self.max_connections
                ),
            }
    
    def close_all(self):
        """Close all connections and shutdown monitoring."""
        self._shutdown_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        # Close all pooled connections
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error closing pooled connection: {e}")
        
        # Close any remaining tracked connections
        for conn in list(self._all_connections):
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing tracked connection: {e}")
        
        # Clear the connections set
        self._all_connections.clear()
        
        with self._lock:
            self._stats = ConnectionStats()


class QueryCache:
    """Advanced LRU cache with TTL, compression, and analytics."""
    
    def __init__(self, max_size: int = 2000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._access_counts = {}
        self._expiry_times = {}
        self._cache_sizes = {}
        self._lock = threading.RLock()
        
        # Analytics
        self._analytics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "memory_usage": 0,
        }
    
    def _get_cache_key(self, query: str, params: Tuple) -> str:
        """Generate cache key with query hash."""
        combined = f"{query.strip()}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _estimate_size(self, data: List[Dict[str, Any]]) -> int:
        """Estimate memory size of data."""
        try:
            return len(json.dumps(data).encode('utf-8'))
        except:
            return len(str(data).encode('utf-8'))
    
    def get(self, query: str, params: Tuple) -> Optional[List[Dict[str, Any]]]:
        """Get cached result with analytics."""
        cache_key = self._get_cache_key(query, params)
        current_time = time.time()
        
        with self._lock:
            if cache_key in self._cache:
                # Check expiration
                if current_time < self._expiry_times[cache_key]:
                    # Update access tracking
                    self._access_times[cache_key] = current_time
                    self._access_counts[cache_key] = (
                        self._access_counts.get(cache_key, 0) + 1
                    )
                    self._analytics["hits"] += 1
                    
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return self._cache[cache_key].copy()
                else:
                    # Expired, remove
                    self._remove_key(cache_key)
                    self._analytics["expirations"] += 1
            
            self._analytics["misses"] += 1
            return None
    
    def put(self, query: str, params: Tuple, result: List[Dict[str, Any]], ttl: Optional[int] = None):
        """Store result with size tracking and eviction."""
        cache_key = self._get_cache_key(query, params)
        current_time = time.time()
        expiry_time = current_time + (ttl or self.default_ttl)
        result_size = self._estimate_size(result)
        
        with self._lock:
            # Check if we need to evict
            while len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_entry()
            
            # Store the result
            self._cache[cache_key] = result.copy()
            self._access_times[cache_key] = current_time
            self._access_counts[cache_key] = 1
            self._expiry_times[cache_key] = expiry_time
            self._cache_sizes[cache_key] = result_size
            
            # Update memory usage
            self._analytics["memory_usage"] = sum(self._cache_sizes.values())
            
            logger.debug(f"Cached result for query: {query[:50]}... (size: {result_size} bytes)")
    
    def _remove_key(self, cache_key: str):
        """Remove a key and update memory tracking."""
        self._cache.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
        self._access_counts.pop(cache_key, None)
        self._expiry_times.pop(cache_key, None)
        size = self._cache_sizes.pop(cache_key, 0)
        self._analytics["memory_usage"] -= size
    
    def _evict_entry(self):
        """Evict entry using LRU + LFU hybrid strategy."""
        if not self._access_times:
            return
        
        # Find candidates with low access frequency
        min_access_count = min(self._access_counts.values())
        low_frequency_keys = [
            key for key, count in self._access_counts.items()
            if count == min_access_count
        ]
        
        # Among low frequency, pick LRU
        oldest_key = min(
            low_frequency_keys,
            key=lambda k: self._access_times[k]
        )
        
        self._remove_key(oldest_key)
        self._analytics["evictions"] += 1
        logger.debug("Evicted cache entry using LRU+LFU strategy")
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get cache analytics and statistics."""
        with self._lock:
            total_requests = self._analytics["hits"] + self._analytics["misses"]
            hit_ratio = (
                self._analytics["hits"] / max(1, total_requests)
            )
            
            return {
                **self._analytics,
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "hit_ratio": hit_ratio,
                "fill_ratio": len(self._cache) / self.max_size,
                "avg_access_count": (
                    sum(self._access_counts.values()) / max(1, len(self._access_counts))
                ),
                "memory_usage_mb": self._analytics["memory_usage"] / (1024 * 1024),
            }
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._expiry_times.clear()
            self._cache_sizes.clear()
            self._analytics["memory_usage"] = 0


class DatabaseManager:
    """database manager with advanced features and monitoring."""
    
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or str(settings.database.database_path)
        
        # Initialize components
        self.connection_pool = AdvancedConnectionPool(
            self.database_path,
            settings.database.pool_size
        )
        self.query_cache = QueryCache(
            settings.database.cache_size,
            settings.database.cache_ttl
        )
        self.query_metrics = QueryMetrics()
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="DatabaseWorker"
        )
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"DatabaseManager initialized for {self.database_path}")
    
    def _initialize_database(self):
        """Initialize database with necessary tables and optimizations."""
        try:
            # Ensure database file exists
            db_path = Path(self.database_path)
            if not db_path.exists():
                logger.warning(f"Database file not found: {self.database_path}")
                # You might want to create the database here or raise an error
            
            # Test connection
            with self.connection_pool.get_connection() as conn:
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                tables = [row[0] for row in result]
                logger.info(f"Database connected successfully. Found tables: {tables}")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def execute_query(
        self,
        query: str,
        params: Tuple = (),
        cache_ttl: Optional[int] = None
    ) -> QueryResult:
        """Execute a query with advanced error handling and retry logic."""
        start_time = time.time()
        query_hash = hashlib.md5(f"{query}:{params}".encode()).hexdigest()
        
        # Check cache for SELECT queries
        cached_result = None
        if query.strip().upper().startswith('SELECT'):
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                execution_time = time.time() - start_time
                self.query_metrics.record_query(query, execution_time, True, True)
                return QueryResult(
                    data=cached_result,
                    execution_time=execution_time,
                    cached=True,
                    query_hash=query_hash
                )
        
        # Execute query
        success = False
        result_data = []
        rows_affected = 0
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(query, params)
                
                if query.strip().upper().startswith('SELECT'):
                    result_data = [dict(row) for row in cursor.fetchall()]
                    rows_affected = len(result_data)
                else:
                    rows_affected = cursor.rowcount
                
                execution_time = time.time() - start_time
                success = True
                
                # Cache SELECT results
                if (query.strip().upper().startswith('SELECT') and 
                    cache_ttl != 0 and result_data):
                    self.query_cache.put(query, params, result_data, cache_ttl)
                
                # Record metrics
                self.query_metrics.record_query(query, execution_time, False, True)
                
                return QueryResult(
                    data=result_data,
                    execution_time=execution_time,
                    cached=False,
                    rows_affected=rows_affected,
                    query_hash=query_hash,
                    metadata={
                        "query_type": query.strip().split()[0].upper(),
                        "param_count": len(params),
                        "connection_wait_time": 0.0
                    }
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.query_metrics.record_query(query, execution_time, False, False)
            
            logger.error(f"Query execution failed: {e}\nQuery: {query}\nParams: {params}")
            raise
    
    async def execute_query_async(
        self,
        query: str,
        params: Tuple = (),
        cache_ttl: Optional[int] = None
    ) -> QueryResult:
        """Execute query asynchronously using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.execute_query,
            query,
            params,
            cache_ttl
        )
    
    def execute_transaction(
        self,
        queries_and_params: List[Tuple[str, Tuple]]
    ) -> List[QueryResult]:
        """Execute multiple queries in a transaction with rollback support."""
        start_time = time.time()
        results = []
        
        try:
            with self.connection_pool.get_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                
                try:
                    for i, (query, params) in enumerate(queries_and_params):
                        cursor = conn.execute(query, params)
                        
                        if query.strip().upper().startswith('SELECT'):
                            data = [dict(row) for row in cursor.fetchall()]
                            rows_affected = len(data)
                        else:
                            data = []
                            rows_affected = cursor.rowcount
                        
                        results.append(QueryResult(
                            data=data,
                            execution_time=0.0,  # Individual timing not tracked in transactions
                            cached=False,
                            rows_affected=rows_affected,
                            query_hash=hashlib.md5(f"{query}:{params}".encode()).hexdigest(),
                            metadata={"transaction_index": i}
                        ))
                    
                    conn.execute("COMMIT")
                    
                    execution_time = time.time() - start_time
                    logger.info(f"Transaction completed successfully in {execution_time:.3f}s "
                              f"({len(queries_and_params)} queries)")
                    
                    # Record transaction metrics
                    for query, params in queries_and_params:
                        self.query_metrics.record_query(
                            query,
                            execution_time / len(queries_and_params),
                            False,
                            True
                        )
                    
                    return results
                    
                except Exception as e:
                    conn.execute("ROLLBACK")
                    logger.error(f"Transaction rolled back due to error: {e}")
                    raise
                    
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failed transaction metrics
            for query, params in queries_and_params:
                self.query_metrics.record_query(
                    query,
                    execution_time / len(queries_and_params),
                    False,
                    False
                )
            
            logger.error(f"Transaction failed: {e}")
            raise
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics and health metrics."""
        pool_stats = self.connection_pool.get_stats()
        cache_analytics = self.query_cache.get_analytics()
        query_metrics = self.query_metrics.get_metrics()
        
        # Database file stats
        db_path = Path(self.database_path)
        file_stats = {}
        if db_path.exists():
            stat = db_path.stat()
            file_stats = {
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        
        # System resources
        process = psutil.Process()
        system_stats = {
            "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
            "cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "threads": process.num_threads(),
        }
        
        return {
            "database_file": file_stats,
            "connection_pool": pool_stats,
            "query_cache": cache_analytics,
            "query_metrics": query_metrics,
            "system_resources": system_stats,
            "health_score": self._calculate_health_score(
                pool_stats, cache_analytics, query_metrics
            ),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _calculate_health_score(
        self,
        pool_stats: Dict[str, Any],
        cache_analytics: Dict[str, Any],
        query_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall database health score (0-100)."""
        score = 100.0
        
        # Pool health (30% weight)
        pool_utilization = pool_stats.get("pool_utilization", 0)
        if pool_utilization > 0.9:
            score -= 20  # High utilization is concerning
        elif pool_utilization > 0.7:
            score -= 10
        
        hit_ratio = pool_stats.get("hit_ratio", 0)
        if hit_ratio < 0.8:
            score -= 10  # Low hit ratio indicates contention
        
        # Cache health (30% weight)
        cache_hit_ratio = cache_analytics.get("hit_ratio", 0)
        if cache_hit_ratio < 0.6:
            score -= 15  # Poor cache performance
        elif cache_hit_ratio < 0.8:
            score -= 5
        
        cache_fill = cache_analytics.get("fill_ratio", 0)
        if cache_fill > 0.95:
            score -= 5  # Cache nearly full
        
        # Query performance (40% weight)
        avg_execution_time = query_metrics.get("avg_execution_time", 0)
        if avg_execution_time > 1.0:
            score -= 20  # Slow queries
        elif avg_execution_time > 0.5:
            score -= 10
        
        slow_query_ratio = (
            query_metrics.get("slow_queries", 0) / 
            max(1, query_metrics.get("total_queries", 1))
        )
        if slow_query_ratio > 0.1:
            score -= 10  # Too many slow queries
        
        success_ratio = (
            query_metrics.get("successful_queries", 0) / 
            max(1, query_metrics.get("total_queries", 1))
        )
        if success_ratio < 0.95:
            score -= 15  # Too many failures
        
        return max(0.0, min(100.0, score))
    
    def optimize_database(self):
        """Perform database optimization operations."""
        logger.info("Starting database optimization...")
        
        optimization_queries = [
            "VACUUM",
            "ANALYZE",
            "REINDEX",
        ]
        
        try:
            with self.connection_pool.get_connection() as conn:
                for query in optimization_queries:
                    start_time = time.time()
                    conn.execute(query)
                    execution_time = time.time() - start_time
                    logger.info(f"Completed {query} in {execution_time:.3f}s")
            
            logger.info("Database optimization completed successfully")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Test basic connectivity
        try:
            with self.connection_pool.get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
            health_status["checks"]["connectivity"] = {"status": "ok", "message": "Database connection successful"}
        except Exception as e:
            health_status["checks"]["connectivity"] = {"status": "error", "message": str(e)}
            health_status["status"] = "unhealthy"
        
        # Check pool health
        pool_stats = self.connection_pool.get_stats()
        if pool_stats["pool_utilization"] > 0.9:
            health_status["checks"]["pool"] = {"status": "warning", "message": "High pool utilization"}
            health_status["status"] = "degraded"
        else:
            health_status["checks"]["pool"] = {"status": "ok", "message": "Pool utilization normal"}
        
        # Check query performance
        query_metrics = self.query_metrics.get_metrics()
        if query_metrics["avg_execution_time"] > 1.0:
            health_status["checks"]["performance"] = {"status": "warning", "message": "Slow query performance"}
            health_status["status"] = "degraded"
        else:
            health_status["checks"]["performance"] = {"status": "ok", "message": "Query performance normal"}
        
        # Overall health score
        stats = self.get_comprehensive_stats()
        health_score = stats["health_score"]
        
        if health_score < 70:
            health_status["status"] = "unhealthy"
        elif health_score < 85:
            health_status["status"] = "degraded"
        
        health_status["health_score"] = health_score
        
        return health_status
    
    def close(self):
        """Close all resources and shutdown gracefully."""
        logger.info("Shutting down DatabaseManager...")
        
        try:
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Close connection pool
            self.connection_pool.close_all()
            
            # Clear cache
            self.query_cache.clear()
            
            logger.info("DatabaseManager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during DatabaseManager shutdown: {e}")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(database_path: Optional[str] = None) -> DatabaseManager:
    """Get database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_path)
    return _db_manager

