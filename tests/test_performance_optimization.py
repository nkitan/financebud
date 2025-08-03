#!/usr/bin/env python3
"""
Performance Comparison Test
===========================

Compare performance between the original MCP implementation and the optimized version.
Tests various scenarios including:
- Cold start performance
- Repeated query performance
- Concurrent query handling
- Memory usage
- Cache effectiveness
"""

import asyncio
import time
import statistics
import json
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import psutil
import threading

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.mcp.persistent_client import get_persistent_mcp_manager
from fastmcp import Client as FastMCPClient

class PerformanceTest:
    """Performance testing suite for MCP implementations."""
    
    def __init__(self):
        self.results = {
            "original": {},
            "optimized": {}
        }
        self.test_queries = [
            ("get_account_summary", {}),
            ("get_recent_transactions", {"limit": 10}),
            ("search_transactions", {"pattern": "UPI", "limit": 20}),
            ("get_monthly_summary", {"year": 2024, "month": 1}),
            ("get_spending_by_category", {"days": 30})
        ]
    
    async def test_original_implementation(self, iterations: int = 10) -> Dict[str, Any]:
        """Test the original FastMCP implementation."""
        print("🔧 Testing Original Implementation...")
        
        results = {
            "cold_start_times": [],
            "query_times": {},
            "total_time": 0,
            "memory_usage": []
        }
        
        start_time = time.time()
        
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}")
            
            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            for tool_name, args in self.test_queries:
                tool_start = time.time()
                
                try:
                    # Create new client each time (original behavior)
                    client = FastMCPClient("mcp_server.py")
                    async with client:
                        result = await client.call_tool(tool_name, args)
                    
                    execution_time = time.time() - tool_start
                    
                    if tool_name not in results["query_times"]:
                        results["query_times"][tool_name] = []
                    results["query_times"][tool_name].append(execution_time)
                    
                except Exception as e:
                    print(f"    Error in {tool_name}: {e}")
                    results["query_times"].setdefault(tool_name, []).append(float('inf'))
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            results["memory_usage"].append(memory_after - memory_before)
        
        results["total_time"] = time.time() - start_time
        
        # Calculate statistics
        for tool_name in results["query_times"]:
            times = [t for t in results["query_times"][tool_name] if t != float('inf')]
            if times:
                results["query_times"][tool_name] = {
                    "avg": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                    "std": statistics.stdev(times) if len(times) > 1 else 0,
                    "successful_calls": len(times),
                    "failed_calls": len(results["query_times"][tool_name]) - len(times)
                }
        
        return results
    
    async def test_optimized_implementation(self, iterations: int = 10) -> Dict[str, Any]:
        """Test the optimized persistent MCP implementation."""
        print("🚀 Testing Optimized Implementation...")
        
        results = {
            "startup_time": 0,
            "query_times": {},
            "total_time": 0,
            "memory_usage": [],
            "cache_performance": {}
        }
        
        # Measure startup time
        startup_start = time.time()
        mcp_manager = await get_persistent_mcp_manager()
        results["startup_time"] = time.time() - startup_start
        
        start_time = time.time()
        
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}")
            
            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            for tool_name, args in self.test_queries:
                tool_start = time.time()
                
                try:
                    # Use persistent connection
                    result = await mcp_manager.call_tool("financial-data-inr", tool_name, args)
                    
                    execution_time = time.time() - tool_start
                    
                    if tool_name not in results["query_times"]:
                        results["query_times"][tool_name] = []
                    results["query_times"][tool_name].append(execution_time)
                    
                except Exception as e:
                    print(f"    Error in {tool_name}: {e}")
                    results["query_times"].setdefault(tool_name, []).append(float('inf'))
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            results["memory_usage"].append(memory_after - memory_before)
        
        results["total_time"] = time.time() - start_time
        
        # Calculate statistics
        for tool_name in results["query_times"]:
            times = [t for t in results["query_times"][tool_name] if t != float('inf')]
            if times:
                results["query_times"][tool_name] = {
                    "avg": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                    "std": statistics.stdev(times) if len(times) > 1 else 0,
                    "successful_calls": len(times),
                    "failed_calls": len(results["query_times"][tool_name]) - len(times)
                }
        
        # Get cache performance
        try:
            health = await mcp_manager.health_check()
            if "financial-data-inr" in health["servers"]:
                server_stats = health["servers"]["financial-data-inr"].get("stats", {})
                results["cache_performance"] = server_stats
        except Exception as e:
            print(f"Could not get cache performance: {e}")
        
        return results
    
    async def test_concurrent_performance(self, concurrent_requests: int = 10) -> Dict[str, Any]:
        """Test concurrent request handling."""
        print(f"⚡ Testing Concurrent Performance ({concurrent_requests} concurrent requests)...")
        
        # Test optimized version with concurrent requests
        mcp_manager = await get_persistent_mcp_manager()
        
        async def make_request():
            start_time = time.time()
            try:
                result = await mcp_manager.call_tool("financial-data-inr", "get_account_summary", {})
                return time.time() - start_time
            except Exception as e:
                print(f"Concurrent request error: {e}")
                return float('inf')
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(concurrent_requests)]
        request_times = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Filter out failed requests
        successful_times = [t for t in request_times if t != float('inf')]
        
        return {
            "concurrent_requests": concurrent_requests,
            "successful_requests": len(successful_times),
            "failed_requests": len(request_times) - len(successful_times),
            "total_time": total_time,
            "avg_request_time": statistics.mean(successful_times) if successful_times else 0,
            "min_request_time": min(successful_times) if successful_times else 0,
            "max_request_time": max(successful_times) if successful_times else 0,
            "requests_per_second": len(successful_times) / total_time if total_time > 0 else 0
        }
    
    def calculate_improvements(self) -> Dict[str, Any]:
        """Calculate performance improvements."""
        improvements = {}
        
        if "original" in self.results and "optimized" in self.results:
            original = self.results["original"]
            optimized = self.results["optimized"]
            
            # Compare query times
            for tool_name in original.get("query_times", {}):
                if tool_name in optimized.get("query_times", {}):
                    orig_avg = original["query_times"][tool_name]["avg"]
                    opt_avg = optimized["query_times"][tool_name]["avg"]
                    
                    if orig_avg > 0:
                        improvement = ((orig_avg - opt_avg) / orig_avg) * 100
                        improvements[tool_name] = {
                            "original_avg": orig_avg,
                            "optimized_avg": opt_avg,
                            "improvement_percent": improvement,
                            "speedup_factor": orig_avg / opt_avg if opt_avg > 0 else float('inf')
                        }
            
            # Compare total times
            if original.get("total_time", 0) > 0 and optimized.get("total_time", 0) > 0:
                total_improvement = ((original["total_time"] - optimized["total_time"]) / original["total_time"]) * 100
                improvements["total_execution"] = {
                    "original_total": original["total_time"],
                    "optimized_total": optimized["total_time"],
                    "improvement_percent": total_improvement,
                    "speedup_factor": original["total_time"] / optimized["total_time"]
                }
            
            # Compare memory usage
            orig_memory = statistics.mean(original.get("memory_usage", [0]))
            opt_memory = statistics.mean(optimized.get("memory_usage", [0]))
            
            if orig_memory > 0:
                memory_improvement = ((orig_memory - opt_memory) / orig_memory) * 100
                improvements["memory_usage"] = {
                    "original_avg_mb": orig_memory,
                    "optimized_avg_mb": opt_memory,
                    "improvement_percent": memory_improvement
                }
        
        return improvements
    
    def print_results(self):
        """Print detailed test results."""
        print("\n" + "="*80)
        print("📊 PERFORMANCE TEST RESULTS")
        print("="*80)
        
        # Original results
        if "original" in self.results:
            print("\n🔧 Original Implementation:")
            original = self.results["original"]
            print(f"  Total execution time: {original.get('total_time', 0):.3f}s")
            print(f"  Average memory usage: {statistics.mean(original.get('memory_usage', [0])):.2f} MB")
            
            print("  Query performance:")
            for tool_name, stats in original.get("query_times", {}).items():
                if isinstance(stats, dict):
                    print(f"    {tool_name}: {stats['avg']:.3f}s avg (±{stats['std']:.3f}s)")
        
        # Optimized results
        if "optimized" in self.results:
            print("\n🚀 Optimized Implementation:")
            optimized = self.results["optimized"]
            print(f"  Startup time: {optimized.get('startup_time', 0):.3f}s")
            print(f"  Total execution time: {optimized.get('total_time', 0):.3f}s")
            print(f"  Average memory usage: {statistics.mean(optimized.get('memory_usage', [0])):.2f} MB")
            
            print("  Query performance:")
            for tool_name, stats in optimized.get("query_times", {}).items():
                if isinstance(stats, dict):
                    print(f"    {tool_name}: {stats['avg']:.3f}s avg (±{stats['std']:.3f}s)")
            
            # Cache performance
            cache_perf = optimized.get("cache_performance", {})
            if cache_perf:
                print(f"  Cache hits: {cache_perf.get('successful_requests', 0)}")
                print(f"  Cache misses: {cache_perf.get('failed_requests', 0)}")
        
        # Concurrent performance
        if "concurrent" in self.results:
            print("\n⚡ Concurrent Performance:")
            concurrent = self.results["concurrent"]
            print(f"  Requests per second: {concurrent.get('requests_per_second', 0):.2f}")
            print(f"  Average request time: {concurrent.get('avg_request_time', 0):.3f}s")
            print(f"  Success rate: {(concurrent.get('successful_requests', 0) / concurrent.get('concurrent_requests', 1) * 100):.1f}%")
        
        # Improvements
        improvements = self.calculate_improvements()
        if improvements:
            print("\n📈 PERFORMANCE IMPROVEMENTS:")
            for metric, data in improvements.items():
                if "improvement_percent" in data:
                    print(f"  {metric}: {data['improvement_percent']:.1f}% faster ({data.get('speedup_factor', 1):.2f}x speedup)")
    
    async def run_all_tests(self, iterations: int = 5):
        """Run all performance tests."""
        print("🧪 Starting Performance Comparison Tests...")
        print(f"📊 Running {iterations} iterations of each test")
        
        try:
            # Test original implementation
            self.results["original"] = await self.test_original_implementation(iterations)
            
            # Test optimized implementation
            self.results["optimized"] = await self.test_optimized_implementation(iterations)
            
            # Test concurrent performance
            self.results["concurrent"] = await self.test_concurrent_performance(20)
            
            # Print results
            self.print_results()
            
            # Save results to file
            with open("performance_test_results.json", "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"\n💾 Results saved to performance_test_results.json")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main test function."""
    test_suite = PerformanceTest()
    await test_suite.run_all_tests(iterations=3)

if __name__ == "__main__":
    asyncio.run(main())
