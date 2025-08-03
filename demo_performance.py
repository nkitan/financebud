#!/usr/bin/env python3
"""
Quick Performance Demo
======================

Demonstrate the performance improvements of the optimized MCP server
by measuring actual response times.
"""

import asyncio
import time
import json
from fastmcp import Client as FastMCPClient

async def measure_tool_call(client, tool_name, args, iterations=3):
    """Measure the time for multiple tool calls."""
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        try:
            result = await client.call_tool(tool_name, args)
            execution_time = time.time() - start_time
            times.append(execution_time)
        except Exception as e:
            print(f"Error in {tool_name}: {e}")
            times.append(float('inf'))
    
    return times

async def demo_performance():
    """Demonstrate the optimized server performance."""
    print("🚀 Optimized MCP Server Performance Demo")
    print("=" * 50)
    
    # Test with optimized server
    print("\n📊 Testing Optimized Server...")
    client = FastMCPClient("optimized_mcp_server.py")
    
    async with client:
        # Test different tools
        tests = [
            ("get_account_summary", {}),
            ("get_recent_transactions", {"limit": 10}),
            ("search_transactions", {"pattern": "UPI", "limit": 5}),
            ("get_monthly_summary", {}),
            ("get_performance_stats", {})
        ]
        
        results = {}
        
        for tool_name, args in tests:
            print(f"\n⚡ Testing {tool_name}...")
            times = await measure_tool_call(client, tool_name, args, iterations=3)
            
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                
                print(f"  Average: {avg_time:.3f}s")
                print(f"  Range: {min_time:.3f}s - {max_time:.3f}s")
                
                results[tool_name] = {
                    "avg": avg_time,
                    "min": min_time,
                    "max": max_time,
                    "times": valid_times
                }
            else:
                print("  ❌ All calls failed")
        
        # Get performance statistics from the server
        print("\n📈 Server Performance Statistics:")
        try:
            stats_result = await client.call_tool("get_performance_stats", {})
            if hasattr(stats_result, 'content') and stats_result.content:
                for content_item in stats_result.content:
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        stats_data = json.loads(getattr(content_item, 'text', '{}'))
                        db_perf = stats_data.get('data', {}).get('database_performance', {})
                        
                        print(f"  Total queries: {db_perf.get('total_queries', 0)}")
                        print(f"  Cache hits: {db_perf.get('cache_hits', 0)}")
                        print(f"  Cache hit ratio: {db_perf.get('cache_hit_ratio', 0):.2%}")
                        print(f"  Avg query time: {db_perf.get('avg_query_time', 0):.3f}s")
                        print(f"  Queries per second: {db_perf.get('queries_per_second', 0):.1f}")
                        print(f"  Connection pool size: {db_perf.get('connection_pool_size', 0)}")
                        break
            else:
                # Fallback: try to get stats as string
                if isinstance(stats_result, str):
                    print(f"  Raw stats: {stats_result[:200]}...")
                        
        except Exception as e:
            print(f"  Could not get server stats: {e}")
        
        # Test cache performance
        print("\n💾 Testing Cache Performance...")
        print("Making 5 identical calls to see caching in action...")
        
        cache_test_times = []
        for i in range(5):
            start_time = time.time()
            result = await client.call_tool("get_account_summary", {})
            execution_time = time.time() - start_time
            cache_test_times.append(execution_time)
            print(f"  Call {i+1}: {execution_time:.3f}s")
        
        if len(cache_test_times) >= 2:
            first_call = cache_test_times[0]
            subsequent_avg = sum(cache_test_times[1:]) / len(cache_test_times[1:])
            cache_speedup = first_call / subsequent_avg if subsequent_avg > 0 else 1
            
            print(f"\n📊 Cache Performance Analysis:")
            print(f"  First call (cold): {first_call:.3f}s")
            print(f"  Subsequent calls (cached): {subsequent_avg:.3f}s average")
            print(f"  Cache speedup: {cache_speedup:.1f}x faster")
        
        print("\n✅ Performance demo completed!")
        print("\n🎯 Key Optimizations Active:")
        print("  • Database connection pooling")
        print("  • Query result caching")
        print("  • Optimized JSON serialization")
        print("  • Prepared statements")
        print("  • Performance monitoring")

if __name__ == "__main__":
    asyncio.run(demo_performance())
