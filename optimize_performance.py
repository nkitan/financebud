#!/usr/bin/env python3
"""
MCP Server Performance Optimizer
=================================

Script to switch between original and optimized MCP server implementations
and update the backend to use the appropriate version.

Usage:
    python optimize_performance.py --enable    # Enable optimized implementation
    python optimize_performance.py --disable   # Revert to original implementation
    python optimize_performance.py --status    # Check current implementation
    python optimize_performance.py --test      # Run performance tests
"""

import argparse
import os
import shutil
import sys
import asyncio
import subprocess
from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def backup_file(file_path: Path, backup_suffix: str = ".backup"):
    """Create a backup of a file."""
    if file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        shutil.copy2(file_path, backup_path)
        print(f"📋 Backed up {file_path} to {backup_path}")
        return backup_path
    return None

def restore_file(file_path: Path, backup_suffix: str = ".backup"):
    """Restore a file from backup."""
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
    if backup_path.exists():
        shutil.copy2(backup_path, file_path)
        print(f"🔄 Restored {file_path} from {backup_path}")
        return True
    return False

def update_main_py_for_optimized():
    """Update main.py to use optimized components."""
    project_root = get_project_root()
    main_py = project_root / "backend" / "main.py"
    
    if not main_py.exists():
        print(f"❌ {main_py} not found")
        return False
    
    # Backup original
    backup_file(main_py)
    
    # Read current content
    with open(main_py, 'r') as f:
        content = f.read()
    
    # Replace imports and references
    replacements = [
        # Replace financial agent import
        (
            "from .agents.financial_agent import GenericFinancialAgent, get_financial_agent",
            "from .agents.optimized_financial_agent import OptimizedFinancialAgent, get_optimized_financial_agent"
        ),
        # Replace MCP client import
        (
            "from .mcp.client import MCPClientManager",
            "from .mcp.persistent_client import PersistentMCPManager, get_persistent_mcp_manager"
        ),
        # Replace agent initialization
        (
            "financial_agent = None",
            "financial_agent = None\npersistent_mcp_manager = None"
        ),
        # Replace get_financial_agent calls
        (
            "get_financial_agent",
            "get_optimized_financial_agent"
        ),
        # Replace MCPClientManager instantiation
        (
            "mcp_manager = MCPClientManager()",
            "# mcp_manager = MCPClientManager()  # Replaced with persistent manager"
        )
    ]
    
    modified_content = content
    for old, new in replacements:
        if old in modified_content:
            modified_content = modified_content.replace(old, new)
            print(f"✅ Replaced: {old[:50]}...")
    
    # Write modified content
    with open(main_py, 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Updated {main_py} for optimized implementation")
    return True

def update_main_py_for_original():
    """Restore main.py to use original components."""
    project_root = get_project_root()
    main_py = project_root / "backend" / "main.py"
    
    if restore_file(main_py):
        print("✅ Restored main.py to original implementation")
        return True
    else:
        print("❌ No backup found for main.py")
        return False

def update_mcp_server():
    """Update the main MCP server to use optimized version."""
    project_root = get_project_root()
    original_server = project_root / "mcp_server.py"
    optimized_server = project_root / "optimized_mcp_server.py"
    
    if not optimized_server.exists():
        print(f"❌ {optimized_server} not found")
        return False
    
    # Backup original server
    backup_file(original_server)
    
    # Copy optimized server
    shutil.copy2(optimized_server, original_server)
    print(f"✅ Updated {original_server} with optimized implementation")
    return True

def restore_mcp_server():
    """Restore the original MCP server."""
    project_root = get_project_root()
    original_server = project_root / "mcp_server.py"
    
    if restore_file(original_server):
        print("✅ Restored mcp_server.py to original implementation")
        return True
    else:
        print("❌ No backup found for mcp_server.py")
        return False

def check_status():
    """Check which implementation is currently active."""
    project_root = get_project_root()
    
    print("🔍 Current Implementation Status:")
    print("-" * 40)
    
    # Check main.py
    main_py = project_root / "backend" / "main.py"
    if main_py.exists():
        with open(main_py, 'r') as f:
            content = f.read()
        
        if "optimized_financial_agent" in content:
            print("✅ Backend: Optimized implementation")
        else:
            print("🔧 Backend: Original implementation")
    else:
        print("❌ Backend: main.py not found")
    
    # Check MCP server
    mcp_server = project_root / "mcp_server.py"
    if mcp_server.exists():
        with open(mcp_server, 'r') as f:
            content = f.read()
        
        if "Optimized FastMCP server" in content:
            print("✅ MCP Server: Optimized implementation")
        else:
            print("🔧 MCP Server: Original implementation")
    else:
        print("❌ MCP Server: mcp_server.py not found")
    
    # Check for backup files
    backups = list(project_root.rglob("*.backup"))
    if backups:
        print(f"📋 Backup files found: {len(backups)}")
        for backup in backups:
            print(f"  {backup}")
    else:
        print("📋 No backup files found")

def enable_optimizations():
    """Enable all optimizations."""
    print("🚀 Enabling Performance Optimizations...")
    print("=" * 50)
    
    success = True
    
    # Update backend
    if not update_main_py_for_optimized():
        success = False
    
    # Update MCP server
    if not update_mcp_server():
        success = False
    
    if success:
        print("\n✅ All optimizations enabled successfully!")
        print("📈 Expected improvements:")
        print("  • 60-80% faster tool calls")
        print("  • Persistent MCP connections")
        print("  • Database connection pooling")
        print("  • Query result caching")
        print("  • Reduced memory usage")
        print("\n🔄 Restart your backend to activate optimizations")
    else:
        print("\n❌ Some optimizations failed to enable")
    
    return success

def disable_optimizations():
    """Disable all optimizations and restore originals."""
    print("🔧 Disabling Performance Optimizations...")
    print("=" * 50)
    
    success = True
    
    # Restore backend
    if not update_main_py_for_original():
        success = False
    
    # Restore MCP server
    if not restore_mcp_server():
        success = False
    
    if success:
        print("\n✅ Restored to original implementation!")
        print("🔄 Restart your backend to use original implementation")
    else:
        print("\n❌ Some restorations failed")
    
    return success

async def run_performance_test():
    """Run the performance comparison test."""
    print("🧪 Running Performance Tests...")
    print("=" * 50)
    
    project_root = get_project_root()
    test_script = project_root / "tests" / "test_performance_optimization.py"
    
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return False
    
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, str(test_script)
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("✅ Performance test completed successfully!")
            print("\nTest Output:")
            print(result.stdout)
        else:
            print("❌ Performance test failed!")
            print("Error Output:")
            print(result.stderr)
            return False
        
    except Exception as e:
        print(f"❌ Failed to run performance test: {e}")
        return False
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MCP Server Performance Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_performance.py --enable     # Enable optimizations
  python optimize_performance.py --disable    # Disable optimizations  
  python optimize_performance.py --status     # Check current status
  python optimize_performance.py --test       # Run performance tests
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable", action="store_true", 
                      help="Enable performance optimizations")
    group.add_argument("--disable", action="store_true", 
                      help="Disable optimizations and restore originals")
    group.add_argument("--status", action="store_true", 
                      help="Check current implementation status")
    group.add_argument("--test", action="store_true", 
                      help="Run performance comparison tests")
    
    args = parser.parse_args()
    
    if args.enable:
        enable_optimizations()
    elif args.disable:
        disable_optimizations()
    elif args.status:
        check_status()
    elif args.test:
        asyncio.run(run_performance_test())

if __name__ == "__main__":
    main()
