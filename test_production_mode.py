"""
Test script to demonstrate production vs development mode
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.env_utils import is_production_mode, is_debug_mode, should_save_stage_outputs

def test_environment_modes():
    """Test different environment modes"""
    
    print("=== Testing Environment Modes ===\n")
    
    # Test current mode
    current_mode = "production" if is_production_mode() else "development"
    print(f"Current mode: {current_mode}")
    print(f"Is production: {is_production_mode()}")
    print(f"Is debug: {is_debug_mode()}")
    
    print("\n=== File Saving Behavior ===")
    print(f"Stage 1 files saved: {should_save_stage_outputs('stage_1')}")
    print(f"Stage 2 files saved: {should_save_stage_outputs('stage_2')}")
    print(f"Stage 3 files saved: {should_save_stage_outputs('stage_3')}")
    
    print("\n=== Expected Behavior ===")
    if is_production_mode():
        print("PRODUCTION MODE:")
        print("- Only ERROR level logging")
        print("- Only Stage 3 final_aggregation.json saved")
        print("- No intermediate files")
        print("- Minimal console output")
    else:
        print("DEVELOPMENT MODE:")
        print("- Full INFO level logging")
        print("- All stage outputs saved")
        print("- Intermediate files included")
        print("- Verbose console output")
    
    print(f"\nNODE_ENV = {os.getenv('NODE_ENV', 'not set')}")

def switch_to_production():
    """Switch to production mode for testing"""
    print("\n=== Switching to Production Mode ===")
    os.environ['NODE_ENV'] = 'production'
    
    # Reload environment
    load_dotenv(override=True)
    
    print(f"NODE_ENV = {os.getenv('NODE_ENV')}")
    print(f"Is production: {is_production_mode()}")
    print(f"Stage 1 files saved: {should_save_stage_outputs('stage_1')}")
    print(f"Stage 2 files saved: {should_save_stage_outputs('stage_2')}")
    print(f"Stage 3 files saved: {should_save_stage_outputs('stage_3')}")

def switch_to_development():
    """Switch to development mode for testing"""
    print("\n=== Switching to Development Mode ===")
    os.environ['NODE_ENV'] = 'development'
    
    # Reload environment
    load_dotenv(override=True)
    
    print(f"NODE_ENV = {os.getenv('NODE_ENV')}")
    print(f"Is production: {is_production_mode()}")
    print(f"Stage 1 files saved: {should_save_stage_outputs('stage_1')}")
    print(f"Stage 2 files saved: {should_save_stage_outputs('stage_2')}")
    print(f"Stage 3 files saved: {should_save_stage_outputs('stage_3')}")

if __name__ == "__main__":
    # Test current configuration
    test_environment_modes()
    
    # Test switching modes
    switch_to_production()
    switch_to_development()
    
    print("\n=== Instructions ===")
    print("To enable PRODUCTION MODE:")
    print("1. Set NODE_ENV=production in .env file")
    print("2. Or set environment variable: export NODE_ENV=production")
    print("\nTo enable DEVELOPMENT MODE:")
    print("1. Set NODE_ENV=development in .env file")
    print("2. Or unset NODE_ENV (defaults to development)")
