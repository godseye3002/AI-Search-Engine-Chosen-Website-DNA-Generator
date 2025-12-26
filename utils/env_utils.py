"""
Environment utilities for production/development mode handling
"""

import os
from dotenv import load_dotenv

load_dotenv()

def is_production_mode():
    """Check if running in production mode"""
    node_env = os.getenv('NODE_ENV', 'development').lower()
    return node_env == 'production'

def is_debug_mode():
    """Check if running in debug/development mode"""
    return not is_production_mode()

def get_log_level():
    """Get appropriate log level based on environment"""
    return "ERROR" if is_production_mode() else "INFO"

def should_save_intermediate_files():
    """Check if intermediate files should be saved"""
    return is_debug_mode()

def should_save_stage_outputs(stage_name):
    """Check if stage outputs should be saved"""
    if is_production_mode():
        # In production, only save Stage 3 final output
        return stage_name == 'stage_3'
    return True  # In debug mode, save all stages
