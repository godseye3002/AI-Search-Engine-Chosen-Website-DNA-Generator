"""
Database Module for DNA Pipeline

Contains Supabase integration and database-driven workflow components.
"""

from .supabase_manager import SupabaseDataManager, DataSource, ProductAnalysisRecord, DNAAnalysisRecord

__all__ = [
    'SupabaseDataManager',
    'DataSource', 
    'ProductAnalysisRecord',
    'DNAAnalysisRecord'
]
