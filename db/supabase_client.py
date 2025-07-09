"""
Supabase client configuration and initialization.
This module provides a shared Supabase client instance for database operations.
"""

import os
from dotenv import load_dotenv
from supabase.client import create_client
from typing import Optional

load_dotenv()

class SupabaseConnectionError(Exception):
    """Custom exception for Supabase connection errors"""
    pass

def get_supabase_client():
    """Initialize and return Supabase client"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise SupabaseConnectionError("Supabase credentials not found in environment")
            
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        raise SupabaseConnectionError(f"Failed to initialize Supabase client: {str(e)}")

# Initialize the Supabase client
supabase = get_supabase_client() 