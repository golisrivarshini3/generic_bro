"""
Supabase client configuration and initialization.
This module provides a shared Supabase client instance for database operations.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions
from functools import lru_cache

class SupabaseConnectionError(Exception):
    """Custom exception for Supabase connection errors."""
    pass

@lru_cache()
def get_supabase_client() -> Client:
    """
    Creates and returns a cached Supabase client instance.
    The client is cached to avoid creating multiple connections.
    
    Returns:
        Client: A configured Supabase client instance
        
    Raises:
        SupabaseConnectionError: If environment variables are missing or connection fails
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Supabase credentials
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_key: Optional[str] = os.getenv("SUPABASE_KEY")
    
    # Validate environment variables
    if not supabase_url or not supabase_key:
        missing_vars = []
        if not supabase_url:
            missing_vars.append("SUPABASE_URL")
        if not supabase_key:
            missing_vars.append("SUPABASE_KEY")
        raise SupabaseConnectionError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    try:
        # Initialize Supabase client with basic options
        options = ClientOptions(schema='public')
        
        # Initialize Supabase client
        client = create_client(supabase_url, supabase_key, options=options)
        
        # Test connection by making a simple query
        client.table('generic medicines list').select("Name").limit(1).execute()
        
        return client
        
    except Exception as e:
        raise SupabaseConnectionError(f"Failed to initialize Supabase client: {str(e)}")

# Create a shared client instance
try:
    supabase: Client = get_supabase_client()
except SupabaseConnectionError as e:
    print(f"Error initializing Supabase client: {str(e)}")
    raise

# Export the client as the main interface
__all__ = ["supabase", "SupabaseConnectionError"] 