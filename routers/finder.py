from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional, Dict, Any
import os
from db.supabase_client import supabase, SupabaseConnectionError
from models.schemas import (
    MedicineSearchRequest,
    Medicine,
    SearchResponse,
    AutocompleteSuggestion,
    AutocompleteResponse
)
from decimal import Decimal
from functools import lru_cache
from fastapi.responses import JSONResponse
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Define constants
MEDICINES_TABLE = 'generic medicines list'
MAX_RESULTS = 50  # Maximum number of results to return
MAX_SIMILAR_RESULTS = 20  # Maximum number of similar results
SUGGESTION_LIMIT = 10  # Maximum number of suggestions

def clean_search_value(value: str) -> str:
    """Clean and standardize search value."""
    if not value:
        return ""
    # Remove any leading/trailing whitespace
    cleaned = value.strip()
    # Remove any extra spaces around hyphens
    cleaned = "-".join(part.strip() for part in cleaned.split("-"))
    # Remove any quotes that might interfere with the query
    cleaned = cleaned.replace("'", "''")
    return cleaned

def build_search_query(table_query, field: str, value: str, exact: bool = False) -> Any:
    """Build a search query for a field."""
    if not value:
        return table_query
    
    cleaned_value = clean_search_value(value)
    if not cleaned_value:
        return table_query
    
    if exact:
        return table_query.eq(field, cleaned_value)
    return table_query.ilike(field, f"%{cleaned_value}%")

def safe_get(data: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary that might be None."""
    if data is None:
        return default
    return data.get(key, default)

# Cache for suggestions
@lru_cache(maxsize=1000)
def get_cached_suggestions(field: str, query: Optional[str] = None) -> List[str]:
    """Cache suggestions to reduce database load"""
    try:
        table_query = supabase.table(MEDICINES_TABLE).select(field)
        
        if query:
            cleaned_query = clean_search_value(query)
            table_query = table_query.ilike(field, f"%{cleaned_query}%")
        
        response = table_query.execute()
        
        if not response.data:
            return []
            
        suggestions = set()
        for item in response.data:
            value = item.get(field)
            if value and isinstance(value, str):
                suggestions.add(value)
        
        return sorted(list(suggestions))[:10]  # Limit to 10 suggestions
    except Exception as e:
        logger.error(f"Error in get_cached_suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        return []  # Return empty list instead of raising error

def create_medicine_from_db(data: Dict[str, Any]) -> Medicine:
    """Create a Medicine instance from database data."""
    try:
        return Medicine.model_validate(data)
    except Exception as e:
        logger.error(f"Error creating Medicine from data: {data}")
        logger.error(f"Error details: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_all_types() -> List[str]:
    """Get all unique types from the database for debugging."""
    try:
        response = supabase.table(MEDICINES_TABLE) \
            .select('Type') \
            .execute()
        types = set()
        for item in response.data:
            if item.get('Type'):
                types.add(item['Type'])
        return sorted(list(types))
    except Exception as e:
        logger.error(f"Error getting all types: {str(e)}")
        return []

@router.get("/suggestions/{field}", response_model=AutocompleteResponse)
async def get_suggestions(
    field: str,
    query: Optional[str] = Query(default=None, min_length=0),
):
    """Get suggestions for autocomplete dropdowns."""
    try:
        # Validate field
        valid_fields = ["Name", "Formulation", "Type", "Dosage"]
        if field not in valid_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid field. Must be one of: {', '.join(valid_fields)}"
            )

        # Build query
        table_query = supabase.table(MEDICINES_TABLE).select(field).limit(SUGGESTION_LIMIT)
        
        if query:
            cleaned_query = clean_search_value(query)
            table_query = table_query.ilike(field, f"%{cleaned_query}%")
        
        response = table_query.execute()
        
        if not response.data:
            return AutocompleteResponse(suggestions=[])
            
        # Get unique values
        suggestions = set()
        for item in response.data:
            value = item.get(field)
            if value and isinstance(value, str):
                suggestions.add(value)
        
        return AutocompleteResponse(
            suggestions=[
                AutocompleteSuggestion(value=value, field_type=field)
                for value in sorted(list(suggestions))[:SUGGESTION_LIMIT]
            ]
        )

    except SupabaseConnectionError:
        logger.error("Supabase connection error in get_suggestions")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Database connection error. Please try again later."}
        )
    except Exception as e:
        logger.error(f"Error in get_suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)}
        )

@router.post("/search", response_model=SearchResponse)
async def search_medicines(search_request: MedicineSearchRequest):
    """Search for medicines with flexible filters."""
    try:
        logger.info(f"Search request: {search_request}")
        
        # Build the base query
        base_query = supabase.table(MEDICINES_TABLE).select("*")
        
        # Track which filters are applied
        applied_filters = []
        
        # Add filters with proper error handling and logging
        if search_request.name:
            name_value = clean_search_value(search_request.name)
            if name_value:
                # Try exact match first
                exact_query = base_query.eq("Name", name_value)
                exact_response = exact_query.execute()
                
                if exact_response.data:
                    # If exact match found, use it
                    exact_match = exact_response.data[0]
                    # Get similar medicines based on formulation
                    formulation = exact_match.get("Formulation")
                    similar_query = base_query.neq("Name", name_value).ilike("Formulation", f"%{formulation}%")
                    similar_response = similar_query.limit(MAX_SIMILAR_RESULTS).execute()
                    
                    return SearchResponse(
                        exact_match=create_medicine_from_db(exact_match),
                        similar_formulations=[create_medicine_from_db(m) for m in similar_response.data],
                        Uses=exact_match.get("Uses"),
                        Side_Effects=exact_match.get("Side Effects")
                    )
                else:
                    # If no exact match, do partial match
                    base_query = base_query.ilike("Name", f"%{name_value}%")
                    applied_filters.append(f"name: {name_value}")
        
        if search_request.formulation:
            formulation_value = clean_search_value(search_request.formulation)
            if formulation_value:
                # Try exact match first
                exact_query = base_query.eq("Formulation", formulation_value)
                exact_response = exact_query.execute()
                
                if exact_response.data:
                    # If exact match found, use first as exact and rest as similar
                    exact_match = exact_response.data[0]
                    similar_formulations = exact_response.data[1:MAX_SIMILAR_RESULTS]
                    
                    # Get more similar formulations if needed
                    if len(similar_formulations) < MAX_SIMILAR_RESULTS:
                        remaining = MAX_SIMILAR_RESULTS - len(similar_formulations)
                        similar_query = base_query.ilike("Formulation", f"%{formulation_value}%") \
                            .neq("Formulation", formulation_value)
                        similar_response = similar_query.limit(remaining).execute()
                        similar_formulations.extend(similar_response.data)
                    
                    return SearchResponse(
                        exact_match=create_medicine_from_db(exact_match),
                        similar_formulations=[create_medicine_from_db(m) for m in similar_formulations],
                        Uses=exact_match.get("Uses"),
                        Side_Effects=exact_match.get("Side Effects")
                    )
                else:
                    # If no exact match, do partial match
                    base_query = base_query.ilike("Formulation", f"%{formulation_value}%")
                    applied_filters.append(f"formulation: {formulation_value}")
        
        # Handle type search with more flexibility
        if search_request.type:
            type_value = clean_search_value(search_request.type)
            if type_value:
                # Try exact match first
                exact_query = base_query.eq("Type", type_value)
                exact_response = exact_query.execute()
                
                if exact_response.data:
                    # If exact match found, use those results
                    medicines = exact_response.data
                else:
                    # If no exact match, try partial match with word boundaries
                    words = type_value.split()
                    query = base_query
                    for word in words:
                        query = query.ilike("Type", f"%{word}%")
                    medicines_response = query.execute()
                    medicines = medicines_response.data
                    
                    if not medicines:
                        # If still no results, try a more lenient search
                        base_query = base_query.ilike("Type", f"%{type_value}%")
                    else:
                        base_query = query
                
                applied_filters.append(f"type: {type_value}")
        
        # Handle dosage search with more flexibility
        if search_request.dosage:
            dosage_value = clean_search_value(search_request.dosage)
            if dosage_value:
                # Try exact match first
                exact_query = base_query.eq("Dosage", dosage_value)
                exact_response = exact_query.execute()
                
                if exact_response.data:
                    # If exact match found, use those results
                    medicines = exact_response.data
                else:
                    # If no exact match, try partial match with word boundaries
                    words = dosage_value.split()
                    query = base_query
                    for word in words:
                        query = query.ilike("Dosage", f"%{word}%")
                    medicines_response = query.execute()
                    medicines = medicines_response.data
                    
                    if not medicines:
                        # If still no results, try a more lenient search
                        base_query = base_query.ilike("Dosage", f"%{dosage_value}%")
                    else:
                        base_query = query
                
                applied_filters.append(f"dosage: {dosage_value}")

        # Log applied filters
        logger.info(f"Applied filters: {', '.join(applied_filters)}")
        
        # Execute final query with limit
        response = base_query.limit(MAX_RESULTS).execute()
        
        if not response.data:
            logger.info(f"No medicines found for filters: {applied_filters}")
            return SearchResponse(
                exact_match=None,
                similar_formulations=[],
                Uses=None,
                Side_Effects=None
            )
        
        # Process results
        medicines = response.data
        logger.info(f"Found {len(medicines)} medicines matching criteria: {applied_filters}")
        
        # Convert to Medicine objects
        processed_medicines = [create_medicine_from_db(m) for m in medicines]
        
        # Sort results by relevance if type search
        if search_request.type and type_value:
            processed_medicines.sort(
                key=lambda x: (
                    # Exact matches first
                    x.type.lower() != type_value.lower(),
                    # Then partial matches by length
                    len(x.type)
                )
            )
        
        return SearchResponse(
            exact_match=None,
            similar_formulations=processed_medicines,
            Uses=None,
            Side_Effects=None
        )

    except SupabaseConnectionError:
        logger.error("Supabase connection error in search_medicines")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Database connection error. Please try again later."}
        )
    except Exception as e:
        logger.error(f"Error in search_medicines: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)}
        )

@router.get("/medicine/{name}", response_model=Medicine)
async def get_medicine_details(name: str):
    """
    Get detailed information about a specific medicine by name.
    """
    try:
        response = supabase.table(MEDICINES_TABLE) \
            .select("*") \
            .eq("Name", name) \
            .limit(1) \
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Medicine not found")

        medicine = response.data[0]
        return Medicine.model_validate(medicine)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 