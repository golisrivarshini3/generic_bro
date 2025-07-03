from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional, Dict, Any, Literal
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

# Define the table name as a constant
MEDICINES_TABLE = 'generic medicines list'
MAX_RESULTS = 15  # Maximum number of results to return

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

def apply_price_sort(query, sort_order: Optional[str] = None) -> Any:
    """Apply price sorting to the query."""
    if sort_order == "low_to_high":
        return query.order("Cost of branded", desc=False)
    elif sort_order == "high_to_low":
        return query.order("Cost of branded", desc=True)
    return query

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

        # Get suggestions from cache
        suggestions = get_cached_suggestions(field, query)
        logger.info(f"Got {len(suggestions)} suggestions for {field} with query: {query}")

        # If getting type suggestions, log all available types for debugging
        if field == "Type":
            all_types = get_all_types()
            logger.info(f"All available types in database: {all_types}")

        return AutocompleteResponse(
            suggestions=[
                AutocompleteSuggestion(value=value, field_type=field)
                for value in suggestions
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
async def search_medicines(
    search_request: MedicineSearchRequest,
    sort_order: Optional[Literal["none", "low_to_high", "high_to_low"]] = None
):
    """Search for medicines with flexible filters and optional price sorting."""
    try:
        logger.info(f"Search request: {search_request}, Sort order: {sort_order}")
        
        # Build the query
        query = supabase.table(MEDICINES_TABLE).select("*")

        # Track if we're doing a type or dosage only search
        is_type_or_dosage_search = (search_request.type or search_request.dosage) and not (search_request.name or search_request.formulation)

        # Add filters
        if search_request.name:
            query = build_search_query(query, "Name", search_request.name)
            logger.info(f"Added name filter: {search_request.name}")
        
        if search_request.formulation:
            # For formulation, use ilike for more flexible matching
            formulation_value = clean_search_value(search_request.formulation)
            query = query.ilike("Formulation", f"%{formulation_value}%")
            logger.info(f"Added formulation filter: {formulation_value}")
        
        if search_request.type:
            # For type, try both exact match and partial match
            type_value = clean_search_value(search_request.type)
            # Use ilike for more flexible matching
            query = query.ilike("Type", f"%{type_value}%")
            logger.info(f"Added type filter: {type_value}")
        
        if search_request.dosage:
            query = build_search_query(query, "Dosage", search_request.dosage)
            logger.info(f"Added dosage filter: {search_request.dosage}")

        # Add sorting if specified
        if sort_order and sort_order != "none":
            query = apply_price_sort(query, sort_order)
            logger.info(f"Added price sorting: {sort_order}")

        # Add limit for type/dosage searches
        if is_type_or_dosage_search:
            query = query.limit(MAX_RESULTS)
            logger.info(f"Added limit of {MAX_RESULTS} for type/dosage search")

        # Execute query
        logger.info("Executing search query...")
        response = query.execute()
        
        if not response.data:
            logger.info("No medicines found")
            return SearchResponse(
                exact_match=None,
                similar_formulations=[],
                Uses=None,
                Side_Effects=None
            )
            
        medicines = response.data
        logger.info(f"Found {len(medicines)} medicines matching the criteria")

        # For type or dosage searches, return all results as similar formulations
        if is_type_or_dosage_search:
            logger.info("Processing type/dosage search results")
            try:
                processed_medicines = [create_medicine_from_db(m) for m in medicines]
                return SearchResponse(
                    exact_match=None,
                    similar_formulations=processed_medicines,
                    Uses=None,
                    Side_Effects=None
                )
            except Exception as e:
                logger.error(f"Error processing type/dosage search results: {str(e)}")
                logger.error(traceback.format_exc())
                raise

        # Handle name/formulation searches
        exact_match_medicine: Optional[Medicine] = None
        exact_match_data: Optional[Dict[str, Any]] = None
        formulation: Optional[str] = None
        
        # If name is provided, try to find exact match by name
        if search_request.name:
            exact_matches = [m for m in medicines if m["Name"].lower() == search_request.name.lower()]
            if exact_matches:
                exact_match_data = exact_matches[0]
                formulation = safe_get(exact_match_data, "Formulation")
                logger.info(f"Found exact match by name: {safe_get(exact_match_data, 'Name')}")
        # If no name match but formulation is provided, try to find exact match by formulation
        elif search_request.formulation:
            exact_matches = [m for m in medicines if m["Formulation"].lower() == search_request.formulation.lower()]
            if exact_matches:
                exact_match_data = exact_matches[0]
                formulation = safe_get(exact_match_data, "Formulation")
                logger.info(f"Found exact match by formulation: {formulation}")

        # Get similar formulations
        similar_formulations = []
        if search_request.formulation:
            # For formulation search, use all results as similar formulations
            similar_formulations = medicines
            logger.info(f"Using all {len(medicines)} results as similar formulations")
        elif formulation:
            # For name search with exact match, get medicines with similar formulation
            similar_query = supabase.table(MEDICINES_TABLE) \
                .select("*") \
                .ilike("Formulation", f"%{formulation}%")
            
            if exact_match_data:
                similar_query = similar_query.neq("Name", safe_get(exact_match_data, "Name"))
            
            # Apply sorting to similar formulations query if specified
            if sort_order and sort_order != "none":
                similar_query = apply_price_sort(similar_query, sort_order)
            
            similar_response = similar_query.execute()
            similar_formulations = similar_response.data or []
            logger.info(f"Found {len(similar_formulations)} similar formulations")
        else:
            # If no exact match, use all results as similar formulations
            similar_formulations = medicines
            logger.info("Using all results as similar formulations")

        # Convert data to Medicine objects
        try:
            processed_medicines = [create_medicine_from_db(m) for m in similar_formulations]
            if exact_match_data:
                exact_match_medicine = create_medicine_from_db(exact_match_data)
        except Exception as e:
            logger.error(f"Error converting medicines data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        return SearchResponse(
            exact_match=exact_match_medicine,
            similar_formulations=processed_medicines,
            Uses=safe_get(exact_match_data, "Uses"),
            Side_Effects=safe_get(exact_match_data, "Side Effects")
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