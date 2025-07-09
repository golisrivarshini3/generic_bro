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

def clean_type_value(value: str) -> str:
    """Clean and standardize type value."""
    if not value:
        return ""
    
    # Debug: Log the raw input value
    logger.info(f"Raw type value: {repr(value)}")
    
    # Basic cleaning: remove extra whitespace
    cleaned = " ".join(value.split())
    logger.info(f"After basic cleaning: {repr(cleaned)}")
    
    return cleaned

def build_search_query(table_query, field: str, value: str, exact: bool = False) -> Any:
    """Build a search query for a field."""
    if not value:
        return table_query
    
    # Map the field names to database column names
    field_mapping = {
        "type": "Type",
        "name": "Name",
        "formulation": "Formulation",
        "dosage": "Dosage",
        # Add the capitalized versions too for backward compatibility
        "Type": "Type",
        "Name": "Name",
        "Formulation": "Formulation",
        "Dosage": "Dosage"
    }
    
    db_field = field_mapping.get(field)
    if not db_field:
        logger.error(f"Unknown field: {field}")
        return table_query
    
    if db_field == "Type":
        cleaned_value = clean_type_value(value)
        if not cleaned_value:
            return table_query
            
        logger.info(f"Building type search query for: {repr(cleaned_value)}")
        logger.info(f"Using database field: {db_field}")
        
        # Simple case-insensitive search
        return table_query.ilike(db_field, f"%{cleaned_value}%")
    
    cleaned_value = clean_search_value(value)
    if not cleaned_value:
        return table_query
    
    if exact:
        return table_query.eq(db_field, cleaned_value)
    return table_query.ilike(db_field, f"%{cleaned_value}%")

def safe_get(data: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary that might be None."""
    if data is None:
        return default
    return data.get(key, default)

def apply_price_sort(query, sort_order: Optional[str] = None) -> Any:
    """Apply price sorting to the query based on branded medicine price."""
    if sort_order == "low_to_high":
        return query.order("Cost of branded", desc=False)
    elif sort_order == "high_to_low":
        return query.order("Cost of branded", desc=True)
    return query

# Cache for suggestions
def get_suggestions(field: str, query: Optional[str] = None) -> List[str]:
    """Get suggestions from the database"""
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
        logger.error(f"Error in get_suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        return []  # Return empty list instead of raising error

def create_medicine_from_db(data: Dict[str, Any]) -> Medicine:
    """Create a Medicine instance from database data with proper cost calculations."""
    try:
        # Ensure cost fields are Decimal
        if "Cost of branded" in data:
            data["Cost of branded"] = Decimal(str(data["Cost of branded"]))
        if "Cost of generic" in data:
            data["Cost of generic"] = Decimal(str(data["Cost of generic"]))
        
        # Calculate cost difference if not present
        if "Cost difference" not in data or data["Cost difference"] is None:
            if "Cost of branded" in data and "Cost of generic" in data:
                data["Cost difference"] = data["Cost of branded"] - data["Cost of generic"]
        
        # Calculate savings if not present
        if "Savings" not in data or data["Savings"] is None:
            if "Cost of branded" in data and "Cost of generic" in data:
                branded_price = float(data["Cost of branded"])
                generic_price = float(data["Cost of generic"])
                if branded_price > 0:
                    data["Savings"] = round(((branded_price - generic_price) / branded_price) * 100, 1)

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
async def get_suggestions_endpoint(
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

        # Get suggestions
        suggestions = get_suggestions(field, query)
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
        # Detailed request logging
        logger.info("=== New Search Request ===")
        logger.info(f"Full request object: {search_request}")
        logger.info(f"Raw request dict: {search_request.model_dump()}")
        
        if search_request.type:
            logger.info("=== Type Field Details ===")
            logger.info(f"Raw type value: {repr(search_request.type)}")
            logger.info(f"Type value length: {len(search_request.type)}")
            logger.info(f"Type value bytes: {[ord(c) for c in search_request.type]}")
            logger.info(f"Type value after strip: {repr(search_request.type.strip())}")
        
        logger.info(f"Name filter: {repr(search_request.name) if search_request.name else 'None'}")
        logger.info(f"Formulation filter: {repr(search_request.formulation) if search_request.formulation else 'None'}")
        logger.info(f"Dosage filter: {repr(search_request.dosage) if search_request.dosage else 'None'}")
        logger.info(f"Sort order: {repr(sort_order)}")
        
        # Get all available types for debugging
        all_types = get_all_types()
        logger.info(f"Available types in database: {all_types}")
        
        # Build the query
        query = supabase.table(MEDICINES_TABLE).select("*")
        logger.info("Created initial query")

        # Track if we're doing a type or dosage search
        is_type_or_dosage_search = (search_request.type or search_request.dosage) and not (search_request.name or search_request.formulation)

        try:
            if search_request.type:
                # Log the type search details
                logger.info("=== Type Search Details ===")
                logger.info(f"Original type value: {repr(search_request.type)}")
                cleaned_type = clean_type_value(search_request.type)
                logger.info(f"Cleaned type value: {repr(cleaned_type)}")
                
                # Build and log the type query
                query = build_search_query(query, "type", search_request.type)
                logger.info("Added type filter to query")
            
            if search_request.formulation:
                query = build_search_query(query, "formulation", search_request.formulation)
                logger.info(f"Added formulation filter")
            
            if search_request.name:
                query = build_search_query(query, "name", search_request.name)
                logger.info(f"Added name filter")
            
            if search_request.dosage:
                query = build_search_query(query, "dosage", search_request.dosage)
                logger.info(f"Added dosage filter")

            # Add sorting if specified
            if sort_order and sort_order != "none":
                query = query.order("Cost of branded", desc=(sort_order == "high_to_low"))
                logger.info(f"Added price sorting: {sort_order}")

            # Add limit for type/dosage searches
            if is_type_or_dosage_search:
                query = query.limit(MAX_RESULTS)
                logger.info(f"Added limit of {MAX_RESULTS} for type/dosage search")

            # Execute query and log the SQL
            logger.info("=== Executing Query ===")
            response = query.execute()
            logger.info(f"Query executed successfully")
            logger.info(f"Number of results: {len(response.data if response.data else [])}")

            if not response.data:
                logger.info("No medicines found")
                return SearchResponse(
                    exact_match=None,
                    similar_formulations=[],
                    Uses=None,
                    Side_Effects=None
                )
            
            medicines = response.data
            logger.info(f"Found {len(medicines)} medicines")

            # For type or dosage searches, return all results as similar formulations
            if is_type_or_dosage_search:
                logger.info("Processing type/dosage search results")
                processed_medicines = [create_medicine_from_db(m) for m in medicines]
                return SearchResponse(
                    exact_match=None,
                    similar_formulations=processed_medicines,
                    Uses=None,
                    Side_Effects=None
                )

            # Handle name/formulation searches
            exact_match_medicine: Optional[Medicine] = None
            exact_match_data: Optional[Dict[str, Any]] = None
            
            # Process exact matches and similar formulations
            if search_request.name:
                exact_matches = [m for m in medicines if m["Name"].lower() == search_request.name.lower()]
                if exact_matches:
                    exact_match_data = exact_matches[0]
                    if exact_match_data and "Name" in exact_match_data:
                        logger.info(f"Found exact match by name: {exact_match_data['Name']}")
            elif search_request.formulation:
                exact_matches = [m for m in medicines if m["Formulation"].lower() == search_request.formulation.lower()]
                if exact_matches:
                    exact_match_data = exact_matches[0]
                    if exact_match_data and "Formulation" in exact_match_data:
                        logger.info(f"Found exact match by formulation: {exact_match_data['Formulation']}")

            # Convert data to Medicine objects
            try:
                if exact_match_data:
                    exact_match_medicine = create_medicine_from_db(exact_match_data)
                processed_medicines = [create_medicine_from_db(m) for m in medicines if m != exact_match_data]
                logger.info(f"Processed {len(processed_medicines)} medicines")
            except Exception as e:
                logger.error(f"Error converting medicines data: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error processing search results: {str(e)}"
                )

            return SearchResponse(
                exact_match=exact_match_medicine,
                similar_formulations=processed_medicines,
                Uses=safe_get(exact_match_data, "Uses"),
                Side_Effects=safe_get(exact_match_data, "Side Effects")
            )

        except Exception as e:
            logger.error("=== Error in Search Process ===")
            logger.error(f"Error details: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing search: {str(e)}"
            )

    except Exception as e:
        logger.error("=== Unexpected Error ===")
        logger.error(f"Error details: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"An unexpected error occurred: {str(e)}"}
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

@router.get("/medicines/by_type", response_model=List[Medicine])
async def get_medicines_by_type(
    type: str = Query(..., description="Medicine type to filter by"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of results to return"),
    sort_order: Optional[Literal["none", "low_to_high", "high_to_low"]] = Query(
        default="none",
        description="Sort order for medicine prices"
    )
):
    """
    Get medicines by type with a simple, focused query.
    This endpoint is optimized for type-based browsing and filtering.
    """
    try:
        logger.info("=== Medicine Type Filter Request ===")
        logger.info(f"Requested type: {repr(type)}")
        logger.info(f"Sort order: {sort_order}")
        
        # Clean and validate the type
        cleaned_type = clean_type_value(type)
        logger.info(f"Cleaned type: {repr(cleaned_type)}")
        
        # Get all available types for comparison
        all_types = get_all_types()
        logger.info(f"Available types: {all_types}")
        
        # Build a simple query
        query = (
            supabase.table(MEDICINES_TABLE)
            .select("*")
            .ilike("Type", f"%{cleaned_type}%")
            .limit(limit)
        )
        
        # Apply sorting if specified
        if sort_order and sort_order != "none":
            query = apply_price_sort(query, sort_order)
            logger.info(f"Applied price sorting: {sort_order}")
        
        logger.info("Executing type filter query...")
        response = query.execute()
        logger.info(f"Query executed successfully")
        
        if not response.data:
            logger.info(f"No medicines found for type: {repr(cleaned_type)}")
            return []
            
        # Convert to Medicine objects
        medicines = []
        for item in response.data:
            try:
                medicine = create_medicine_from_db(item)
                medicines.append(medicine)
            except Exception as e:
                logger.error(f"Error converting medicine data: {str(e)}")
                logger.error(f"Problematic data: {item}")
                continue
        
        logger.info(f"Found {len(medicines)} medicines for type: {repr(cleaned_type)}")
        return medicines
        
    except Exception as e:
        logger.error("=== Error in Type Filter ===")
        logger.error(f"Error details: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error filtering medicines by type: {str(e)}"
        ) 