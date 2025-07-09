from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from db.supabase_client import supabase
import logging
from decimal import Decimal
from math import radians, sin, cos, sqrt, atan2

router = APIRouter()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the table name as a constant
PHARMACIES_TABLE = 'pharmacies'

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth."""
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

@router.get("/search/by-pincode")
async def search_by_pincode(
    pincode: str = Query(..., min_length=6, max_length=6, description="6-digit PIN code")
):
    """Search pharmacies by PIN code."""
    try:
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"Kendra Code","Name","Contact","State Name","District Name","Pin Code","Address","Latitude","Longitude"')\
            .eq('"Pin Code"', pincode)\
            .execute()
            
        if not response.data:
            return {"pharmacies": [], "message": f"No pharmacies found for PIN code {pincode}"}
            
        return {"pharmacies": response.data}
    except Exception as e:
        logger.error(f"Error searching by PIN code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/by-district")
async def search_by_district(
    district_name: str = Query(..., min_length=1, description="District name")
):
    """Search pharmacies by district name."""
    try:
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"Kendra Code","Name","Contact","State Name","District Name","Pin Code","Address","Latitude","Longitude"')\
            .ilike('"District Name"', f"%{district_name}%")\
            .execute()
            
        if not response.data:
            return {"pharmacies": [], "message": f"No pharmacies found in district {district_name}"}
            
        return {"pharmacies": response.data}
    except Exception as e:
        logger.error(f"Error searching by district: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/by-state")
async def search_by_state(
    state_name: str = Query(..., min_length=1, description="State name")
):
    """Search pharmacies by state name. Returns maximum 15 results."""
    try:
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"Kendra Code","Name","Contact","State Name","District Name","Pin Code","Address","Latitude","Longitude"')\
            .ilike('"State Name"', f"%{state_name}%")\
            .limit(15)\
            .execute()
            
        if not response.data:
            return {"pharmacies": [], "message": f"No pharmacies found in state {state_name}"}
            
        return {"pharmacies": response.data}
    except Exception as e:
        logger.error(f"Error searching by state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/by-area")
async def search_by_area(
    area: str = Query(..., min_length=1, description="Area or locality name")
):
    """Search pharmacies by area/locality."""
    try:
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"Kendra Code","Name","Contact","State Name","District Name","Pin Code","Address","Latitude","Longitude"')\
            .ilike('"Address"', f"%{area}%")\
            .execute()
            
        if not response.data:
            return {"pharmacies": [], "message": f"No pharmacies found in area {area}"}
            
        return {"pharmacies": response.data}
    except Exception as e:
        logger.error(f"Error searching by area: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/by-coordinates")
async def search_by_coordinates(
    latitude: float = Query(..., description="Latitude of the user's location"),
    longitude: float = Query(..., description="Longitude of the user's location"),
    radius_km: float = Query(default=5.0, gt=0, le=50, description="Search radius in kilometers")
):
    """Search pharmacies near given coordinates within specified radius."""
    try:
        # First, get all pharmacies
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"Kendra Code","Name","Contact","State Name","District Name","Pin Code","Address","Latitude","Longitude"')\
            .execute()
            
        if not response.data:
            return {"pharmacies": [], "message": "No pharmacies found with valid coordinates"}
            
        # Filter pharmacies within radius
        nearby_pharmacies = []
        for pharmacy in response.data:
            if pharmacy.get('Latitude') and pharmacy.get('Longitude'):
                try:
                    lat = float(pharmacy['Latitude'])
                    lon = float(pharmacy['Longitude'])
                    distance = haversine_distance(latitude, longitude, lat, lon)
                    if distance <= radius_km:
                        pharmacy['distance'] = round(distance, 2)
                        nearby_pharmacies.append(pharmacy)
                except (ValueError, TypeError):
                    # Skip pharmacies with invalid coordinates
                    continue
        
        # Sort by distance
        nearby_pharmacies.sort(key=lambda x: x['distance'])
            
        if not nearby_pharmacies:
            return {
                "pharmacies": [],
                "message": f"No pharmacies found within {radius_km}km of your location"
            }
            
        return {"pharmacies": nearby_pharmacies}
    except Exception as e:
        logger.error(f"Error searching by coordinates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suggestions/pincode")
async def get_pincode_suggestions(
    query: Optional[str] = Query(None, min_length=1, description="Partial PIN code")
):
    """Get unique PIN code suggestions."""
    try:
        base_query = supabase.table(PHARMACIES_TABLE)\
            .select('"Pin Code"')
            
        if query:
            base_query = base_query.ilike('"Pin Code"', f"{query}%")
            
        response = base_query.execute()
        
        suggestions = list(set(item['Pin Code'] for item in response.data if item.get('Pin Code')))
        suggestions.sort()
        
        return {"suggestions": suggestions[:10]}  # Limit to 10 suggestions
    except Exception as e:
        logger.error(f"Error getting PIN code suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suggestions/{field}")
async def get_field_suggestions(
    field: str,
    query: Optional[str] = Query(None, min_length=1),
):
    """Get suggestions for state, district, or area."""
    field_mapping = {
        "state": "State Name",
        "district": "District Name",
        "area": "Address"
    }
    
    if field not in field_mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid field. Must be one of: {', '.join(field_mapping.keys())}"
        )
        
    try:
        db_field = field_mapping[field]
        base_query = supabase.table(PHARMACIES_TABLE)\
            .select(f'"{db_field}"')
            
        if query:
            base_query = base_query.ilike(f'"{db_field}"', f"%{query}%")
            
        response = base_query.execute()
        
        suggestions = list(set(item[db_field] for item in response.data if item.get(db_field)))
        suggestions.sort()
        
        return {"suggestions": suggestions[:10]}  # Limit to 10 suggestions
    except Exception as e:
        logger.error(f"Error getting {field} suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 

@router.get("/suggest/pincode")
async def suggest_pincode(
    input: str = Query(..., description="PIN code prefix to search for")
):
    """Get PIN code suggestions that start with the input."""
    try:
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"Pin Code"')\
            .ilike('"Pin Code"', f"{input}%")\
            .execute()
            
        suggestions = list(set(str(item['Pin Code']) for item in response.data if item.get('Pin Code')))
        suggestions.sort()
        
        return {"suggestions": suggestions[:10]}
    except Exception as e:
        logger.error(f"Error getting PIN code suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suggest/district")
async def suggest_district(
    input: str = Query(..., description="District name to search for")
):
    """Get district suggestions that contain the input."""
    try:
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"District Name"')\
            .ilike('"District Name"', f"%{input}%")\
            .execute()
            
        suggestions = list(set(item['District Name'] for item in response.data if item.get('District Name')))
        suggestions.sort()
        
        return {"suggestions": suggestions[:10]}
    except Exception as e:
        logger.error(f"Error getting district suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suggest/state")
async def suggest_state(
    input: str = Query(..., description="State name to search for")
):
    """Get state suggestions that contain the input."""
    try:
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"State Name"')\
            .ilike('"State Name"', f"%{input}%")\
            .execute()
            
        suggestions = list(set(item['State Name'] for item in response.data if item.get('State Name')))
        suggestions.sort()
        
        return {"suggestions": suggestions[:10]}
    except Exception as e:
        logger.error(f"Error getting state suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suggest/area")
async def suggest_area(
    input: str = Query(..., description="Area/locality to search for in address")
):
    """Get area suggestions from addresses that contain the input."""
    try:
        response = supabase.table(PHARMACIES_TABLE)\
            .select('"Address"')\
            .ilike('"Address"', f"%{input}%")\
            .execute()
            
        suggestions = list(set(item['Address'] for item in response.data if item.get('Address')))
        suggestions.sort()
        
        return {"suggestions": suggestions[:10]}
    except Exception as e:
        logger.error(f"Error getting area suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 