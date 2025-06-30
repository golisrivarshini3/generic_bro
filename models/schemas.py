"""
Pydantic models for request and response data validation.
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, List, Dict, Any
from decimal import Decimal

class MedicineSearchRequest(BaseModel):
    """
    Request model for medicine search endpoint.
    At least one search field must be provided.
    """
    name: Optional[str] = Field(None, alias="Name", description="Brand or generic name of the medicine")
    formulation: Optional[str] = Field(None, alias="Formulation", description="Medicine formulation (e.g., Glimepiride 1mg)")
    type: Optional[str] = Field(None, alias="Type", description="Medicine type (e.g., A-Anti Diabetic)")
    dosage: Optional[str] = Field(None, alias="Dosage", description="Medicine dosage (e.g., 1mg)")

    @validator('name', 'formulation', 'type', 'dosage')
    def validate_string_fields(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v

    @validator('name', 'formulation', 'type', 'dosage')
    def validate_at_least_one_field(cls, v, values: Dict[str, Any]):
        """Ensure at least one search field is provided"""
        # Skip validation for the first field
        if not values:
            return v
            
        # Check if any field has a value
        if v is not None:
            return v
            
        for field_value in values.values():
            if field_value is not None:
                return v
                
        raise ValueError("At least one search field must be provided")

    model_config = ConfigDict(populate_by_name=True)

class Medicine(BaseModel):
    """
    Base model for medicine data.
    """
    name: str = Field(..., alias="Name", description="Name of the medicine (e.g., TAB GLIMEPRIDE)")
    dosage: str = Field(..., alias="Dosage", description="Medicine dosage (e.g., 1mg)")
    formulation: str = Field(..., alias="Formulation", description="Medicine formulation (e.g., Glimepiride 1mg)")
    cost_of_branded: Decimal = Field(..., alias="Cost of branded", description="Price of branded version")
    cost_of_generic: Decimal = Field(..., alias="Cost of generic", description="Price of generic version")
    cost_difference: Optional[Decimal] = Field(None, alias="Cost difference", description="Difference between branded and generic cost")
    savings: Optional[float] = Field(None, alias="Savings", description="Percentage savings when choosing generic over branded")
    type: str = Field(..., alias="Type", description="Type of medicine (e.g., A-Anti Diabetic)")
    uses: str = Field(..., alias="Uses", description="Uses and indications")
    side_effects: str = Field(..., alias="Side Effects", description="Known side effects")

    @validator('cost_of_branded', 'cost_of_generic', 'cost_difference')
    def validate_prices(cls, v):
        if v is not None and v < 0:
            raise ValueError("Price cannot be negative")
        return v

    @validator('savings')
    def calculate_savings(cls, v, values):
        """Calculate savings percentage if not provided"""
        if v is None and 'cost_of_branded' in values and 'cost_of_generic' in values:
            branded_price = float(values['cost_of_branded'])
            generic_price = float(values['cost_of_generic'])
            if branded_price > 0:
                savings = ((branded_price - generic_price) / branded_price) * 100
                return round(savings, 1)
        return v

    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True
    )

class SearchResponse(BaseModel):
    """
    Response model for medicine search results.
    """
    exact_match: Optional[Medicine] = Field(None, description="Exact match if found")
    similar_formulations: List[Medicine] = Field(
        default_factory=list,
        description="Other medicines with the same formulation"
    )
    Uses: Optional[str] = Field(None, description="Uses of the exact match medicine")
    Side_Effects: Optional[str] = Field(None, description="Side effects of the exact match medicine")

    model_config = ConfigDict(populate_by_name=True)

class AutocompleteSuggestion(BaseModel):
    """
    Model for autocomplete suggestions.
    """
    value: str = Field(..., description="The suggested value")
    field_type: str = Field(
        ...,
        description="Type of the suggestion (Name, Formulation, Type, Dosage)"
    )

    @validator('field_type')
    def validate_field_type(cls, v):
        valid_types = {'Name', 'Formulation', 'Type', 'Dosage'}
        if v not in valid_types:
            raise ValueError(f"field_type must be one of: {', '.join(valid_types)}")
        return v

class AutocompleteResponse(BaseModel):
    """
    Response model for autocomplete suggestions.
    """
    suggestions: List[AutocompleteSuggestion] = Field(
        ...,
        description="List of autocomplete suggestions"
    ) 