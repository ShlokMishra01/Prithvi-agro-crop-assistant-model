from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any
from datetime import datetime

class VisionInput(BaseModel):
    image_path: str = Field(..., description="Local path to the image to analyze")

    @field_validator("image_path")
    def validate_image_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Image path cannot be blank.")
        if not value.lower().endswith((".jpg", ".jpeg", ".png")):
            raise ValueError("Image must be a JPG, JPEG, or PNG file.")
        return value

class WeatherInput(BaseModel):
    """Input schema for the weather tool."""
    latitude: float = Field(..., description="Latitude in decimal degrees.")
    longitude: float = Field(..., description="Longitude in decimal degrees.")
    forecast_days: int = Field(
        default=7,
        ge=0,
        le=16,
        description="Number of forecast days to retrieve (0-16). A value of 1 gets today's forecast."
    )
    past_days: int = Field(
        default=0,
        ge=0,
        le=92,
        description="Number of past days to retrieve (0-92). A value of 1 gets yesterday's data."
    )
    include_hourly: bool = Field(
        default=False,
        description="Set to True to include a detailed hourly forecast. Defaults to False as it's verbose."
    )

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query text")

    @field_validator("query")
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Search query cannot be blank.")
        return value

class DatePeriod(BaseModel):
    """A single period for temporal analysis."""
    period_name: str = Field(..., description="A unique name for this time period (e.g., 'vegetative_stage', 'last_30_days').")
    start_date: str = Field(..., description="The start date in 'YYYY-MM-DD' format.")
    end_date: str = Field(..., description="The end date in 'YYYY-MM-DD' format.")

    @field_validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format.")
        return v

    @model_validator(mode="after")
    def validate_date_order(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date.")
        return self

class GeospatialInput(BaseModel):
    latitude: float = Field(..., description="Latitude for the center of the analysis area.")
    longitude: float = Field(..., description="Longitude for the center of the analysis area.")
    radius_meters: int = Field(default=200, ge=50, le=2000, description="Radius in meters to analyze (50-2000).")
    date_periods: List[DatePeriod] = Field(..., description="A list of time periods to analyze and compare.")

class GeocodingInput(BaseModel):
    """Input schema for the geocoding tool."""
    location_name: str = Field(..., description="The city, region, or place name to find coordinates for (e.g., 'Bekaa Valley, Lebanon', 'central Iowa').")

    @field_validator("location_name")
    def validate_location_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Location name cannot be blank.")
        return value
