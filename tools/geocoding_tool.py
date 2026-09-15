import os
import requests
from typing import Dict, Any
from langchain_core.tools import tool
from dotenv import load_dotenv

from .tool_schemas import GeocodingInput

load_dotenv()

@tool
def geocode_location(location_name: str) -> Dict[str, Any]:
    """
    Retrieves the geographical coordinates (latitude and longitude) for a given location name.

    This tool is essential for converting user-provided place names (like cities, regions, or countries)
    into the precise latitude and longitude required by other tools such as `weather_tool` and `geospatial_tool`.

    ARGUMENTS:
    - `location_name` (str): The name of the location to geocode. It should be as specific as possible for best results.

    Usage Guidance for the Agent:
    1.  **Be Specific:** Always prefer a format like 'City, State, Country'. For example, 'Fresno, California, US' is much better than just 'Fresno'.
    2.  **Handle Vague Regions:** If a user provides a non-specific region (e.g., 'central Iowa', 'the Midwest'), you MUST reformat it to be a more general, officially recognized area for the API call. For 'central Iowa', the best `location_name` to use is 'Iowa, US'. For 'Bekaa Valley', use 'Bekaa Valley, Lebanon'.
    3.  **Add Context:** If a location could be in multiple countries, add the country to resolve ambiguity. For example, use 'Paris, France' instead of 'Paris'.

    RETURNS:
    A dictionary containing the 'latitude', 'longitude', and a formatted 'full_name' of the found location, or an error if not found.
    Example: {"latitude": 36.7394, "longitude": -119.785, "full_name": "Fresno, California, US"}
    """

    try:
        # Validate the input using the Pydantic schema
        _ = GeocodingInput(location_name=location_name)

        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return {"error": "Geocoding failed: OPENWEATHER_API_KEY is not set."}

        # Use request parameters so special characters in place names are encoded safely.
        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {"q": location_name, "limit": 1, "appid": api_key}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        if not data:
            return {"error": f"Could not find coordinates for the location: '{location_name}'."}

        # Extract the relevant information from the first result
        location = data[0]
        lat = location.get("lat")
        lon = location.get("lon")
        name = location.get("name", "")
        country = location.get("country", "")
        state = location.get("state", "")

        full_name = f"{name}, {state}, {country}".replace(", ,", ",").strip(", ")

        return {
            "latitude": lat,
            "longitude": lon,
            "full_name": full_name
        }

    except Exception as e:
        return {"error": f"An unexpected error occurred during geocoding: {e}"}
