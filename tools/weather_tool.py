import logging
from typing import Dict, Any, List

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from langchain_core.tools import tool

from .tool_schemas import WeatherInput

# --- Configuration & Helpers (No changes) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

def get_wmo_description(code: float) -> str:
    """Helper function to safely get the weather description from a WMO code that might be NaN."""
    if np.isnan(code):
        return "Not available"
    return WMO_CODES.get(int(code), "Unknown")

CURRENT_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "wind_speed_10m", "soil_moisture_0_to_1cm"]
DAILY_VARS = ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_probability_max", "et0_fao_evapotranspiration", "wind_speed_10m_max"]
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation_probability", "weather_code", "wind_speed_10m", "soil_moisture_0_to_1cm", "et0_fao_evapotranspiration"]


@tool
def weather_tool(
    latitude: float, longitude: float, forecast_days: int = 7, past_days: int = 0, include_hourly: bool = False
) -> Dict[str, Any]:
    """
    Retrieves a comprehensive weather report, including current conditions, historical data, and future forecasts for a specific geographical location.

    This is the primary tool for all weather-related queries. It provides key agronomic data points such as temperature, precipitation, humidity, wind, soil moisture, and reference evapotranspiration (ET₀), which is crucial for irrigation planning.

    The tool is highly flexible. The agent MUST specify the desired time window using the 'forecast_days' and 'past_days' parameters to ensure efficient data retrieval.

    It returns a structured JSON object containing a `summary`, `current_conditions`, `daily_data`, and optionally `hourly_data`.

    ARGUMENTS:
    - `latitude` (float): The geographical latitude for the location.
    - `longitude` (float): The geographical longitude for the location.
    - `forecast_days` (int): The number of future days to include in the forecast.
        - Must be between 0 and 16.
        - `forecast_days=1` will retrieve ONLY today's forecast.
        - `forecast_days=0` will retrieve no future data.
    - `past_days` (int): The number of historical days to retrieve data for.
        - Must be between 0 and 92.
        - `past_days=1` will retrieve ONLY yesterday's data.
        - `past_days=0` will retrieve no historical data.
    - `include_hourly` (bool): Set to True to include a detailed, hour-by-hour data breakdown.
        - Defaults to False.
        - Use this ONLY when the user asks a specific question about hourly conditions (e.g., "Will it be windy this afternoon?") as the output is very verbose.

    EXAMPLE USAGE:
    1. For a user asking "What was the weather like last week?":
       weather_tool(latitude=41.59, longitude=-93.62, forecast_days=0, past_days=7)

    2. For a user asking "What is the forecast for today and tomorrow?":
       weather_tool(latitude=36.74, longitude=-119.78, forecast_days=2)

    3. For a user asking "How much did it rain yesterday?":
        weather_tool(latitude=41.59, longitude=-93.62, forecast_days=0, past_days=1)
    """
    try:
        input_data = WeatherInput(
            latitude=latitude, longitude=longitude, forecast_days=forecast_days,
            past_days=past_days, include_hourly=include_hourly
        )
        cache_session = requests_cache.CachedSession('.cache', expire_after=1800)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        params = {
            "latitude": input_data.latitude, "longitude": input_data.longitude, "timezone": "auto",
            "current": CURRENT_VARS, "daily": DAILY_VARS
        }
        
        # API requires at least 1 forecast day if past_days is not set.
        # We handle the slicing later to match the user request exactly.
        params["forecast_days"] = max(1, input_data.forecast_days)
        if input_data.past_days > 0:
            params["past_days"] = input_data.past_days
        if input_data.include_hourly:
            params["hourly"] = HOURLY_VARS

        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]

        output_json: Dict[str, Any] = {"location": {"latitude": latitude, "longitude": longitude}}
        
        # Process CURRENT data
        current = response.Current()
        output_json["current_conditions"] = {
            "time": pd.to_datetime(current.Time(), unit="s", utc=True).isoformat(),
            "temperature_celsius": round(float(current.Variables(0).Value()), 1),
            "relative_humidity_percent": round(float(current.Variables(1).Value()), 1),
            "precipitation_mm": round(float(current.Variables(2).Value()), 2),
            "weather": get_wmo_description(current.Variables(3).Value()),
            "wind_speed_kmh": round(float(current.Variables(4).Value()), 1),
            "soil_moisture_0_1cm_m3m3": round(float(current.Variables(5).Value()), 3),
        }
        
        # Process DAILY data
        daily = response.Daily()
        daily_dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
        
        all_daily_data = []
        for i in range(len(daily_dates)):
            day = {
                "date": daily_dates[i].strftime('%Y-%m-%d'),
                "weather": get_wmo_description(daily.Variables(0).ValuesAsNumpy()[i]),
                "temp_max_celsius": round(float(daily.Variables(1).ValuesAsNumpy()[i]), 1),
                "temp_min_celsius": round(float(daily.Variables(2).ValuesAsNumpy()[i]), 1),
                "precipitation_total_mm": round(float(daily.Variables(3).ValuesAsNumpy()[i]), 2),
                "precipitation_probability_max_percent": int(daily.Variables(4).ValuesAsNumpy()[i]),
                "et0_fao_evapotranspiration_mm": round(float(daily.Variables(5).ValuesAsNumpy()[i]), 2),
                "wind_speed_max_kmh": round(float(daily.Variables(6).ValuesAsNumpy()[i]), 1),
            }
            all_daily_data.append(day)

        total_days_requested = input_data.past_days + input_data.forecast_days
        if total_days_requested > 0:
            output_json["daily_data"] = all_daily_data[:total_days_requested]
        else:
            output_json["daily_data"] = []

        if input_data.past_days > 14 and output_json.get("daily_data"):
            df = pd.DataFrame(output_json["daily_data"])
            period_summary = {
                "period_start_date": df['date'].min(),
                "period_end_date": df['date'].max(),
                "average_max_temp_celsius": float(round(df['temp_max_celsius'].mean(), 1)),
                "average_min_temp_celsius": float(round(df['temp_min_celsius'].mean(), 1)),
                "total_precipitation_mm": float(round(df['precipitation_total_mm'].sum(), 2)),
                "average_et0_mm_per_day": float(round(df['et0_fao_evapotranspiration_mm'].mean(), 2)),
                "days_with_rain": int(df[df['precipitation_total_mm'] > 0].shape[0]),
            }
            output_json["period_summary"] = period_summary

        # Process HOURLY data (if requested)
        if input_data.include_hourly:
            # Applying the same fixes for hourly data
            hourly = response.Hourly()
            hourly_dates = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
            all_hourly_data = []
            for i in range(len(hourly_dates)):
                hour = {
                    "time": hourly_dates[i].isoformat(),
                    "weather": get_wmo_description(hourly.Variables(3).ValuesAsNumpy()[i]),
                    "temperature_celsius": round(float(hourly.Variables(0).ValuesAsNumpy()[i], 1)),
                    "relative_humidity_percent": int(hourly.Variables(1).ValuesAsNumpy()[i]),
                    "precipitation_probability_percent": int(hourly.Variables(2).ValuesAsNumpy()[i]),
                    "wind_speed_kmh": round(float(hourly.Variables(4).ValuesAsNumpy()[i], 1)),
                    "soil_moisture_0_1cm_m3m3": round(float(hourly.Variables(5).ValuesAsNumpy()[i], 3)),
                    "et0_fao_evapotranspiration_mm": round(float(hourly.Variables(6).ValuesAsNumpy()[i], 2)),
                }
                all_hourly_data.append(hour)
            total_hours_requested = (input_data.past_days + input_data.forecast_days) * 24
            output_json["hourly_data"] = all_hourly_data[:total_hours_requested]


        # Create a more intelligent summary
        summary_parts: List[str] = [f"Current: {output_json['current_conditions']['temperature_celsius']}°C, {output_json['current_conditions']['weather']}."]
        if "period_summary" in output_json:
             summary_parts.append(f"Summary for the last {input_data.past_days} days: Avg Max Temp {output_json['period_summary']['average_max_temp_celsius']}°C, Total Precip {output_json['period_summary']['total_precipitation_mm']} mm.")
        elif input_data.past_days > 0:
            past_data = output_json.get("daily_data", [])[:input_data.past_days]
            total_precip = sum(d['precipitation_total_mm'] for d in past_data)
            summary_parts.append(f"In the last {input_data.past_days} days, total precipitation was {total_precip:.2f} mm.")
        
        if input_data.forecast_days > 0:
            today_index = input_data.past_days
            if today_index < len(all_daily_data):
                today_forecast = all_daily_data[today_index]
                summary_parts.append(f"Today's forecast: Max {today_forecast['temp_max_celsius']}°C, {today_forecast['weather']}.")

        output_json["summary"] = " ".join(summary_parts)
        return output_json

    except Exception as e:
        logging.exception(f"Dynamic weather tool failed for {latitude}, {longitude}")
        return {"error": f"An unexpected error occurred in the weather tool: {e}"}