import os
import logging
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

from sentinelhub import (
    SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig, bbox_to_dimensions
)
from utm.error import OutOfRangeError
from langchain_core.tools import tool
from .tool_schemas import GeospatialInput

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class GeospatialAnalysisTool:
    def __init__(self):
        # ... (init method is correct and remains the same)
        self.config = SHConfig()
        self.config.sh_client_id = os.getenv("SENTINEL_CLIENT_ID")
        self.config.sh_client_secret = os.getenv("SENTINEL_CLIENT_SECRET")
        if not self.config.sh_client_id or not self.config.sh_client_secret:
            raise ValueError("Missing Sentinel Hub credentials")

    def _create_bounding_box(self, latitude: float, longitude: float, radius_meters: int) -> BBox:
        # ... (this method is correct and remains the same)
        lat_degree_per_meter = 1 / 111320
        lon_degree_per_meter = 1 / (111320 * np.cos(np.radians(latitude)))
        lat_radius = radius_meters * lat_degree_per_meter
        lon_radius = radius_meters * lon_degree_per_meter
        return BBox([
            longitude - lon_radius, latitude - lat_radius,
            longitude + lon_radius, latitude + lat_radius
        ], crs=CRS.WGS84)

    def _calculate_indices(self, data: np.ndarray) -> Dict[str, Any]:
        scl_band = data[:, :, 7].astype(int)
        water_pixels = np.sum(scl_band == 6)
        total_pixels = data.shape[0] * data.shape[1]
        
        if total_pixels > 0 and (water_pixels / total_pixels) > 0.9:
            return {"error": "The specified area consists primarily of water, not vegetation."}

        epsilon = 1e-8
        no_data_mask = data[:, :, 0] == -999
        
        blue, green, red, red_edge1, red_edge2, nir, swir = [np.ma.masked_array(data[:, :, i], mask=no_data_mask) for i in range(7)]
        
        ndvi = (nir - red) / (nir + red + epsilon)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon)
        ndwi = (green - nir) / (green + nir + epsilon)
        gndvi = (nir - green) / (nir + green + epsilon)
        savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5
        msi = swir / (nir + epsilon)
        mtci = (red_edge2 - red_edge1) / (red_edge1 - red + epsilon)
        
        mean_mtci = float(np.ma.mean(mtci))
        capped_mtci = min(max(mean_mtci, 0), 10)

        indices = {
            "NDVI": round(float(np.ma.mean(ndvi)), 3), "EVI": round(float(np.ma.mean(evi)), 3),
            "NDWI": round(float(np.ma.mean(ndwi)), 3), "GNDVI": round(float(np.ma.mean(gndvi)), 3),
            "SAVI": round(float(np.ma.mean(savi)), 3), "MSI": round(float(np.ma.mean(msi)), 3),
            "MTCI": round(capped_mtci, 3),
        }
        
        interpretation = {
            "vegetation_vigor": "High" if indices["NDVI"] > 0.6 else "Moderate" if indices["NDVI"] > 0.25 else "Low",
            "water_stress": "Potential Stress" if indices["MSI"] > 0.9 else "Adequate Moisture",
            "chlorophyll_content": "High" if indices["MTCI"] > 4.0 else "Moderate" if indices["MTCI"] > 2.0 else "Low",
        }
        
        return {"indices": indices, "interpretation": interpretation}

    def execute(self, latitude: float, longitude: float, radius_meters: int, date_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = {}
        
        try:
            aoi_bbox = self._create_bounding_box(latitude, longitude, radius_meters)
            aoi_size = bbox_to_dimensions(aoi_bbox, resolution=10)
        except OutOfRangeError as e:
            logging.warning(f"Coordinate error for {latitude}, {longitude}: {e}")
            return {"error": f"Invalid coordinates provided. Latitude must be between -80 and 84 degrees. Error: {e}"}

        evalscript = """
        //VERSION=3
        function setup() { return { input: ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "SCL"], output: { bands: 8, sampleType: "FLOAT32" } }; }
        function evaluatePixel(sample) {
            if (sample.SCL === 3 || sample.SCL === 7 || sample.SCL === 8 || sample.SCL === 9 || sample.SCL === 10) { return Array(8).fill(-999); }
            return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B08, sample.B11, sample.SCL];
        }
        """

        for period in date_periods:
            period_name = period['period_name']
            time_interval = (datetime.strptime(period['start_date'], '%Y-%m-%d'), datetime.strptime(period['end_date'], '%Y-%m-%d'))
            
            try:
                request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A, time_interval=time_interval,
                        mosaicking_order='leastCC', maxcc=0.3
                    )],
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    bbox=aoi_bbox, size=aoi_size, config=self.config,
                )
                data_list = request.get_data()

                if not data_list or (data_list[0][:, :, 0] == -999).all():
                    results[period_name] = {"error": f"No usable satellite data found for the period {period['start_date']} to {period['end_date']}."}
                    continue

                # --- THIS IS THE FINAL FIX ---
                # Calculate the result for the period
                period_result = self._calculate_indices(data_list[0])
                
                # If the result for this period is an error (e.g., it's water), and it's the only period,
                # then return the error as the top-level result.
                if "error" in period_result and len(date_periods) == 1:
                    return period_result
                
                # Otherwise, assign the result (or error) to its period name
                results[period_name] = period_result
                # --- END OF FIX ---

            except Exception as e:
                logging.exception(f"Geospatial analysis failed for period {period_name}")
                results[period_name] = {"error": f"API error during analysis for period {period_name}: {str(e)}"}
        
        return results


@tool
def geospatial_tool(
    latitude: float, longitude: float, radius_meters: int, date_periods: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyzes and compares advanced agricultural metrics from Sentinel-2 satellite imagery for a specific location across multiple time periods.

    This is a powerful and specialized tool for detailed temporal analysis of crop health. It should be used when a user asks to compare field conditions over time, such as between different growing stages, or before and after an event.

    It returns a detailed JSON object where each key is a `period_name` you provide. The value for each key contains the calculated raw `indices` and a high-level `interpretation`.

    ARGUMENTS:
    - `latitude` (float): The geographical latitude for the exact center of the analysis area. Must be between -80 and 84.
    - `longitude` (float): The geographical longitude for the exact center of the analysis area.
    - `radius_meters` (int): The radius of the circular area to analyze, in meters. Must be between 50 and 2000. A typical small field might be 200-500 meters, while a larger commercial field might be 1000-2000 meters.
    - `date_periods` (List[Dict[str, Any]]): A list of one or more time periods to analyze. Each period MUST be a dictionary with three keys:
        - `period_name` (str): A unique, descriptive name for this time window (e.g., 'early_vegetative_stage', 'post_hail_damage', 'last_30_days'). This name will be used as the key in the JSON output.
        - `start_date` (str): The beginning of the time window in 'YYYY-MM-DD' format.
        - `end_date` (str): The end of the time window in 'YYYY-MM-DD' format.

    EXAMPLE USAGE for a user asking to compare a field's health between May and August:
    geospatial_tool(
        latitude=41.59,
        longitude=-93.62,
        radius_meters=500,
        date_periods=[
            {"period_name": "may_2025", "start_date": "2025-05-01", "end_date": "2025-05-31"},
            {"period_name": "august_2025", "start_date": "2025-08-01", "end_date": "2025-08-31"}
        ]
    )
    """
    try:
        validated_input = GeospatialInput(
            latitude=latitude, longitude=longitude,
            radius_meters=radius_meters, date_periods=date_periods
        )
    except Exception as e:
        return {"error": f"Invalid geospatial input: {e}"}
        
    tool_instance = GeospatialAnalysisTool()
    return tool_instance.execute(
        validated_input.latitude, validated_input.longitude,
        validated_input.radius_meters, [p.dict() for p in validated_input.date_periods]
    )