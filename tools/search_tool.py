import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_core.tools import tool
from .tool_schemas import SearchInput

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# Shared trusted domains for scientific search. This list is now passed directly
# to the search API instead of being used for manual query formatting.
TRUSTED_DOMAINS: List[str] = [
    "fao.org",          # Food and Agriculture Organization
    "cgiar.org",        # Consultative Group on International Agricultural Research
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    # Add any specific .edu or .gov domains if you want to prioritize them,
    # though searching all of them can be too broad.
    # For example: "extension.purdue.edu", "ag.iastate.edu"
]


@tool
def general_web_search(query: str) -> Dict[str, Any]:
    """
    Searches the general web for recent or broad information.
    Returns a structured JSON object containing a list of search results.
    Each result includes a title, url, content snippet, and relevance score.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY environment variable is not set."}
    
    # Validate inputs via Pydantic model
    try:
        _ = SearchInput(query=query)
    except Exception as e:
        return {"error": f"Invalid search input: {e}"}

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            topic="general" # Explicitly set topic
        )
        
        results = response.get("results", [])
        if not results:
            return {"status": "No results found.", "results": []}

        # Format results into a clean list of dictionaries
        structured_results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
                "score": r.get("score"),
            }
            for r in results
        ]
        
        return {
            "status": f"Successfully found {len(structured_results)} results.",
            "results": structured_results
        }

    except Exception as e:
        logging.exception(f"General web search failed for query: '{query}'")
        return {"error": f"An unexpected error occurred during the general web search: {e}"}


@tool
def scientific_search(query: str) -> Dict[str, Any]:
    """
    Searches for high-trust scientific and agricultural information from a curated list of domains.
    Returns a structured JSON object containing a synthesized answer and a list of supporting sources.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY environment variable is not set."}

    # Validate inputs via Pydantic model
    try:
        _ = SearchInput(query=query)
    except Exception as e:
        return {"error": f"Invalid search input: {e}"}

    try:
        client = TavilyClient(api_key=api_key)
        # Use the `include_domains` parameter for precise, API-level filtering.
        # This is more effective than manually adding 'site:' operators to the query.
        response = client.search(
            query=query,
            search_depth="advanced", # Advanced search is better for in-depth topics
            max_results=4,
            include_answer=True, # Request an LLM-generated summary from Tavily
            include_domains=TRUSTED_DOMAINS
        )

        results = response.get("results", [])
        
        # Structure the output for the LLM
        structured_results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
                "score": r.get("score"),
            }
            for r in results
        ]

        return {
            "synthesized_answer": response.get("answer", "No synthesized answer was provided."),
            "status": f"Found {len(structured_results)} results from trusted scientific sources.",
            "results": structured_results
        }

    except Exception as e:
        logging.exception(f"Scientific search failed for query: '{query}'")
        return {"error": f"An unexpected error occurred during the scientific search: {e}"}