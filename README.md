# AgriBot Field Advisor

> A research-oriented decision-support prototype for connecting crop observations with weather, satellite, image, and agricultural reference data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)](https://langchain-ai.github.io/langgraph/)

AgriBot Field Advisor is my adapted implementation of an agronomy assistant. It accepts a farmer or field-manager question, optionally with a plant image or location, then selects useful evidence sources before producing a practical, clearly formatted response. The project is intended for research and prototype use; it is not a replacement for field inspection, laboratory testing, or local agronomic advice.

## Capabilities

- Classify a JPG or PNG plant image using a configurable deployed vision endpoint.
- Review current conditions, forecasts, historical weather, soil moisture, wind, and ET0.
- Compare Sentinel-2 vegetation indices across one or more field periods.
- Resolve named locations to coordinates for downstream analysis.
- Retrieve agricultural and scientific references to support an answer.
- Stream the advisor's progress and final response in a focused Streamlit interface.

## Workflow

```text
Field question + optional image/location
              |
       LangGraph planner
              |
 vision | weather | satellite | location | research
              |
  evidence-aware field guidance
```

The planner retains concise summaries of completed tool calls to reduce duplicate work while preserving detailed results for the final synthesis.

## Project Layout

```text
.
├── main.py                 # Streamlit interface and upload handling
├── agronomist_agent.py     # UI-facing streaming wrapper
├── agent/
│   ├── graph.py            # Planner, tools, and response synthesis
│   ├── prompt.py           # Agent and synthesis guidance
│   └── state.py            # Shared graph state
├── tools/
│   ├── geocoding_tool.py   # Location lookup
│   ├── geospatial_tool.py  # Sentinel-2 analysis
│   ├── search_tool.py      # Agricultural and web research
│   ├── tool_schemas.py     # Input validation
│   ├── vision_tool.py      # Crop image classification
│   └── weather_tool.py     # Weather analysis
├── finetuning-clip.py      # Optional model fine-tuning utility
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run main.py
```

Create a `.env` file in the repository root before starting the app. Keep it out of version control.

```dotenv
# Required for the LangGraph planner
GROQ_API_KEY=your_groq_key
# Optional fallback when the primary key is rate limited
GROQ_API_KEY_BACKUP=your_second_groq_key
GROQ_MODEL=llama-3.3-70b-versatile

# Required by research and location tools when those capabilities are used
TAVILY_API_KEY=your_tavily_key
OPENWEATHER_API_KEY=your_openweather_key

# Required for Sentinel-2 analysis
SENTINEL_CLIENT_ID=your_sentinel_client_id
SENTINEL_CLIENT_SECRET=your_sentinel_client_secret

# Optional vision service settings
VISION_API_URL=https://your-vision-service.example/classify
VISION_API_TIMEOUT_SECONDS=30
AGRIBOT_MAX_IMAGE_MB=8

# Optional synthesis fallback
OPENROUTER_API_KEY=your_openrouter_key
```

Open-Meteo is used for the default weather retrieval and does not need a key. Sentinel Hub credentials are only needed when satellite analysis is selected.

## Recent implementation notes

- Uploads are stored with generated filenames to avoid collisions between sessions.
- Image size and extension are validated both in the UI and before a vision request.
- The vision request timeout and image-size cap are configurable through environment variables.
- Search, location, and satellite date inputs reject blank or invalid values earlier in the workflow.
- Location requests use HTTPS and safely encoded request parameters.

## Example questions

- “What could be causing these brown spots on my tomato leaves?” (attach an image)
- “What is the main weather risk for my wheat over the next seven days in Pune?”
- “Compare field health before and after heavy rain.” (provide coordinates and date ranges)
- “Find research-backed management options for powdery mildew on grapes.”

## Operational notes

- Image classifications are decision support, not a confirmed diagnosis.
- Satellite analyses depend on cloud-free Sentinel-2 observations and the selected date windows.
- API responses depend on third-party availability, credentials, and quotas.
- Temporary uploads are stored in `.streamlit_tmp/`; local caches are ignored by Git.

## Attribution

This adapted prototype uses open-source and hosted components including Streamlit, LangGraph, LangChain, Groq, Sentinel Hub, Open-Meteo, Tavily, OpenWeather, and CLIP-based crop-disease classification tooling. Please retain applicable licenses and citations when extending or distributing the project.

## License

No license has been declared. Add an appropriate license before distributing the project or accepting external contributions.
