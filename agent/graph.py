import os
import json
import logging
import time
import re
from typing import Dict, Any, List
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from groq import APIStatusError, RateLimitError

from .state import AgentState
from .prompt import SYSTEM_PROMPT, SYNTHESIS_PROMPT_TEMPLATE
from tools.tool_schemas import SearchInput, VisionInput, WeatherInput, GeospatialInput, GeocodingInput
from tools.search_tool import scientific_search, general_web_search
from tools.vision_tool import vision_classify
from tools.weather_tool import weather_tool
from tools.geospatial_tool import geospatial_tool
from tools.geocoding_tool import geocode_location

load_dotenv()

# Assemble available tools (LangChain tool objects)
TOOLS: List[Tool] = [
    geocode_location,  # Tier 0: Location resolution (prerequisite for weather/geospatial)
    scientific_search,
    general_web_search,
    vision_classify,
    weather_tool,
    geospatial_tool,
]
TOOL_REGISTRY = {t.name: t for t in TOOLS}

# Planning model with dual API key fallback strategy
PRIMARY_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECONDARY_GROQ_API_KEY = os.getenv("GROQ_API_KEY_BACKUP")

MODEL_PRIMARY = ChatGroq(
    api_key=PRIMARY_GROQ_API_KEY,
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
    max_retries=0,  # Disable retries to fail fast on rate limits
)

MODEL_SECONDARY = ChatGroq(
    api_key=SECONDARY_GROQ_API_KEY,
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
    max_retries=0,  # Disable retries to fail fast on rate limits
) if SECONDARY_GROQ_API_KEY else None

SYNTHESIS_MODEL_GROQ = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    temperature=0.2,
    max_retries=0,  # Disable retries to fail fast on rate limits
)

SYNTHESIS_MODEL_OPENROUTER = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0.2
)

def format_agent_prompt(state: AgentState) -> List[Any]:

    """
    Builds a highly memory-efficient prompt using a "Strategic Summary" of past tool calls.
    This provides the planner with the critical outcomes of previous steps without sending
    the full raw data, thus preventing context window overflow. The full, unabridged
    data is still preserved in the state for the final synthesis node.
    """
    user_input = state.get("user_input", "")
    image_path = state.get("image_path")
    lat = state.get("lat")
    lon = state.get("lon")
    intermediate_steps = state.get("intermediate_steps", [])

    # NEW: Track executed tool signatures to prevent duplicates
    executed_signatures = []
    prior_summary_text = "No tools have been called yet."
    
    if intermediate_steps:
        summary_parts = []
        for step in intermediate_steps:
            tool_name = step.get("tool")
            tool_input = step.get("input")
            tool_output = step.get("output", {})

            # NEW: Create a signature for deduplication tracking
            # Sort keys for consistent comparison
            sorted_args = sorted(tool_input.items())
            args_str = ", ".join([f"{k}={repr(v)}" for k, v in sorted_args])
            signature = f"{tool_name}({args_str})"
            executed_signatures.append(signature)

            # We generate a concise summary that includes the most critical,
            # decision-informing piece of information from the output.
            summary = ""
            if tool_output.get("error"):
                summary = f"❌ ERROR: {tool_output.get('error')}"
            
            elif tool_name == "vision_classify":
                pred = tool_output.get('predicted_class', 'unknown')
                conf = tool_output.get('confidence', 0)
                confidence_status = "HIGH (≥0.75)" if conf >= 0.75 else "LOW (<0.75 - needs verification)"
                summary = f"✓ Image classified as '{pred}' with {confidence_status} confidence ({conf:.2f})"

            elif tool_name == "weather_tool":
                summary = f"✓ {tool_output.get('summary', 'Weather data retrieved successfully.')}"

            elif tool_name == "geospatial_tool":
                period_summaries = []
                for period, data in tool_output.items():
                    if isinstance(data, dict) and data.get("interpretation"):
                        interp = data["interpretation"]
                        period_summaries.append(
                            f"Period '{period}': Vigor={interp.get('vegetation_vigor')}, "
                            f"WaterStress={interp.get('water_stress')}, "
                            f"Chlorophyll={interp.get('chlorophyll_content')}"
                        )
                summary = "✓ Satellite analysis complete. " + " | ".join(period_summaries) if period_summaries else "✓ Analysis complete (no detailed interpretation)"
            
            elif tool_name in ["scientific_search", "general_web_search"]:
                status = tool_output.get("status", "Search completed.")
                num_results = len(tool_output.get("results", []))
                has_synthesis = bool(tool_output.get("synthesized_answer"))
                if has_synthesis and tool_name == "scientific_search":
                    summary = f"✓ Scientific search complete with synthesized answer from {num_results} trusted sources"
                else:
                    summary = f"✓ {status} ({num_results} results)"

            else:
                summary = f"✓ Tool '{tool_name}' executed successfully"

            summary_parts.append(
                f"Step {len(summary_parts) + 1}:\n"
                f"- Tool Called: `{tool_name}`\n"
                f"- Input: {tool_input}\n"
                f"- Outcome: {summary}"
            )
        prior_summary_text = "\n---\n".join(summary_parts)

    # NEW: Add explicit deduplication section
    dedup_section = ""
    if executed_signatures:
        dedup_section = (
            f"\n⚠️ ALREADY EXECUTED TOOLS (DO NOT REPEAT THESE EXACT CALLS):\n"
            f"{chr(10).join(f'  - {sig}' for sig in executed_signatures)}\n"
        )

    human_msg = (
        f"User Query:\n{user_input}\n\n"
        f"Context:\n"
        f"- image_path: {image_path}\n"
        f"- coordinates: lat={lat}, lon={lon}\n\n"
        f"Summary of Executed Tool Calls:\n"
        f"---------------------\n"
        f"{prior_summary_text}\n"
        f"---------------------\n"
        f"{dedup_section}"
        f"\nYour Task:\n"
        f"Based on the user query and the summary of outcomes from the tools called so far, decide the single next step. "
        f"The full, detailed JSON from these tools is saved in memory and will be available for the final answer synthesis, but you must make your decision based on the summaries alone. "
        f"Your choices are:\n"
        f"1. **Call another tool:** If the summaries indicate that critical information is still missing. "
        f"IMPORTANT: Check the 'ALREADY EXECUTED TOOLS' list above - do NOT repeat those exact calls!\n"
        f"2. **Finish:** If the summaries show you have successfully gathered all necessary pieces of information to answer the user's query."
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_msg)]

def agent_node(state: AgentState) -> AgentState:
    """
    LLM reasoning node with fallback strategy:
    - Tries PRIMARY Groq API key first
    - Immediately switches to SECONDARY Groq API key if rate limited (429) or error occurs
    - If both are rate limited, waits the requested time and retries
    - Receives AgentState with prior intermediate_steps
    - Chooses which tool to call (with parameters) or final answer
    """
    messages = format_agent_prompt(state)
    resp = None
    error = None
    
    # Helper function to extract wait time from rate limit error message
    def extract_wait_time(error_message: str) -> float:
        """Extract wait time in seconds from Groq rate limit error message."""
        match = re.search(r'Please try again in ([\d.]+)s', str(error_message))
        if match:
            return float(match.group(1))
        return 10.0  # Default to 10 seconds if we can't parse
    
    # Try primary API key first
    try:
        llm = MODEL_PRIMARY.bind_tools(TOOLS)
        resp = llm.invoke(messages)
        print("AGENT NODE: Using PRIMARY Groq API key")
    except RateLimitError as e:
        # Immediately switch on rate limit without retrying
        error = e
        primary_wait_time = extract_wait_time(str(e))
        print(f"AGENT NODE: Primary API key rate limited (429). Switching to secondary immediately...")
        
        # Fallback to secondary API key if available
        if MODEL_SECONDARY:
            try:
                llm = MODEL_SECONDARY.bind_tools(TOOLS)
                resp = llm.invoke(messages)
                print("AGENT NODE: SECONDARY Groq API key succeeded")
            except RateLimitError as e2:
                # Both keys are rate limited - wait and retry with primary
                secondary_wait_time = extract_wait_time(str(e2))
                wait_time = max(primary_wait_time, secondary_wait_time)
                print(f"AGENT NODE: Both API keys rate limited. Waiting {wait_time:.1f}s before retrying...")
                time.sleep(wait_time)
                
                # Retry with primary after waiting
                try:
                    llm = MODEL_PRIMARY.bind_tools(TOOLS)
                    resp = llm.invoke(messages)
                    print("AGENT NODE: Retry with PRIMARY key succeeded after waiting")
                except Exception as e3:
                    raise Exception(f"All retry attempts failed. Primary: {str(error)}, Secondary: {str(e2)}, Retry: {str(e3)}")
            except Exception as e2:
                print(f"AGENT NODE: Secondary API key also failed: {str(e2)}")
                raise Exception(f"Both Groq API keys failed. Primary rate limited, Secondary: {str(e2)}")
        else:
            # No secondary key - wait and retry with primary
            print(f"AGENT NODE: No backup key configured. Waiting {primary_wait_time:.1f}s before retrying...")
            time.sleep(primary_wait_time)
            try:
                llm = MODEL_PRIMARY.bind_tools(TOOLS)
                resp = llm.invoke(messages)
                print("AGENT NODE: Retry with PRIMARY key succeeded after waiting")
            except Exception as e2:
                raise Exception(f"Primary Groq API key failed after retry: {str(e2)}")
                
    except APIStatusError as e:
        # Handle other API errors (4xx, 5xx)
        error = e
        print(f"AGENT NODE: Primary API key failed with status error: {str(e)}")
        
        # Fallback to secondary API key if available
        if MODEL_SECONDARY:
            try:
                llm = MODEL_SECONDARY.bind_tools(TOOLS)
                resp = llm.invoke(messages)
                print("AGENT NODE: SECONDARY Groq API key succeeded")
            except Exception as e2:
                print(f"AGENT NODE: Secondary API key also failed: {str(e2)}")
                raise Exception(f"Both Groq API keys failed. Primary: {str(error)}, Secondary: {str(e2)}")
        else:
            raise Exception(f"Primary Groq API key failed and no backup key configured: {str(error)}")
    except Exception as e:
        # Handle any other unexpected errors
        error = e
        print(f"AGENT NODE: Primary API key failed: {str(e)}")
        
        # Fallback to secondary API key if available
        if MODEL_SECONDARY:
            try:
                llm = MODEL_SECONDARY.bind_tools(TOOLS)
                resp = llm.invoke(messages)
                print("AGENT NODE: SECONDARY Groq API key succeeded")
            except Exception as e2:
                print(f"AGENT NODE: Secondary API key also failed: {str(e2)}")
                raise Exception(f"Both Groq API keys failed. Primary: {str(error)}, Secondary: {str(e2)}")
        else:
            raise Exception(f"Primary Groq API key failed and no backup key configured: {str(error)}")

    # Extract tool calls if any
    tool_calls = getattr(resp, "tool_calls", None) or resp.additional_kwargs.get("tool_calls", [])
    
    if not tool_calls:
        print("AGENT NODE: Decided to generate final answer.") 
        print('='*50)
        return {**state, "decision": {"type": "final"}}

    calls_summary = []
    for call in tool_calls:
        tool_name = call.get('name')
        tool_args = call.get('args', {})
        args_str = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in tool_args.items()])
        calls_summary.append(f"{tool_name}({args_str})")
    
    print(f"AGENT NODE: Decided to call tools: [{', '.join(calls_summary)}]")
    print('='*50)
    return {**state, "decision": {"type": "tools", "calls": tool_calls, "message": resp}}


def tool_node(state: AgentState) -> AgentState:
    """
    Execute the tool calls decided by the LLM.
    - Each tool invocation appends to intermediate_steps
    - No deduplication logic is hardcoded; LLM decides it based on prior_steps
    """
    decision = state.get("decision", {})
    calls = decision.get("calls", [])
    intermediate_steps = state.get("intermediate_steps", [])

    for call in calls:
        # Normalize call object
        name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else None)
        args = getattr(call, "args", None) or (call.get("args") if isinstance(call, dict) else {})

        tool = TOOL_REGISTRY.get(name)
        if not tool:
            intermediate_steps.append({"tool": name, "input": args, "output": {"tool": name, "error": f"Unknown tool: {name}"}})
            continue

        try:
            result = tool.invoke(args)
            intermediate_steps.append({"tool": name, "input": args, "output": result})
        except Exception as e:
            intermediate_steps.append({"tool": name, "input": args, "output": {"tool": name, "error": str(e)}})

    return {**state, "intermediate_steps": intermediate_steps}


def router_function(state: AgentState) -> str:
    """Routes the workflow based on the agent's last decision."""
    decision = state.get("decision", {}).get("type")
    if decision == "tools":
        print("ROUTER: Decision is 'tools'. Routing to action.")
        print('='*50)
        return "action"
    print(f"ROUTER: Decision is '{decision}'. Routing to synthesis.")
    return "synthesize_final_answer"


# --- CREATE THE NEW NODE FUNCTION ---
def synthesis_node(state: AgentState) -> AgentState:
    """
    Generates the final, clean, user-facing response using a primary model with a fallback.
    Immediately switches to fallback on rate limit (429) to reduce response time.
    If both are rate limited, waits the requested time and retries.
    """
    print("SYNTHESIS NODE: Generating final answer.")
    
    # Helper function to extract wait time from rate limit error message
    def extract_wait_time(error_message: str) -> float:
        """Extract wait time in seconds from Groq rate limit error message."""
        match = re.search(r'Please try again in ([\d.]+)s', str(error_message))
        if match:
            return float(match.group(1))
        return 10.0  # Default to 10 seconds if we can't parse

    # Prepare the prompt and input data
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_PROMPT_TEMPLATE),
        ("human",
         "Original User Query:\n{user_input}\n\n"
         "Full History of Tool Calls and Outputs:\n"
         "---------------------\n"
         "{intermediate_steps_str}\n"
         "---------------------\n\n"
         "Based on all the information above, please generate the final, complete answer now.")
    ])
    history_str = "\n---\n".join([json.dumps(s, indent=2) for s in state.get("intermediate_steps", [])])
    synthesis_input = {
        "user_input": state["user_input"],
        "intermediate_steps_str": history_str
    }

    primary_chain = synthesis_prompt | SYNTHESIS_MODEL_GROQ | StrOutputParser()
    fallback_chain = synthesis_prompt | SYNTHESIS_MODEL_OPENROUTER | StrOutputParser()

    final_answer_text = ""
    try:
        # Attempt to invoke the primary chain (Groq)
        print("SYNTHESIS NODE: Attempting to use primary model (Groq)...")
        final_answer_text = primary_chain.invoke(synthesis_input)
        print("SYNTHESIS NODE: Primary model (Groq) succeeded.")

    except RateLimitError as e:
        # Immediately switch on rate limit without waiting for retry
        primary_wait_time = extract_wait_time(str(e))
        logging.warning(f"Primary synthesis model (Groq) rate limited (429). Switching to OpenRouter immediately.")
        print(f"SYNTHESIS NODE: Primary model rate limited. Switching to OpenRouter immediately...")
        try:
            final_answer_text = fallback_chain.invoke(synthesis_input)
            print("SYNTHESIS NODE: Fallback model (OpenRouter) succeeded.")
        except RateLimitError as e2:
            # Both models rate limited - wait and retry
            fallback_wait_time = extract_wait_time(str(e2))
            wait_time = max(primary_wait_time, fallback_wait_time)
            logging.warning(f"Both synthesis models rate limited. Waiting {wait_time:.1f}s before retrying...")
            print(f"SYNTHESIS NODE: Both models rate limited. Waiting {wait_time:.1f}s before retrying...")
            time.sleep(wait_time)
            try:
                final_answer_text = primary_chain.invoke(synthesis_input)
                print("SYNTHESIS NODE: Retry with primary model succeeded after waiting.")
            except Exception as e3:
                logging.error(f"All synthesis models failed after retry: {e3}")
                final_answer_text = "I'm sorry, but I'm currently unable to generate a final response due to rate limiting on all available services. Please try again in a moment."
        except Exception as fallback_e:
            logging.error(f"Fallback synthesis model (OpenRouter) also failed: {fallback_e}")
            final_answer_text = "I'm sorry, but I'm currently unable to generate a final response due to issues with my provider services. Please try again shortly."

    except APIStatusError as e:
        # Handle other Groq-specific API errors (4xx, 5xx)
        logging.warning(f"Primary synthesis model (Groq) failed with API error: {e}. Falling back to OpenRouter.")
        print(f"SYNTHESIS NODE: Primary model failed with status error. Falling back to OpenRouter...")
        try:
            final_answer_text = fallback_chain.invoke(synthesis_input)
            print("SYNTHESIS NODE: Fallback model (OpenRouter) succeeded.")
        except Exception as fallback_e:
            logging.error(f"Fallback synthesis model (OpenRouter) also failed: {fallback_e}")
            final_answer_text = "I'm sorry, but I'm currently unable to generate a final response due to issues with my provider services. Please try again shortly."

    except Exception as e:
        # Catch any other unexpected errors and try the fallback
        logging.warning(f"An unexpected error occurred with the primary model (Groq): {e}. Falling back to OpenRouter.")
        print(f"SYNTHESIS NODE: Primary model failed. Falling back to OpenRouter...")
        try:
            final_answer_text = fallback_chain.invoke(synthesis_input)
            print("SYNTHESIS NODE: Fallback model (OpenRouter) succeeded.")
        except Exception as fallback_e:
            logging.error(f"Fallback synthesis model (OpenRouter) also failed: {fallback_e}")
            final_answer_text = "I'm sorry, but I'm currently unable to generate a final response due to issues with my provider services. Please try again shortly."
    
    return {**state, "final_answer": {"text": final_answer_text}}


def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("action", tool_node)
    # --- ADD THE NEW NODE TO THE GRAPH DEFINITION ---
    workflow.add_node("synthesize_final_answer", synthesis_node)
    
    workflow.set_entry_point("agent")
    
    # --- UPDATE THE CONDITIONAL EDGES ---
    workflow.add_conditional_edges(
        "agent",
        router_function,
        {
            "action": "action",
            # This new path sends the flow to the synthesis node before ending
            "synthesize_final_answer": "synthesize_final_answer"
        }
    )
    workflow.add_edge("action", "agent")
    
    # --- THE GRAPH NOW OFFICIALLY ENDS AFTER SYNTHESIS ---
    workflow.add_edge("synthesize_final_answer", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


agent_executor = build_graph()
