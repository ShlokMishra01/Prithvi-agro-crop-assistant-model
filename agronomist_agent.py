from typing import TypedDict, Optional, Iterator, Dict, Any, List
from uuid import uuid4
import json

from agent.graph import agent_executor


class AgentState(TypedDict, total=False):
    """UI-facing AgentState for Streamlit app."""

    user_query: str
    image_path: Optional[str]
    step_count: int
    errors: List[str]
    final_response: Optional[str]
    final_trace: Dict[str, Any]

class _AgentWrapper:
    """
    Wrapper to provide a streaming API for the Streamlit UI.
    It translates the raw LangGraph stream into a simplified format for easy display.
    """
    def _create_strategic_summary(self, tool_name: str, tool_output: Dict[str, Any]) -> str:
        """Helper to create the concise summaries we need for the UI."""
        if tool_output.get("error"):
            return f"Error: {tool_output.get('error')}"
        
        if tool_name == "vision_classify":
            pred = tool_output.get('predicted_class', 'unknown')
            conf = tool_output.get('confidence', 0)
            return f"Diagnosis: '{pred}' (Confidence: {conf:.2f})"

        if tool_name == "weather_tool":
            return tool_output.get("summary", "Weather data retrieved.")

        if tool_name == "geospatial_tool":
            period_summaries = []
            for period, data in tool_output.items():
                if isinstance(data, dict) and data.get("interpretation"):
                    interp = data["interpretation"]
                    period_summaries.append(f"Period '{period}': Vigor={interp.get('vegetation_vigor')}")
            return "Analysis complete. " + "; ".join(period_summaries)
        
        if tool_name in ["scientific_search", "general_web_search"]:
            num_results = len(tool_output.get("results", []))
            return f"Found {num_results} results."
            
        return "Tool executed successfully."

    def stream(self, initial_state: AgentState) -> Iterator[Dict[str, Any]]:
        """Streams events from the agent execution, formatted for the UI."""
        internal_state: Dict[str, Any] = {
            "user_input": initial_state.get("user_query", ""),
            "image_path": initial_state.get("image_path"),
            "lat": initial_state.get("lat"),
            "lon": initial_state.get("lon"),
            "intermediate_steps": [],
        }
        thread_id = f"ui-run-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        # Use agent_executor.stream() to get real-time events
        for event in agent_executor.stream(internal_state, config=config):
            for node_name, node_state in event.items():
                if node_name == "agent":
                    # The agent node decides which tools to call
                    decision = node_state.get("decision", {})
                    if decision.get("type") == "tools":
                        for call in decision.get("calls", []):
                            tool_name = call.get("name")
                            tool_args = call.get("args", {})
                            yield {
                                "event": "on_tool_start",
                                "data": {"name": tool_name, "input": tool_args}
                            }
                
                elif node_name == "action":
                    # The action node returns the result of the tool calls
                    steps = node_state.get("intermediate_steps", [])
                    if steps:
                        last_step = steps[-1]
                        tool_name = last_step.get("tool")
                        tool_output = last_step.get("output", {})
                        summary = self._create_strategic_summary(tool_name, tool_output)
                        yield {
                            "event": "on_tool_end",
                            "data": {"name": tool_name, "output_summary": summary}
                        }

                elif node_name == "synthesize_final_answer":
                    # The synthesis node starts generating the final answer
                    yield {"event": "on_synthesis_start", "data": {}}
                    final_answer = node_state.get("final_answer", {})
                    yield {
                        "event": "on_final_response",
                        "data": {
                            "text": final_answer.get("text", "No final text provided."),
                            "trace": node_state # The full final state is the trace
                        }
                    }

def build_agent() -> _AgentWrapper:
    return _AgentWrapper()
