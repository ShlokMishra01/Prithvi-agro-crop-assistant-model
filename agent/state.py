from typing import List, TypedDict, Any, Optional, Dict

class AgentState(TypedDict, total=False):
    """Per-turn state passed through the LangGraph nodes.

    Keys
    ----
    - user_input: Original user text prompt
    - image_path: Optional path to a user-provided image
    - chat_history: Conversation memory so far (list of messages)
    - lat/lon: Optional coordinates supplied or inferred
    - agent_outcomes: List of tool outputs collected during this turn
    - intermediate_steps: List of (tool_invocation, observation) pairs
    - decision: The last LLM decision (tool calls or final answer)
    - final_answer: The composed final answer, when done
    """

    user_input: str
    image_path: Optional[str]
    chat_history: List[Dict[str, Any]]
    intermediate_steps: List[Dict[str, Any]]
    decision: Optional[Dict[str, Any]]
    final_answer: Optional[str]
