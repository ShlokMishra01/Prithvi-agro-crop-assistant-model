import os
import logging
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from agronomist_agent import build_agent, AgentState

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()


def get_max_upload_bytes() -> int:
    """Return the configured upload cap, falling back to a conservative default."""
    try:
        size_mb = int(os.getenv("AGRIBOT_MAX_IMAGE_MB", "8"))
        return max(size_mb, 1) * 1024 * 1024
    except ValueError:
        logging.warning("Invalid AGRIBOT_MAX_IMAGE_MB; using the 8 MB default.")
        return 8 * 1024 * 1024


MAX_UPLOAD_BYTES = get_max_upload_bytes()

st.set_page_config(
    page_title="AgriBot Field Advisor",
    page_icon="🌱",
    layout="wide",
)

# --- Session State Initialization ---
if "final_response" not in st.session_state:
    st.session_state.final_response = None
if "final_trace" not in st.session_state:
    st.session_state.final_trace = {}
if "current_image_path" not in st.session_state:
    st.session_state.current_image_path = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []
if "show_left_panel" not in st.session_state:
    st.session_state.show_left_panel = True

# =================
# UI Layout
# =================

# Determine if we should use two-column layout
use_two_columns = st.session_state.final_response is not None

# Initialize run variable
run = False

if not use_two_columns:
    # Center the title in single-column view
    st.markdown("<h1 style='text-align: center;'>🌱 AgriBot Field Advisor</h1>", unsafe_allow_html=True)
else:
    # Left-align title in two-column view
    st.title("🌱 AgriBot Field Advisor")

if use_two_columns:
    # Add toggle button for left panel at the top
    col_toggle, col_spacer = st.columns([1, 5])
    with col_toggle:
        if st.button("◀ Hide" if st.session_state.show_left_panel else "▶ Show", key="toggle_panel"):
            st.session_state.show_left_panel = not st.session_state.show_left_panel
            st.rerun()
    
    # Two-column layout after response is generated - dynamic sizing based on left panel visibility
    if st.session_state.show_left_panel:
        left_col, right_col = st.columns([1, 1], gap="large")
    else:
        # When left panel is hidden, show only right column
        right_col = st.container()
    
    if st.session_state.show_left_panel:
        with left_col:
            st.markdown("### 📝 Input & Agent Progress")
            st.markdown("---")
            
            # --- User Inputs (Display the original query, disabled) ---
            # Retrieve the original query from session state
            displayed_query = st.session_state.get("original_query", "")
            user_query = st.text_area("User Query", value=displayed_query, placeholder="e.g., Brown spots on tomato leaves...", height=100, disabled=True)
            image_file = st.file_uploader("Upload Leaf Image (optional)", type=["jpg", "jpeg", "png"], disabled=True)
            
            # --- Action Buttons ---
            col_run, col_clear = st.columns(2)
            with col_run: 
                run = st.button("▶️ Run Agent", width='stretch', type="primary", disabled=True)
            with col_clear:
                if st.button("🔄 New Query", width='stretch', key="clear_two_col"):
                    st.session_state.final_response = None
                    st.session_state.final_trace = {}
                    st.session_state.agent_logs = []
                    st.session_state.current_image_path = None
                    st.session_state.show_left_panel = True
                    if "original_query" in st.session_state:
                        del st.session_state.original_query
                    st.rerun()
            
            st.markdown("---")
            
            # Display image and agent logs side by side
            img_col, log_col = st.columns([1, 1])
            
            with img_col:
                # Display uploaded image if available
                if st.session_state.current_image_path and os.path.exists(st.session_state.current_image_path):
                    st.markdown("#### 🖼️ Analyzed Image")
                    img_container = st.container(border=True)
                    with img_container:
                        st.image(st.session_state.current_image_path, width='stretch')
            
            with log_col:
                # Display agent logs in a container
                if st.session_state.agent_logs:
                    st.markdown("#### 🤖 Activity Log")
                    log_container = st.container(border=True)
                    with log_container:
                        for log in st.session_state.agent_logs:
                            st.markdown(log)
    
    with right_col:
        st.markdown("### 💬 Final Response")
        st.markdown("---")
        
        # Display response in a container for better visual separation
        response_container = st.container(border=True)
        with response_container:
            response_text = st.session_state.final_response.get("text", "No text found.")
            st.markdown(response_text, unsafe_allow_html=True)
        
        st.markdown("")  # Add spacing
        with st.expander("🔍 View Full Execution Trace", expanded=False):
            st.json(st.session_state.final_trace)

else:
    # Single-column layout (initial state) - centered for better UX
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Add some visual styling
        st.markdown("""
            <style>
            .big-font {
                font-size: 18px !important;
                text-align: center;
                color: #4CAF50;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="big-font">🌿 Describe the field issue and optionally attach a plant image. The advisor will gather relevant evidence.</p>', unsafe_allow_html=True)
        st.markdown("")  # Add spacing
        
        # --- User Inputs ---
        user_query = st.text_area("User Query", placeholder="e.g., Brown spots on tomato leaves...", height=120, key="query_single")
        
        st.markdown("")  # Add spacing
        image_file = st.file_uploader("Upload plant image (optional, JPG or PNG)", type=["jpg", "jpeg", "png"], key="image_single")
        st.caption(f"Maximum image size: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        
        # NO image preview here - only show after response is generated
        
        st.markdown("")  # Add spacing
        
        # --- Action Buttons ---
        col_run, col_clear = st.columns(2)
        with col_run: 
            run = st.button("▶️ Run Agent", width='stretch', type="primary", key="run_single")
        with col_clear:
            if st.button("🔄 Clear", width='stretch', key="clear_single"):
                st.session_state.final_response = None
                st.session_state.final_trace = {}
                st.session_state.agent_logs = []
                st.session_state.current_image_path = None
                st.rerun()

# =================
# Agent Execution and DYNAMIC Display Logic
# =================

if run:
    if not user_query or not user_query.strip():
        st.error("Please describe the crop or field question before running the advisor.")
    elif image_file is not None and image_file.size > MAX_UPLOAD_BYTES:
        st.error(f"The uploaded image is too large. Please choose a file under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    else:
        # Clear previous results before a new run
        st.session_state.final_response = None
        st.session_state.final_trace = {}
        st.session_state.agent_logs = []
        
        # Save the original query to session state
        st.session_state.original_query = user_query.strip()
        
        # Prepare inputs
        image_path = None
        if image_file is not None:
            tmp_dir = ".streamlit_tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            extension = os.path.splitext(image_file.name)[1].lower()
            image_path = os.path.join(tmp_dir, f"{uuid4().hex}{extension}")
            with open(image_path, "wb") as f: f.write(image_file.getbuffer())
            st.session_state.current_image_path = image_path
        else:
            st.session_state.current_image_path = None

        initial_state = {
            "user_query": user_query.strip(),
            "image_path": image_path,
            "lat": None,
            "lon": None,
        }

        # --- Agent Execution with Status Display ---
        try:
            agent = build_agent()
            
            # Create centered columns to match input field width
            _, status_col, _ = st.columns([1, 2, 1])
            
            # Flag to track if we should rerun after stream completes
            should_rerun = False
            
            # Use st.status to show the agent's progress with loading animation
            with status_col:
                with st.status("🧑‍🌾 Agent is planning...", expanded=True) as status:
                    for event in agent.stream(initial_state):
                        if event["event"] == "on_tool_start":
                            tool_name = event['data']['name']
                            status.update(label=f"🔧 Calling tool: `{tool_name}`...")
                            st.session_state.agent_logs.append(f"🔧 Calling `{tool_name}`...")
                        
                        elif event["event"] == "on_tool_end":
                            tool_name = event['data']['name']
                            st.session_state.agent_logs.append(f"✅ `{tool_name}` finished")
                        
                        elif event["event"] == "on_synthesis_start":
                            status.update(label="✍️ Synthesizing final answer...")
                            st.session_state.agent_logs.append("✍️ Synthesizing final answer...")
                        
                        elif event["event"] == "on_final_response":
                            st.session_state.final_response = event["data"]
                            st.session_state.final_trace = event["data"]["trace"]
                            status.update(label="✅ Agent finished!", state="complete")
                            st.session_state.agent_logs.append("✅ Agent finished!")
                            # Set flag to rerun after stream completes
                            should_rerun = True
            
            # Rerun AFTER the stream has fully completed
            if should_rerun:
                st.rerun()

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.error("Agent execution failed", exc_info=True)
            st.session_state.agent_logs.append(f"❌ Error: {str(e)}")
