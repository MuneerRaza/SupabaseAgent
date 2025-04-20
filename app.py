# app.py
import streamlit as st
import json
import time # For simulating steps
from graph import run_graph # Import the function that runs your agent
import logging # Optional: Configure logging if needed for the Streamlit app itself

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic AI System",
    page_icon="🧠",
    layout="wide"
)

# --- Logging ---
# Configure logging for Streamlit app (optional)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Title ---
st.title("🧠 Agentic AI System Interface")
st.caption("Powered by LangGraph, Groq, Gemini, Whisper & Supabase")

# --- Session State Initialization ---
# Initialize session state variables to hold output and errors across reruns
if 'output' not in st.session_state:
    st.session_state.output = None
if 'error' not in st.session_state:
    st.session_state.error = None
if 'processing' not in st.session_state:
    st.session_state.processing = False # To prevent multiple runs if one is active

# --- Input Form ---
with st.form("prompt_form"):
    user_prompt = st.text_area(
        "Enter your prompt:",
        height=100,
        placeholder="e.g., fetch all employees with age over 30, OR get total from receipt image refund_req5.png, OR summarize audio refund_audio_2.mp3"
    )
    submitted = st.form_submit_button("Run Agent")

# --- Agent Execution Logic ---
if submitted and user_prompt and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.output = None # Clear previous output
    st.session_state.error = None   # Clear previous error

    # Display thinking process using st.status
    with st.status("Agent is working...", expanded=True) as status_container:
        try:
            st.write("🔄 Rewriting prompt...")
            time.sleep(0.5) # Simulate work

            st.write("🧭 Routing request...")
            time.sleep(0.5) # Simulate work

            # Simulate the main processing step (could be SQL, Image, Audio)
            # A more sophisticated simulation could guess based on keywords
            if "image" in user_prompt.lower() or ".png" in user_prompt.lower() or "receipt" in user_prompt.lower():
                 st.write("🖼️ Analyzing image(s)...")
            elif "audio" in user_prompt.lower() or ".mp3" in user_prompt.lower() or "summarize" in user_prompt.lower():
                 st.write("🔊 Processing audio...")
            elif any(kw in user_prompt.lower() for kw in ["sql", "fetch", "insert", "update", "delete", "table", "employees", "refund"]):
                 st.write("⚙️ Executing database operation...")
            else:
                 st.write("⚙️ Processing request...")
            time.sleep(0.5) # Simulate work

            # Actually run the agent graph
            # This is where the real work happens, the steps above are visual simulation
            result = run_graph(user_prompt)

            st.write("✅ Finalizing result...")
            time.sleep(0.3)

            # Store result in session state
            if result.get("error_message"):
                st.session_state.error = result["error_message"]
                status_container.update(label="Agent finished with error!", state="error")
            else:
                st.session_state.output = result.get("final_result", "No result returned.")
                status_container.update(label="Agent finished successfully!", state="complete")

        except Exception as e:
            st.session_state.error = f"An unexpected error occurred in the frontend: {str(e)}"
            logging.error(f"Frontend error: {e}", exc_info=True)
            status_container.update(label="Critical Error!", state="error")
        finally:
            st.session_state.processing = False # Allow new runs

    # Trigger a rerun to display the results stored in session_state below the form
    st.rerun()


# --- Display Output/Error ---
st.divider()

if st.session_state.error:
    st.error(f"**Agent Error:**\n\n{st.session_state.error}")

if st.session_state.output:
    st.subheader("Agent Output:")
    output_data = st.session_state.output
    if isinstance(output_data, str):
        try:
            # Try parsing if it looks like a JSON string
            parsed_json = json.loads(output_data)
            st.json(parsed_json)
        except json.JSONDecodeError:
            # If not JSON, display as markdown (handles text nicely)
            st.markdown(output_data)
    elif isinstance(output_data, (dict, list)):
        # If it's already a dict or list, display as JSON
        st.json(output_data)
    else:
        # Otherwise, display as plain text
        st.write(output_data)