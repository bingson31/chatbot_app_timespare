# --- Installation ---
# Make sure you have installed all the required libraries.
# Open your terminal and run the following command:
# pip install streamlit google-generativeai langchain-google-genai langgraph langchain PyPDF2

import streamlit as st
import os
import re
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import PyPDF2

# --- PDF Parsing and Data Extraction ---

# Global variable to store the structured data from the PDF
PDF_DATA = None

def extract_text_from_pdf(pdf_path):
    """Extracts text from all pages of a PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def parse_frt_and_parts_data(text):
    """
    Parses the extracted text to create a structured representation of
    FRT, LON codes, parts, and their relationships.
    This is a simplified parser and might need refinement for complex PDFs.
    """
    data = {"frt_data": {}, "parts_data": {}, "group_to_parts": {}}
    lines = text.split('\n')

    current_group = None

    # Regex patterns to identify different data types
    frt_pattern = re.compile(r'([\w\s,./-]+\.{2,})\s*([\d.]+)') # e.g., CAMSHAFT.................. 1.7
    part_pattern = re.compile(r'(\d+)\s+([\w-]+)\s+(.+?)\s+(\d+)\s*') # e.g., 2 12391-KVY-900 GASKET, HEAD COVER 1
    group_pattern = re.compile(r'^(E-\d+|F-\d+)\s+(.+)') # e.g., E-2 CYLINDER HEAD COVER

    # Simplified parsing logic
    for i, line in enumerate(lines):
        # Check for group headers
        group_match = group_pattern.match(line)
        if group_match:
            current_group = group_match.group(1).strip()
            group_name = group_match.group(2).strip()
            if current_group not in data["group_to_parts"]:
                data["group_to_parts"][current_group] = {
                    "name": group_name,
                    "parts": []
                }
            continue

        # Check for FRT data
        frt_match = frt_pattern.match(line)
        if frt_match:
            service_item = frt_match.group(1).replace('.', ' ').strip().upper()
            frt_value = float(frt_match.group(2))
            # Heuristic to associate FRT with a LON code if mentioned nearby
            lon_code = f"LON-{service_item[:4]}{i}" # Create a plausible LON for demo
            data["frt_data"][service_item] = {"frt": frt_value, "lon": lon_code}
            continue

        # Check for parts data
        part_match = part_pattern.match(line)
        if part_match and current_group:
            part_number = part_match.group(2).strip()
            description = part_match.group(3).strip().upper()
            part_info = {
                "part_number": part_number,
                "description": description,
                "lon": None,
                "frt": None
            }

            # Try to find associated FRT data
            for item, frt_info in data["frt_data"].items():
                if item in description:
                    part_info["lon"] = frt_info["lon"]
                    part_info["frt"] = frt_info["frt"]
                    break

            data["parts_data"][part_number] = part_info
            if current_group:
                data["group_to_parts"][current_group]["parts"].append(part_number)

    return data

def load_and_process_pdf(pdf_path="Katalog-Suku-Cadang-Honda-BeAT.pdf"):
    """Loads and processes the PDF if it hasn't been processed yet."""
    global PDF_DATA
    if PDF_DATA is None:
        if os.path.exists(pdf_path):
            text = extract_text_from_pdf(pdf_path)
            PDF_DATA = parse_frt_and_parts_data(text)
        else:
            st.error(f"Error: PDF file not found at '{pdf_path}'. Please make sure the file is in the same directory.")
            st.stop()
    return PDF_DATA

# --- LangChain Tools ---

@tool
def frt_estimator(job_description: str) -> str:
    """
    Estimates the Flat Rate Time (FRT) for a specific repair job based on its description.
    Use this to find out how many hours a standard repair takes.
    For example: 'How long to replace front brake pads?'
    """
    data = load_and_process_pdf()
    if not data:
        return "Could not load data from PDF."

    job_upper = job_description.upper()
    found_jobs = []

    # Search in both FRT data and parts descriptions
    for item, info in data["frt_data"].items():
        if all(word in item for word in job_upper.split()):
            found_jobs.append(f"The job '{item}' (LON {info['lon']}) has an FRT of {info['frt']} hours.")

    for part_num, info in data["parts_data"].items():
        if info['frt'] is not None and all(word in info['description'] for word in job_upper.split()):
             found_jobs.append(f"Replacing '{info['description']}' ({part_num}) (LON {info['lon']}) has an FRT of {info['frt']} hours.")


    if not found_jobs:
        return f"Could not find an exact FRT estimate for '{job_description}'. Please try rephrasing the job."

    return "\n".join(list(set(found_jobs))) # Return unique findings

@tool
def engine_breakdown_advisor(component_or_issue: str) -> str:
    """
    Suggests relevant spare parts based on a machine issue or component category.
    Use this to get part recommendations for a job like a 'CVT overhaul' or if you need parts for the 'cylinder head'.
    """
    data = load_and_process_pdf()
    if not data:
        return "Could not load data from PDF."

    query_upper = component_or_issue.upper()
    relevant_groups = []
    suggested_parts = []

    # Find relevant machine groups
    for group_code, group_info in data["group_to_parts"].items():
        if query_upper in group_info["name"]:
            relevant_groups.append(group_code)

    if not relevant_groups:
        return f"Could not identify a component group for '{component_or_issue}'. Try using terms like 'Cylinder', 'Crankcase', 'CVT', etc."

    # Get parts from those groups
    for group_code in relevant_groups:
        group_name = data['group_to_parts'][group_code]['name']
        parts_in_group = data["group_to_parts"][group_code]["parts"]
        for part_num in parts_in_group[:5]: # Limit to 5 parts per group for brevity
            part_info = data["parts_data"].get(part_num)
            if part_info:
                frt_info = f" (FRT: {part_info['frt']}h)" if part_info['frt'] else ""
                suggested_parts.append(f"- From {group_name} ({group_code}): {part_info['description']} ({part_info['part_number']}){frt_info}")

    if not suggested_parts:
        return f"Found the group(s) {', '.join(relevant_groups)} but could not retrieve specific parts."

    response = f"For '{component_or_issue}', you might need parts from the following groups: {', '.join(relevant_groups)}\n"
    response += "Commonly related parts include:\n" + "\n".join(list(set(suggested_parts)))
    return response

# --- LangGraph Setup ---

class AgentState(TypedDict):
    messages: Annotated[List, lambda x, y: x + y]

# Initialize tools
tools = [frt_estimator, engine_breakdown_advisor]

def get_model(api_key):
    """Initializes the ChatGoogleGenerativeAI model with the provided API key."""
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, convert_system_message_to_human=True)

# Define agent logic
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def call_model(state, model):
    if not model:
        # Return a message asking for the API key if the model is not initialized
        return {"messages": [AIMessage(content="Please enter your Google Gemini API key in the sidebar to begin.")]}
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def call_tool(state):
    messages = state['messages']
    last_message = messages[-1]
    tool_calls = last_message.tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_to_call = {t.name: t for t in tools}[tool_name]
        tool_output = tool_to_call.invoke(tool_call['args'])
        tool_messages.append(ToolMessage(
            content=str(tool_output),
            tool_call_id=tool_call['id']
        ))
    return {"messages": tool_messages}

# --- Streamlit UI ---

st.set_page_config(page_title="AI Motorcycle Repair Advisor", page_icon="🤖")
st.title("AI Motorcycle Repair Advisor")
st.caption("Your intelligent assistant for FRT estimation and parts advice.")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Google Gemini API Key", type="password")
    if not google_api_key:
        st.warning("Please enter your Google Gemini API key to use the chatbot.")

    st.info("This chatbot uses a local PDF file for knowledge. Make sure `Katalog-Suku-Cadang-Honda-BeAT.pdf` is in the same directory.")
    
    if st.button("Reset Chat History"):
        st.session_state.messages = [AIMessage(content="Hi! I'm your AI repair advisor. Ask me about repair times (FRT) or for parts advice.")]
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hi! I'm your AI repair advisor. Ask me about repair times (FRT) or for parts advice.")]

# Display chat messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Main chat input loop
if prompt := st.chat_input("Ask a question like 'How long for a CVT overhaul?'"):
    if not google_api_key:
        st.warning("Please enter your Google Gemini API key in the sidebar first.")
    else:
        # Initialize model with API key
        model = get_model(google_api_key)
        if model:
            # Load PDF data on first run
            try:
                load_and_process_pdf()
            except Exception as e:
                st.error(f"Failed to load or parse the PDF: {e}")
                st.stop()


            # Append user message to history
            st.session_state.messages.append(HumanMessage(content=prompt))
            st.chat_message("user").write(prompt)

            # Define the graph
            workflow = StateGraph(AgentState)
            workflow.add_node("agent", lambda state: call_model(state, model))
            workflow.add_node("action", call_tool)
            workflow.set_entry_point("agent")
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "continue": "action",
                    "end": END,
                },
            )
            workflow.add_edge("action", "agent")
            app = workflow.compile()

            # Invoke the graph
            with st.spinner("Thinking..."):
                final_state = app.invoke({"messages": st.session_state.messages})
                response_message = final_state["messages"][-1]

            # Update session state with the new message
            st.session_state.messages.append(response_message)
            st.chat_message("assistant").write(response_message.content)
            st.rerun()

