import gradio as gr
import os
import json
import base64
import re
import io
from azure.core.credentials import AzureKeyCredential
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal, Dict, List, Any, TypedDict, Annotated, Union
import operator
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from models import GraphState, RouterInitialDecision, ButtonList, CheckboxList
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI API credentials from environment variables
llm_model_name = os.getenv("LLM_MODEL_NAME")
llm_api_key = os.getenv("AZURE_LLAMA_90B_VISION_API_KEY")
llm_11b_api_key = os.getenv("AZURE_LLAMA_11B_VISION_API_KEY")
llm_endpoint = os.getenv("LLM_END_POINT")
llm_api_version = os.getenv("LLM_API_VERSION")

# Loading prompts from their corresponding txt files and setting them to a dictionary.
def load_prompt_from_file(filepath):
    """Loads a prompt from a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

# Define the directory where the prompt files are stored.  Adjust if needed.
PROMPT_DIR = "agent-prompts"

# System prompt mapping
SYSTEM_PROMPTS = {
    "Calendar": load_prompt_from_file(os.path.join(PROMPT_DIR, "calendar_system_prompt.txt")),
    "Icon": load_prompt_from_file(os.path.join(PROMPT_DIR, "icon_system_prompt.txt")),
    "Combobox": load_prompt_from_file(os.path.join(PROMPT_DIR, "combobox_system_prompt.txt")),
    "Url": load_prompt_from_file(os.path.join(PROMPT_DIR, "url_system_prompt.txt")),
    "Button": load_prompt_from_file(os.path.join(PROMPT_DIR, "button_system_prompt.txt")),
    "Textbox": load_prompt_from_file(os.path.join(PROMPT_DIR, "textbox_system_prompt.txt")),
    "Checkbox": load_prompt_from_file(os.path.join(PROMPT_DIR, "checkbox_system_prompt.txt")),
    "Switch": load_prompt_from_file(os.path.join(PROMPT_DIR, "switch_system_prompt.txt")),
    "Router": load_prompt_from_file(os.path.join(PROMPT_DIR, "router_system_prompt.txt")),
}

def initialize_model(use_90b=True, json_response=True):
    """Initialize the Azure AI Chat model with the appropriate API key."""
    api_key = llm_api_key if use_90b else llm_11b_api_key
    
    kwargs = {}
    if json_response:
        kwargs["response_format"] = {"type": "json_object"}
    
    model = AzureAIChatCompletionsModel(
        model_name=llm_model_name,  
        endpoint=llm_endpoint,
        credential=AzureKeyCredential(api_key),
        api_version=llm_api_version,
        **kwargs
    )
    return model

# Function to extract JSON from text
def extract_json(text):
    """Extract JSON object from text that might contain additional content."""
    # Try to find JSON with regex pattern matching
    json_pattern = r'({[\s\S]*})'
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1)
        try:
            # Try to parse as JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If the above failed, try a more aggressive approach
    # Look for any text between curly braces
    braces_pattern = r'{[^{}]*}'
    matches = re.findall(braces_pattern, text)
    
    for potential_json in matches:
        try:
            # Make sure it's valid JSON
            return json.loads(potential_json)
        except json.JSONDecodeError:
            continue
    
    # If all else fails, try to find and fix common JSON issues
    # Example: replace single quotes with double quotes
    fixed_text = text.replace("'", '"')
    
    # Try to find and parse JSON one more time
    match = re.search(json_pattern, fixed_text)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If we still can't extract JSON, raise an exception
    raise ValueError(f"Could not extract valid JSON from: {text}")

# Router agent function
def router_agent(state: GraphState) -> GraphState:
    print("--- Entering Router Agent ---")
    max_retries = 3
    attempts = 0
    messages = list(state["messages"]) # Start with existing messages, if any

    model = initialize_model(use_90b=state["use_90b"], json_response=True)
    
    # Prepare initial messages if history is empty
    if not messages:
        print("Initializing message history.")
        human_content = [
            {"type": "text", "text": "Analyze the following UI screenshot."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{state['image_mime_type']};base64,{state['image_base64']}"},
            }
        ]
        initial_human_message = HumanMessage(content=human_content)
        system_message = SystemMessage(content=SYSTEM_PROMPTS["Router"]) # Ensure this prompt asks for JSON matching RouterInitialDecision
        messages.extend([system_message, initial_human_message])
    

    while attempts < max_retries:
        attempts += 1
        print(f"\nAttempt {attempts}/{max_retries}...")

        print("Calling LLM via API...")
        try:
            response = model.invoke(messages)
            print("LLM Response Received.")
            # Add AI response to history *immediately* for context in potential retries
            messages.append(AIMessage(content=response.content)) # Add the actual AIMessage object

            print("Raw LLM Response Content:")
            print(response.content)

            # --- Try parsing and validation ---
            print("Attempting to extract and validate JSON...")
            response_json = extract_json(response.content)
            print("JSON extracted successfully:")
            print(response_json)

            # Ensure analysis_needed is a list (handle LLM variability)
            if "analysis_required" in response_json and isinstance(response_json["analysis_required"], str):
                 print("Adjusting analysis_needed from string to list.")
                 response_json["analysis_required"] = [response_json["analysis_required"]]


            # Validate against the Pydantic model
            decision = RouterInitialDecision.model_validate(response_json)
            print("Pydantic validation successful!")

            # --- Success Case ---
            # Return the updates to the state
            return {
                "device": decision.device,
                "analysis_required": decision.analysis_required,
                "messages": messages # Return the final message history
            }

        except (ValueError, json.JSONDecodeError, Exception) as e: # Catch JSON errors and Pydantic validation errors
            print(f"Error during attempt {attempts}: {e}")
            if attempts >= max_retries:
                print("Maximum retries reached. Failing.")
                # Optionally raise the error or return a specific failure state
                return {"error": "Router failed after max retries",
                        "messages": messages
                }

            # --- Prepare Error Feedback for Retry ---
            print("Preparing error feedback for LLM retry...")
            error_feedback = (
                f"Your previous response could not be processed. "
                f"Error: {e}. "
                f"Please carefully review the initial request and the required JSON format "
                f"(device: 'ios'|'android', analysis_needed: list[str]) "
                f"and provide a valid JSON response. Ensure the entire response is only the JSON object."
                # Optional: Include Pydantic schema for more detail
                # f"Expected schema: {RouterInitialDecision.model_json_schema()}"
            )
            # Add the error feedback as a new HumanMessage for the next attempt
            messages.append(HumanMessage(content=error_feedback))
            print("Error feedback added to messages for next attempt.")

        # Loop continues for the next attempt

    # Should not be reached if max_retries > 0, but as a fallback
    print("Exiting router agent loop unexpectedly.")
    return {"messages": messages} # Return current messages if loop finishes weirdly

# Specialized agent function
def specialized_agent(state: GraphState) -> GraphState:
    # Get the list of screenshot types from classification
    screenshot_types = state["classification"]["screenshot_types"]
    
    # Check if we've processed all types
    if state["current_type_index"] >= len(screenshot_types):
        # We've processed all types, so combine the results
        combined_response = {
            "platform": state["classification"]["platform"],
            "elements": state["element_analyses"]
        }
        state["final_response"] = combined_response
        return state
    
    # Get the current type to process
    current_type = screenshot_types[state["current_type_index"]]
    
    # Get the appropriate system prompt
    system_prompt = SYSTEM_PROMPTS.get(current_type, "")
    
    if not system_prompt:
        # Skip unsupported types
        state["current_type_index"] += 1
        state["processed_types"].append(current_type)
        return state
    
    # Initialize the model with JSON response format
    model = initialize_model(use_90b=state["use_90b"], json_response=True)
    
    # Create the messages for the model
    # For Calendar type, we need to add a task parameter
    if current_type == "Calendar":
        # A placeholder task description for calendar analysis
        task_description = "Analyze the calendar and identify its current state and selected date."
        system_prompt = system_prompt.format(task=task_description, image_base64=state["image_base64"])
    
    human_message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{state['image_mime_type']};base64,{state['image_base64']}"},
            }
        ],
    )
    system_message = SystemMessage(content=system_prompt)
    
    # Get the response from the model
    response = model.invoke([system_message, human_message])
    
    # Save the raw response
    state["raw_specialized_responses"].append(response.content)
    
    try:
        # Try to extract JSON from the response
        element_analysis = extract_json(response.content)
        
        # Ensure the type field is present
        if "type" not in element_analysis:
            element_analysis["type"] = current_type
        
        # Add the analysis to our list
        state["element_analyses"].append(element_analysis)
        
    except Exception as e:
        # Handle any errors in parsing
        element_analysis = {
            "type": current_type,
            "error": str(e),
            "raw_response": response.content
        }
        state["element_analyses"].append(element_analysis)
    
    # Mark this type as processed
    state["processed_types"].append(current_type)
    state["current_type_index"] += 1
    
    return state

# Define the router function to decide next node
def router_decider(state: GraphState) -> str:
    # Get the list of screenshot types from classification
    screenshot_types = state["classification"]["screenshot_types"]
    
    # Check if we've processed all types
    if state["current_type_index"] >= len(screenshot_types):
        return "end"
    else:
        return "specialized_agent"

# Function to process the image through our agent graph
def process_image_with_graph(image, model_choice):
    try:
        # Determine which model to use
        use_90b = model_choice == "90B Model"
        
        # Extract the mime type, and base64 representation, of the PIL image.
        buffered = io.BytesIO()
        image_format = image.format.lower() if image.format else "png"
        # Get the MIME type
        mime_type = f"image/{image_format}"
        image.save(buffered, format=image_format)
        img_byte = buffered.getvalue()
        image_base64 = base64.b64encode(img_byte).decode('utf-8')
        
        # Initialize the state
        initial_state: GraphState = {
            "image_base64": image_base64,
            "image_mime_type": mime_type,
            "use_90b": use_90b,
            "device": None,
            "analysis_needed" : [],
            "button_analysis" : ButtonList(),
            "checkbox_analysis" : CheckboxList(),
            "messages": [], #Initialize as empty
            "final_response" : None
        }
        
        # Build and run the graph
        workflow = build_graph()
        print("Graph has been compiled.")
        final_state = workflow.invoke(initial_state)
        
        
        return (
            json.dumps(final_state["device"], indent=2),
            json.dumps(final_state["analysis_needed"], indent=2),
        )
    
    except Exception as e:
        return f"Error: {str(e)}", "Error processing image", "{}"

# Build the LangGraph
def build_graph():
    # Define the graph
    builder = StateGraph(GraphState)
    # Add nodes
    builder.add_node("router_agent", router_agent)
    # Set the entry point
    builder.set_entry_point("router_agent")
    builder.set_finish_point("router_agent")
    # Compile the graph
    return builder.compile()


# Create Gradio interface
with gr.Blocks(title="UI Analysis with LangGraph") as demo:
    gr.Markdown("# UI Analysis with LangGraph")
    gr.Markdown("Upload a screenshot of a mobile app UI for intelligent analysis. The system can detect multiple UI elements in a single screenshot.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload UI Screenshot")
            model_choice = gr.Radio(
                ["90B Model", "11B Model"], 
                label="Select Model", 
                value="90B Model"
            )
            submit_button = gr.Button("Analyze UI", variant="stop")
        
        with gr.Column(scale=1):
            image_type_output = gr.JSON(label="Detected Element Types")
            # analysis_output = gr.JSON(label="Detailed Analysis")
            # debug_output = gr.JSON(label="Debug Information (Raw Responses)", visible=False)
    
    submit_button.click(
        fn=process_image_with_graph,
        inputs=[image_input, model_choice],
        outputs=[image_type_output,] # analysis_output, debug_output
    )
    
# Launch the interface
if __name__ == "__main__":
    demo.launch()