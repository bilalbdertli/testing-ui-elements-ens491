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
from typing import Literal, Dict, List, Any, TypedDict, Annotated, Union, Type
import operator
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from models import GraphState, RouterInitialDecision, ButtonList, CheckboxList, ComboboxList, IconList, SwitchList, TextboxList, URLList, CalendarList
 
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI API credentials from environment variables
llm_model_name = os.getenv("VLM_MODEL_NAME")
llm_api_key = os.getenv("AZURE_LLAMA_4_MAVERICK_17B_API_KEY")
llm_11b_api_key = os.getenv("AZURE_LLAMA_11B_VISION_API_KEY")
llm_endpoint = os.getenv("VLM_END_POINT")
llm_api_version = os.getenv("LM_API_VERSION")

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
def router_agent(state: GraphState) ->  Dict[str, Any]:
    print("--- Entering Router Agent ---")
    max_retries = 3
    attempts = 0
    messages = list(state["messages"]) # Start with existing messages, if any

    model = initialize_model(use_90b=state["use_90b"], json_response=True)
    
    # Prepare initial messages if history is empty
    if not messages:
        print("Initializing message history.")
        human_content = [
            {"type": "text", "text": state['human_request']},
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

def button_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Button"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], ButtonList, state["analysis_required"])

def checkbox_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
     # ...
    agent_name = "Checkbox"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], CheckboxList, state["analysis_required"])


def calendar_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Calendar"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], CalendarList, state["analysis_required"])


def textbox_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Textbox"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], TextboxList, state["analysis_required"])


def url_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Url"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], URLList, state["analysis_required"])


def icon_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Icon"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], IconList, state["analysis_required"])

def combobox_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Combobox"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], ComboboxList, state["analysis_required"])

def switch_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Switch"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], SwitchList, state["analysis_required"])
    

def call_special_agent_and_parse(agent_name: str, system_prompt:SystemMessage, human_request: HumanMessage, return_type:Type[BaseModel], current_list):
    result = specialized_agent(agent_name,  system_prompt, human_request, return_type)

        # 3. Process the result and prepare state updates
    update_dict = {}
    if "error" in result:
        print(f"Specialized agent for {agent_name} reported an error: {result['error']}")
        update_dict[f"{agent_name.lower()}_analysis"] = SwitchList() # Store empty SwitchList
        # update_dict["messages"] = state["messages"] + [AIMessage(content=f"Error during {agent_name} analysis.")]
    else:
        update_dict[f"{agent_name.lower()}_analysis"] = result["decision"]
        # update_dict["messages"] = state["messages"] + [AIMessage(content=f"{agent_name} analysis complete.")]


    # 4. ALWAYS update analysis_needed list
    # Use lower() for case-insensitive comparison if needed
    updated_needed = [item for item in current_list if item.lower() != agent_name.lower()]
    print(f"Updating analysis_needed from {current_list} to {updated_needed}")
    update_dict["analysis_required"] = updated_needed

    # 5. Return the dictionary containing updates for GraphState
    return update_dict

# Specialized agent function
def specialized_agent(agent_name: str, agent_system_prompt: SystemMessage, human_prompt: HumanMessage, validation_model:Type[BaseModel]) -> Dict[str, Any]:
    print(f"--- Entering {agent_name} Agent ---")
    max_retries = 3
    attempts = 0
    messages = [agent_system_prompt, human_prompt]
    
    # Initialize the model with JSON response format
    model = initialize_model(use_90b=True, json_response=True)
    
    while attempts < max_retries:
        attempts += 1
        print(f"\nAttempt {attempts}/{max_retries}...")

        print(f"Calling {agent_name} LLM via API...")
        try:
            response = model.invoke(messages)
            # Add AI response to history *immediately* for context in potential retries
            messages.append(AIMessage(content=response.content)) # Add the actual AIMessage object

            print(f"Raw {agent_name} LLM Response Content:")
            print(response.content)

            # --- Try parsing and validation ---
            print("Attempting to extract and validate JSON...")
            response_json = extract_json(response.content)
            print("JSON extracted successfully:")
            print(response_json)


            # Validate against the Pydantic model
            decision = validation_model.model_validate(response_json)
            print("Pydantic validation successful!")

            # --- Success Case ---
            # Return the updates to the state
            return {
                "decision": decision,
                "messages": messages # Return the final message history
            }

        except (ValueError, json.JSONDecodeError, Exception) as e: # Catch JSON errors and Pydantic validation errors
            print(f"Error during {agent_name} attempt {attempts}: {e}")
            if attempts >= max_retries:
                print(f"{agent_name} maximum retries reached. Failing.")
                # Optionally raise the error or return a specific failure state
                return {
                    "error": f"{agent_name} agent failed after max retries: {e}",
                    "messages": messages
                }

            # --- Prepare Error Feedback for Retry ---
            print(f"Preparing error feedback for {agent_name} LLM retry...")
            error_feedback = (
                f"Your previous response could not be processed. "
                f"Error: {e}. "
                f"Please carefully review the initial request and the required JSON format "
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
    return {"error": f"{agent_name} agent finished unexpectedly.", "messages": messages} # Return current messages if loop finishes weirdly


def decide_next_step(state: GraphState) -> str:
    """
    Determines the next specialist agent to call based on analysis_needed,
    or signals completion if the list is empty.
    """
    print("--- Deciding Next Step ---")
    needed = state["analysis_required"]
    print(f"Analysis needed: {needed}")

    if not needed:
        print("Decision: All analyses complete. Routing to END.")
        return "__end__" # Special key indicating completion

    # Select the next agent type from the list
    next_agent_type = needed[0]
    print(f"Decision: Routing to '{next_agent_type}' agent.")
    return next_agent_type # e.g., "button", "checkbox"


# Function to process the image through our agent graph
def process_image_with_graph(image, model_choice, human_request):
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
            "human_request": human_request,
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
            json.dumps(final_state["analysis_required"], indent=2),
        )
    
    except Exception as e:
        return f"Error: {str(e)}", "Error processing image", "{}"

# Build the LangGraph
def build_graph():
    # Define the graph
    builder = StateGraph(GraphState)
    # Add nodes
    builder.add_node("router_agent", router_agent)
    builder.set_entry_point("router_agent")
    builder.add_node("decider", lambda state: None)
    builder.add_edge("router_agent", "decider")
    
    builder.add_node("Calendar", calendar_agent_node)
    builder.add_node("Icon", icon_agent_node)
    builder.add_node("Combobox", combobox_agent_node)
    builder.add_node("Url", url_agent_node)
    builder.add_node("Button", button_agent_node)
    builder.add_node("Textbox", textbox_agent_node)
    builder.add_node("Switch", switch_agent_node)
    builder.add_node("Checkbox", checkbox_agent_node)
    # Set the entry point
    
    builder.add_conditional_edges(
        "decider",         # Source node is now 'decider'
        decide_next_step,  # Function that returns the key of the next node
        {
            # Map the return value of decide_next_step to the next node's key
            "Calendar": "Calendar",
            "Icon": "Icon",
            "Combobox": "Combobox",
            "Url": "Url",
            "Button": "Button",
            "Textbox": "Textbox",
            "Switch": "Switch", 
            "Checkbox": "Checkbox",
            "__end__": END  # Special key mapping to the end of the graph
        }
    )
    builder.add_edge("Calendar", "decider")
    builder.add_edge("Icon", "decider")
    builder.add_edge("Combobox", "decider")
    builder.add_edge("Url", "decider")
    builder.add_edge("Button", "decider")
    builder.add_edge("Textbox", "decider")
    builder.add_edge("Switch", "decider")
    builder.add_edge("Checkbox", "decider")

    # Compile the graph
    return builder.compile()


# Create Gradio interface
with gr.Blocks(title="UI Analysis with LangGraph") as demo:
    gr.Markdown("# UI Analysis with LangGraph")
    gr.Markdown("Upload a screenshot of a mobile app UI for intelligent analysis. The system can detect multiple UI elements in a single screenshot.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload UI Screenshot")
            human_request = gr.Textbox(
                label="Enter your request",
                placeholder="e.g., Find all the buttons in the image.",
            )
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
        inputs=[image_input, model_choice, human_request],
        outputs=[image_type_output,] # analysis_output, debug_output
    )
    
# Launch the interface
if __name__ == "__main__":
    demo.launch()