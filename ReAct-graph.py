import gradio as gr
import os
import json
import base64
import re
import io
import uuid
from azure.core.credentials import AzureKeyCredential
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal, Dict, List, Any, TypedDict, Annotated, Union, Type, Optional, Tuple
import operator
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from models import GraphState, RouterInitialDecision, RouterDecision, ButtonList, CheckboxList, ComboboxList, IconList, SwitchList, TextboxList, URLList, CalendarList, Button, Checkbox, Combobox, Icon, Switch, Textbox, URL, Calendar
from PIL import Image
 
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

            # Ensure analysis_required is a list (handle LLM variability)
            if "analysis_required" in response_json and isinstance(response_json["analysis_required"], str):
                 print("Adjusting analysis_required from string to list.")
                 response_json["analysis_required"] = [response_json["analysis_required"]]


            # Validate against the Pydantic model
            decision = RouterDecision.model_validate(response_json)
            print("Pydantic validation successful!")

            # --- Success Case ---
            # Return the updates to the state
            return {
                "device": decision.device,
                "analysis_required": [decision.target_element_type] if decision.target_element_type else [],
                "messages": messages, # Return the final message history
                "router_target_element_type": decision.target_element_type
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
                f"(device: 'ios'|'android', analysis_required: str) "
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
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], Button, state["analysis_required"])

def checkbox_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
     # ...
    agent_name = "Checkbox"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], Checkbox, state["analysis_required"])


def calendar_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Calendar"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], Calendar, state["analysis_required"])


def textbox_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Textbox"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], Textbox, state["analysis_required"])


def url_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Url"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], URL, state["analysis_required"])


def icon_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Icon"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], Icon, state["analysis_required"])

def combobox_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Combobox"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], Combobox, state["analysis_required"])

def switch_agent_node(state: GraphState) -> Dict[str, Any]: # <-- Change here
    # ...
    agent_name = "Switch"
    return call_special_agent_and_parse(agent_name, SystemMessage(content=SYSTEM_PROMPTS[agent_name]), state["messages"][1], Switch, state["analysis_required"])
    

def call_special_agent_and_parse(agent_name: str, system_prompt:SystemMessage, human_request: HumanMessage, return_type:Type[BaseModel], current_list) -> Dict[str, Any]:
    result = specialized_agent(agent_name,  system_prompt, human_request, return_type)

        # 3. Process the result and prepare state updates
    update_dict = {}
    if "error" in result:
        print(f"Specialized agent for {agent_name} reported an error: {result['error']}")
        error_object = {
            "error": f"Error in {agent_name} agent",
            "details": result["error"],
        }
        update_dict["agent_analysis"] = json.dumps(error_object)
        # update_dict["messages"] = state["messages"] + [AIMessage(content=f"Error during {agent_name} analysis.")]
    else:
        update_dict["agent_analysis"] = json.dumps(result["decision"])
        # update_dict["messages"] = state["messages"] + [AIMessage(content=f"{agent_name} analysis complete.")]


    # 4. ALWAYS update analysis_required list
    # Use lower() for case-insensitive comparison if needed
    updated_needed = [item for item in current_list if item.lower() != agent_name.lower()]
    print(f"Updating analysis_required from {current_list} to {updated_needed}")
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
                "decision": decision.model_dump(mode='json', exclude_none=False),
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
    Determines the next specialist agent to call based on analysis_required,
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

def compare_lists_flexible(llm_list: List, golden_list: List, field_name: str) -> bool:
    if len(llm_list) != len(golden_list):
        return False
    can_sort_by_id = False
    if llm_list and golden_list:
        first_llm = llm_list[0]
        first_golden = golden_list[0]
        if isinstance(first_llm, dict) and isinstance(first_golden, dict) and 'id' in first_llm and 'id' in first_golden:
            can_sort_by_id = True
        elif hasattr(first_llm, 'id') and hasattr(first_golden, 'id'):
             can_sort_by_id = True
    if can_sort_by_id:
        try:
            llm_sorted = sorted(llm_list, key=lambda x: x['id'] if isinstance(x, dict) else x.id)
            golden_sorted = sorted(golden_list, key=lambda x: x['id'] if isinstance(x, dict) else x.id)
            return llm_sorted == golden_sorted
        except (TypeError, KeyError, AttributeError):
             pass
    return llm_list == golden_list


def compare_json_analysis_dict_vs_golden_str(
    llm_fields_dict: Optional[Dict[str, Any]], # LLM's analysis as a Python dictionary
    golden_analysis_json_str: str,
    expected_agent_type_for_golden_validation: Optional[str] = None # Optional: for validating golden JSON
) -> Tuple[float, Dict[str, str]]:
    """
    Compares LLM's analysis dictionary with a golden JSON string, field by field,
    based on the keys present in the golden JSON.
    """
    details = {}
    correct_field_count = 0
    total_field_count = 0

    if not llm_fields_dict: # If LLM produced no parsable analysis
        llm_fields_dict = {} # Treat as empty for comparison

    if not golden_analysis_json_str.strip():
        details["overall"] = "Golden JSON is empty."
        is_llm_empty = not bool(llm_fields_dict) # Check if the dict is empty
        if is_llm_empty:
            details["overall"] = "Match: Both Golden JSON and LLM output are effectively empty."
            return 1.0, details
        else:
            details["overall"] = "Mismatch: Golden JSON empty, LLM output not."
            return 0.0, details

    try:
        golden_data_dict = json.loads(golden_analysis_json_str)
    except json.JSONDecodeError as e:
        details["error"] = f"Invalid Golden JSON: {e}"
        return 0.0, details

    if not isinstance(golden_data_dict, dict):
        details["error"] = "Golden JSON is not a valid JSON object (dictionary)."
        return 0.0, details
    
    if not golden_data_dict: # Handles case where golden JSON is "{}"
        details["overall"] = "Golden JSON is an empty object."
        is_llm_empty = not bool(llm_fields_dict)
        if is_llm_empty:
            details["overall"] = "Match: Both Golden JSON and LLM output are effectively empty."
            return 1.0, details
        else:
            details["overall"] = "Mismatch: Golden JSON empty, but LLM output has data."
            return 0.0, details
        
    for golden_key, golden_value in golden_data_dict.items():
        total_field_count += 1
        field_comparison_key = f"Field '{golden_key}'"

        if golden_key not in llm_fields_dict:
            details[field_comparison_key] = "Mismatch (Key missing in LLM output)"
        else:
            llm_value = llm_fields_dict[golden_key]
            if isinstance(golden_value, list) and isinstance(llm_value, list):
                if compare_lists_flexible(llm_value, golden_value, golden_key):
                    details[field_comparison_key] = "Match"
                    correct_field_count += 1
                else:
                    details[field_comparison_key] = f"Mismatch (LLM: {llm_value}, Golden: {golden_value})"
            elif llm_value == golden_value: # Direct comparison for non-list types
                details[field_comparison_key] = "Match"
                correct_field_count += 1
            else:
                details[field_comparison_key] = f"Mismatch (LLM: {llm_value}, Golden: {golden_value})"

    if total_field_count == 0:
        accuracy = 1.0
        details["overall"] = "No fields in Golden JSON to compare."
    else:
        accuracy = correct_field_count / total_field_count

    return accuracy, details



def perform_comparisons(
    llm_device: Optional[str],
    llm_router_decision: Optional[str], # Router's decision (e.g., "Button", "None")
    llm_analysis_result_dict: Optional[Dict[str, Any]], # Parsed JSON from agent_analysis
    expected_os: str,
    expected_agent_type: str, # String like "Button", "Checkbox", or "None"
    golden_analysis_json_str: str
) -> Tuple[str, str, str]:
    """
    Performs the OS, Agent Type, and JSON Analysis comparisons.
    llm_analysis_result_dict is the parsed dictionary from state.agent_analysis.
    """
    # --- 1. OS Comparison ---
    os_result_str = ""
    if not llm_device:
        os_result_str = "Mismatch: LLM did not determine OS."
    elif llm_device.lower() == expected_os.lower():
        os_result_str = f"Match: Both are '{expected_os}'."
    else:
        os_result_str = f"Mismatch: LLM='{llm_device}', Expected='{expected_os}'."

    # --- 2. Agent Type Comparison ---
    agent_type_result_str = ""
    # Handle the "None" expectation
    expected_is_none = expected_agent_type == "None"
    llm_decision_is_none = llm_router_decision is None or llm_router_decision.lower() == "none"

    if expected_is_none and llm_decision_is_none:
         agent_type_result_str = "Match: Both correctly identified no specific agent needed."
    elif expected_is_none and not llm_decision_is_none:
         agent_type_result_str = f"Mismatch: Expected 'None', but LLM triggered '{llm_router_decision}'."
    elif not expected_is_none and llm_decision_is_none:
         agent_type_result_str = f"Mismatch: Expected '{expected_agent_type}', but LLM triggered 'None'."
    # Both expect a specific type, compare them (case-insensitive)
    elif expected_agent_type.lower() == llm_router_decision.lower():
        agent_type_result_str = f"Match: Both identified '{expected_agent_type}'."
    else:
        agent_type_result_str = f"Mismatch: Expected='{expected_agent_type}', LLM Triggered='{llm_router_decision}'."
    
    # --- 3. JSON Analysis Comparison ---
    analysis_result_str = "N/A"

    agent_types_match_and_specific_expected = (
        not expected_is_none and
        not llm_decision_is_none and
        expected_agent_type.lower() == llm_router_decision.lower()
    )

    if agent_types_match_and_specific_expected:
        if not llm_analysis_result_dict: # Check if the dict from agent_analysis is None or empty
             analysis_result_str = f"Mismatch: Agent type '{expected_agent_type}' matched, but no analysis data found in LLM state (agent_analysis was empty or unparsable)."
        else:
            # Now call compare_json_analysis with the llm_analysis_result_dict
            accuracy, details = compare_json_analysis_dict_vs_golden_str(
                llm_analysis_result_dict, # Pass the dictionary
                golden_analysis_json_str,
                expected_agent_type # Pass for context, e.g., validating golden JSON
            )
            if "error" in details:
                analysis_result_str = details["error"]
            else:
                analysis_result_str = f"Accuracy: {accuracy:.2f}\nDetails:\n" + "\n".join(
                    f"- {k}: {v}" for k, v in details.items()
                )
    # ... (rest of the N/A conditions for analysis_result_str remain the same) ...
    elif expected_is_none and llm_decision_is_none:
        analysis_result_str = "N/A (No specific agent analysis expected or triggered, as per match)"
    elif expected_is_none and not llm_decision_is_none:
        analysis_result_str = f"N/A (Expected no agent, but LLM triggered {llm_router_decision})"
    elif not expected_is_none and llm_decision_is_none:
        analysis_result_str = f"N/A (Expected {expected_agent_type}, but LLM triggered no agent)"
    else: # Agent types mismatched
        analysis_result_str = "N/A (Agent types mismatched, so analysis content not compared)"

    return os_result_str, agent_type_result_str, analysis_result_str




def run_and_compare(
    image_input, human_request, # Agent inputs
    expected_os, expected_agent_type, golden_analysis_json_str # Golden inputs
):
    print("--- Received Inputs ---")
    print(f"Expected OS: {expected_os}")
    print(f"Expected Agent Type: {expected_agent_type}")
    print(f"Golden JSON String:\n{golden_analysis_json_str}")
    if image_input is None:
        return "Error: No image provided.", "", "", ""
    try:
        final_state = process_image_with_graph(image_input, human_request)
        print(f"Type of final_state is {type(final_state)}")
        print(final_state)
        llm_device = final_state.get("device")
        llm_router_decision = final_state.get("router_target_element_type") # This is a string or None

        # Get the raw JSON string from the state
        llm_agent_analysis_str = final_state.get("agent_analysis")
        llm_analysis_result_dict = None # This will be the parsed dict from agent_analysis

        llm_output_display = "No specific agent analysis found in state."
        if llm_agent_analysis_str:
            try:
                llm_analysis_result_dict = json.loads(llm_agent_analysis_str)
                # Pretty print for display
                llm_output_display = json.dumps(llm_analysis_result_dict, indent=2)
            except json.JSONDecodeError:
                llm_output_display = f"Error: Could not parse agent_analysis JSON: {llm_agent_analysis_str}"
                # llm_analysis_result_dict remains None
    except Exception as e:
        print(f"Error during graph execution: {e}")
        return f"Error during graph execution: {e}", "", "", ""
    
    try:
        print(f"Type of analysis result is {type(llm_analysis_result_dict)}")
        print(f"Type of the golden json string is {type(golden_analysis_json_str)}")
        os_comparison_result, agent_type_comparison_result, analysis_comparison_result = perform_comparisons(
            llm_device,
            llm_router_decision, # Pass the router's decision string
            llm_analysis_result_dict, # Pass the parsed dictionary (or None)
            expected_os,
            expected_agent_type,
            golden_analysis_json_str
        )
    except Exception as e:
         print(f"Error during comparison: {e}")
         os_comparison_result = "Comparison Error"
         agent_type_comparison_result = "Comparison Error"
         analysis_comparison_result = f"Error during comparison: {e}"

    return (
        llm_output_display,
        os_comparison_result,
        agent_type_comparison_result,
        analysis_comparison_result
    )

# Function to process the image through our agent graph
def process_image_with_graph(image, human_request) -> dict: # model_choice is removed
    try:
        # Determine which model to use
        use_90b = True #set to True since for now we use only 1 model
        
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
            "analysis_required" : [],
            "router_target_element_type": None,
            "agent_analysis": None,
            "messages": [], #Initialize as empty
            "final_response" : None
        }
        
        # Build and run the graph
        workflow = build_graph()
        print("Graph has been compiled.")
        final_state = workflow.invoke(initial_state)
        return final_state
    
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


def flag_example(
    image, request, exp_os, exp_agent, golden, llm, os_cmp, agent_cmp,
    analysis, reason
):
    os.makedirs("flagged_examples", exist_ok=True)
    uid = uuid.uuid4().hex
    # serialize image as base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    data = {
        "image": img_b64,
        "request": request,
        "expected_os": exp_os,
        "expected_agent_type": exp_agent,
        "golden_json": golden,
        "llm_json": llm,
        "os_comparison": os_cmp,
        "agent_type_comparison": agent_cmp,
        "analysis_accuracy": analysis,
        "flag_reason": reason,
    }
    with open(f"flagged_examples/{uid}.json", "w") as f:
        json.dump(data, f, indent=2)
    return "✅ Example flagged!"

def list_flagged_files():
    """Return a sorted list of all .json filenames in flagged_examples/"""
    folder = "flagged_examples"
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".json"))

def load_flagged_json(filename):
    """Given a filename, load and pretty-print its JSON contents."""
    path = os.path.join("flagged_examples", filename)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error loading {filename}: {e}"

def load_flagged_examples_for_gallery():
    """
    Scans the `flagged_examples/` folder for .json files,
    decodes each image, thumbnails it, and returns a list of
    (PIL.Image, caption) tuples for display in a Gallery.
    """
    folder = "flagged_examples"
    if not os.path.isdir(folder):
        return []
    items = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(folder, fname)
        with open(path, "r") as f:
            data = json.load(f)
        # Decode base64 → PIL Image
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((100, 100))
        # Build a small markdown caption
        lines = []
        for k, v in data.items():
            if k == "image":
                continue
            # pretty‐print nested JSON if needed
            val = json.dumps(v, indent=2) if isinstance(v, dict) else v
            lines.append(f"**{k.replace('_',' ').title()}:** {val}")
        caption = "\n\n".join(lines)
        items.append((img, caption))
    return items

def load_flagged_examples_html():
    """
    Scans `flagged_examples/` for .json files, decodes each image,
    and builds a chunk of HTML where each example is rendered as:
      [thumbnail]   [key: value<br>key: value<br>…]
    """
    folder = "flagged_examples"
    if not os.path.isdir(folder):
        return "<p><em>No flagged examples found.</em></p>"

    snippets = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(folder, fname)
        data = json.load(open(path))

        # We have the raw base64 from your saved JSON:
        img_b64 = data["image"]

        # Optionally re-decode + resize for even better performance:
        raw = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((400, 800))  # keep aspect ratio, max width=400px

        # Re-encode to base64 so we can embed the resized thumbnail:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        # Build metadata HTML
        meta_lines = []
        for k, v in data.items():
            if k == "image":
                continue
            pretty = json.dumps(v, indent=2) if isinstance(v, dict) else v
            meta_lines.append(f"<p><strong>{k.replace('_',' ').title()}:</strong> {pretty}</p>")

        snippet = f"""
        <div style="
          display: flex;
          align-items: flex-start;
          margin-bottom: 1.5rem;
          border-bottom: 1px solid #444;
          padding-bottom: 1rem;
        ">
          <img
            src="data:image/png;base64,{thumb_b64}"
            style="max-width: 300px; width: 100%; height: auto; margin-right: 1rem; border:1px solid #666;"
          />
          <div style="color: #eee; font-size: 0.9rem; line-height:1.4;">
            <p><em>Filename: {fname}</em></p>
            {"".join(meta_lines)}
          </div>
        </div>
        """
        snippets.append(snippet)

    # Wrap everything in a scrollable container
    html = f"""
    <div style="
      max-height: 80vh;
      overflow-y: auto;
      padding-right: 1rem;
    ">
      {''.join(snippets)}
    </div>
    """
    return html

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Agentic UI Analysis"):
            gr.Markdown("# Agentic UI Analysis & Evaluation")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Agent Inputs")
                    input_image = gr.Image(
                        type="pil", label="Upload Screenshot"
                    )
                    input_request = gr.Textbox(
                        lines=2,
                        label="Human Request",
                        placeholder="e.g., Click the save button",
                    )

                    gr.Markdown("## Golden Inputs (for Evaluation)")
                    input_expected_os = gr.Radio(
                        ["ios", "android"],
                        label="Expected OS",
                        value="android",
                    )
                    input_expected_agent_type = gr.Dropdown(
                        choices=[
                            "Button",
                            "Checkbox",
                            "Calendar",
                            "Icon",
                            "Combobox",
                            "Url",
                            "Textbox",
                            "Switch",
                            "None",
                        ],
                        label="Expected Agent Type",
                        value="None",
                    )
                    input_golden_json = gr.Code(
                        language="json",
                        lines=10,
                        label="Expected JSON Analysis Output "
                              "(for the specific agent type)",
                    )

                    run_button = gr.Button("Run Analysis & Compare")

                with gr.Column(scale=1):
                    gr.Markdown("## Agent Output")
                    output_llm_json = gr.Code(
                        language="json",
                        label="LLM Analysis JSON",
                        interactive=False,
                    )

                    gr.Markdown("## Evaluation Results")
                    output_os_comparison = gr.Textbox(
                        label="OS Classification Accuracy",
                        interactive=False,
                    )
                    output_agent_type_comparison = gr.Textbox(
                        label="Agent Type Trigger Accuracy",
                        interactive=False,
                    )
                    output_analysis_accuracy = gr.Textbox(
                        lines=5,
                        label="JSON Analysis Accuracy",
                        interactive=False,
                    )

            run_button.click(
                fn=run_and_compare,
                inputs=[
                    input_image,
                    input_request,
                    input_expected_os,
                    input_expected_agent_type,
                    input_golden_json,
                ],
                outputs=[
                    output_llm_json,
                    output_os_comparison,
                    output_agent_type_comparison,
                    output_analysis_accuracy,
                ],
            )

            flag_reason = gr.Dropdown(
                choices=[
                    "Wrong OS Classification",
                    "Wrong Agent Type Classification",
                    "Incorrect JSON Analysis",
                    "Other",
                ],
                label="Flag Reason",
            )
            flag_button = gr.Button("Flag This Run")

            flag_status = gr.Textbox(
                label="Flag Status", interactive=False
            )

            flag_button.click(
                fn=flag_example,
                inputs=[
                    input_image,
                    input_request,
                    input_expected_os,
                    input_expected_agent_type,
                    input_golden_json,
                    output_llm_json,
                    output_os_comparison,
                    output_agent_type_comparison,
                    output_analysis_accuracy,
                    flag_reason,
                ],
                outputs=flag_status,
            )

        with gr.TabItem("Flagged Examples"):
            gr.Markdown("## Browse Flagged Examples")
            refresh_btn = gr.Button("Refresh List")
            html_view = gr.HTML(value=load_flagged_examples_html())

            # Re-scan & re-render on demand
            refresh_btn.click(
                fn=load_flagged_examples_html,
                inputs=None,
                outputs=html_view
            )

if __name__ == "__main__":
    demo.launch()