import gradio as gr
import os
import json
import base64
import re
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

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI API credentials from environment variables
llm_model_name = os.getenv("LLM_MODEL_NAME")
llm_api_key = os.getenv("AZURE_LLAMA_90B_VISION_API_KEY")
llm_11b_api_key = os.getenv("AZURE_LLAMA_11B_VISION_API_KEY")
llm_endpoint = os.getenv("LLM_END_POINT")
llm_api_version = os.getenv("LLM_API_VERSION")

# Define Enums
class ScreenshotType(str, Enum):
    CALENDAR = "Calendar"
    ICON = "Icon"
    COMBOBOX = "Combobox"
    URL = "url"
    BUTTON = "button"
    TEXTBOX = "textbox",
    SWITCH = "Switch"

class OSPlatform(str, Enum):
    IOS = "ios"
    ANDROID = "android"

# Define Pydantic Schema
class UIElement(BaseModel):
    screenshot_types: List[ScreenshotType] = Field(..., description="Types of UI elements present in the screenshot: Calendar, Icon, Combobox, url, button, or textbox")
    platform: OSPlatform = Field(..., description="Operating system: ios or android")

# System prompts for different agents
ROUTER_SYSTEM_PROMPT = """
You are an expert UI classifier. Analyze the provided screenshot of a mobile application and identify:
1. The types of UI elements present (Calendar, Icon, Combobox, url, button, or textbox)
2. The operating system platform (iOS or Android)

A screenshot may contain multiple UI element types. Identify ALL element types present.

Return your analysis in JSON format with the following structure:
{
  "screenshot_types": ["element_type1", "element_type2", ...],
  "platform": "ios" | "android"
}

Where "element_type" is one of: "Calendar", "Icon", "Combobox", "url", "button", "textbox", "Switch", "Checkbox".

Example for a screenshot with multiple elements:
{
  "screenshot_types": ["Calendar", "button", "textbox"],
  "platform": "android"
}

Example for a screenshot with a single element:
{
  "screenshot_types": ["Icon"],
  "platform": "ios"
}

IMPORTANT: Return ONLY the JSON object without any additional text, explanations, or formatting. The response should be a valid JSON that can be directly parsed.
"""

CALENDAR_SYSTEM_PROMPT = """
You are an expert in UI visual analysis.  
Given a labeled calendar image (provided as an image in base64) and a task description, extract specific details about the calendar's current state.  

Return a JSON object with the following keys:  
  - "type": "Calendar"
  - "state": the calendar's current state, either "open" (expanded) or "closed" (shrunk).  
  - "currently_selected_month": the exact name of the currently selected month.  
  - "currently_selected_month_id": the numeric label of the UI element capturing the currently selected month.  
  - "currently_selected_year": the currently selected year (in four digits).  
  - "currently_selected_year_id": the numeric label of the UI element capturing the currently selected year.  
  - "currently_selected_day": the currently selected day (a number from 1 to 31).  
  - "currently_selected_day_id": the numeric label of the UI element capturing the currently selected day.  
  - "decrease_button_element_id": the numeric label of the UI element for decreasing the date value.  
  - "increase_button_element_id": the numeric label of the UI element for increasing the date value.  

If the calendar is closed but still displays any relevant information, include those details in the JSON output.  

Transcribe the text exactly as it appears in the UI elements without translation or modification. The label IDs correspond to the numeric labels present in the image.  

Include only the specified fields in the JSON output and no additional data.  

Task description: {task}  
Image (in base64): {image_base64}  

Return only the JSON output without any additional text or explanations.
"""

ICON_SYSTEM_PROMPT  = """
You are an expert in UI visual analysis.

I will give you:
1) A mobile app screenshot (in base64)
2) A task description

**Your objective**: Identify **only** the icon(s) that are directly relevant to completing the given task.

- An icon is a small pictorial or symbolic element (e.g., left arrow for navigation, house icon for home screen, etc.).
- “id” of the icon is the numeric label as seen in the screenshot.
- If the icon has **visible text** (a label) in or near it (e.g. “Ana sayfa,” “Geri,” etc.), use **that exact text** as the icon’s label.
- If there is no visible text label, you may provide a short descriptive name (e.g. “back_arrow_icon”).
- Do **not** include icons that are irrelevant to the user’s current task.

––––––––––––––––––––––––––––––––––––––––––––––––
Return only the JSON output with exactly this structure:

```json
{
  "icon": [
    {
      "id": 1,
      "label": "visible_label_or_description"
    }
  ]
}
"""

COMBOBOX_SYSTEM_PROMPT  = """
You are an expert in UI visual analysis.

I will give you:
1) A mobile app screenshot (in base64)
2) A task description

**Your objective**: Identify only the combobox(es) that are directly relevant to completing the given task.

- A combobox is typically a dropdown or selectable list component with a visible text label and/or currently selected option.
- “id” of the combobox is the numeric label as seen in the screenshot.
- “label” is the text label or prompt for the combobox (if visible).
- “selectedOption” is the text of whichever option is currently selected in the combobox.
- “state” can be:
  - **"open"** if the list of options is expanded and visible
  - **"closed"** if the list is collapsed/not expanded
- If the combobox is **"open"**, include a `"visibleOptions"` array listing **each option exactly as seen** in the dropdown.
- If the combobox is **"closed"**, do **not** include the `"visibleOptions"` field.

Return only the JSON output with exactly this structure:

```json
{
  "combobox": [
    {
      "id": 1,
      "label": "Your combobox label",
      "selectedOption": "Currently selected option",
      "state": "open",
      "visibleOptions": [
        "Option 1",
        "Option 2"
      ]
    }
  ]
}
"""

SWITCH_SYSTEM_PROMPT="""
You are given:
1. A mobile app screenshot (in base64 format).
2. A task description.

**Goal:** Identify any toggle switch elements in the screenshot that are relevant to the given task.

**What is a Switch?**
- A switch is a GUI element that indicates whether a feature or setting is active or inactive.
- It typically appears as a sliding toggle. When the small circle is on the right (often with a color change), the switch is ON (active). When on the left (or grayed out), it is OFF (inactive).

**Things to Note:**
- A switch usually has an adjacent label or description (commonly to its left) that explains its function.
- Visual cues like the slider position and color are key to determining the switch’s state.

**Instructions:**
1. **Identify all switches** in the screenshot:
   - Locate the sliding toggle elements.
   - Determine each switch’s state (ON or OFF) based on its visual appearance.
2. **Capture the switch’s label:**
   - Extract the text next to or on the switch that describes its purpose.
3. **Determine relevance:**
   - Compare the switch’s label/description with the task description. If it controls a feature related to the task, include it.
4. **Return the output in JSON format:**
   - Use an array under the `"switch"` key.
   - Each object must include:
     - `"id"`: A numeric identifier (order of appearance).
     - `"label"`: The exact descriptive text of the switch.
     - `"value"`: The state of the switch, either `"on"` or `"off"`.
5. If no relevant switches are found, return:
   ```json
   { "switch": [] }
"""

URL_SYSTEM_PROMPT = """
You are a mobile web interface expert. Analyze the provided URL/link element from a mobile app.
Focus on identifying:
1. The presentation of the URL (full URL, shortened form, descriptive text)
2. Visual indicators of it being a link (color, underline, icon)
3. Tap target size and position
4. Context and purpose within the app
5. Security indicators if present

Return your analysis in JSON format with the following structure:
{
  "type": "url",
  "url_text": "The text of the URL as displayed",
  "presentation": "How the URL is presented",
  "visual_indicators": "Visual indicators of it being a link",
  "tap_target": "Analysis of tap target size and position",
  "context": "Context and purpose within the app",
  "security_indicators": "Security indicators if present",
  "summary": "A concise summary of the URL's presentation and usability"
}

Return only the JSON output without any additional text or explanations.
"""

BUTTON_SYSTEM_PROMPT = """
You are an expert in UI visual analysis.

I will give you:
1) A mobile app screenshot (in base64)
2) A task description

Please identify only the button(s) directly related to completing the task, and classify each button as either "enabled" or "disabled."

“id” of the button is the numeric label as seen in the screenshot.
––––––––––––––––––––––––––––––––––––––––––––––––
**How to identify a button’s state:**

1. **Enabled (Active)**:
   - Usually in the brand color or a clear, vibrant color (e.g., bright red, blue, etc.).
   - Appears fully opaque, not grayed out.
   - Looks clickable (normal or bold text, proper contrast).

2. **Disabled (Inactive)**:
   - Often grayed out, lighter, or partially transparent.
   - Text may be faint or subdued.
   - Does not look clickable.
   - **If a required checkbox is not checked (or a required condition is not met), the button remains disabled even if color differences are subtle.** 
     - For example, if a Terms & Conditions checkbox is mandatory to proceed and it’s not checked in the screenshot, the associated button must be classified as “disabled.”

3. **Important**:
   - Do not assume a visible button is automatically enabled. Verify it looks clickable and any required conditions (like checkboxes) are satisfied.
   - If uncertain, and a mandatory checkbox or field is unfilled, classify the button as "disabled."
   - Only return buttons relevant to the user’s task.

––––––––––––––––––––––––––––––––––––––––––––––––
Return only the JSON output with exactly this structure:

```json
{
  "button": [
    {
      "id": 1,
      "label": "My Button Label",
      "state": "enabled"
    }
  ]
}
"""

TEXTBOX_SYSTEM_PROMPT = """
You are given:
1. A mobile app screenshot (in base64 format).
2. A task description.

**Goal:** Identify any textbox elements in the screenshot that are relevant to the given task.

**What is a Textbox?**
- A textbox is a GUI element designed for text input, typically displayed as a rectangular field.
- It may have an associated label (positioned above, below, or adjacent to the textbox) or include placeholder text indicating its purpose.
- The textbox can show pre-entered text (its current value) or be empty, indicating no input.

**Things to Note:**
- The label or placeholder text provides critical context (e.g., "Name", "Email", "Search") that clarifies the textbox’s function.
- Relevance is determined by matching the textbox’s label or placeholder with the provided task description.

**Instructions:**
1. **Identify all textbox elements** in the screenshot:
   - Locate rectangular fields intended for text entry.
   - Look for any labels or placeholder text that indicate what information is expected.
2. **Capture details for each textbox:**
   - Extract the label or descriptive text associated with the textbox.
   - Extract the placeholder text (if available).
   - Record the current content of the textbox, or mark it as `"empty"` if no text is present.
3. **Determine relevance:**
   - Compare the textbox’s label/placeholder with the task description. Include only the textboxes that are pertinent to the task.
4. **Return the results in JSON format:**
   - Provide an array under the `"textbox"` key.
   - Each object in the array must include:
     - `"id"`: A numeric identifier (based on the order of appearance).
     - `"label"`: The exact descriptive text associated with the textbox.
     - `"placeholder"`: The placeholder text within the textbox, if any.
     - `"value"`: The current text content of the textbox, or `"empty"` if it is blank.
5. If no relevant textboxes are found, return:
   ```json
   { "textbox": [] }
"""
CHECKBOX_SYSTEM_PROMPT="""
You are given:
1. A mobile app screenshot (in base64 format).
2. A task description.

**Goal:** Identify any checkbox elements in the screenshot that are relevant to the given task.

**What is a Checkbox?**
- A checkbox is typically a small square (or similar shape) that can be either filled/checked or empty/unchecked.
- If the box is filled with a checkmark (or similar symbol), it is **checked**.
- If it is empty or unfilled, it is **unchecked**.

**Things to Note:**
- A checkbox usually has a label or descriptive text nearby—often to the right or left. In some layouts, the label may be above or below the checkbox.
- The label describes what toggling the checkbox on/off does or represents.
- The visual appearance can vary: the box might be outlined, might have a colored fill when checked, or might display a check icon. Regardless, the key point is whether it’s checked or unchecked.

**Instructions:**
1. **Identify all checkboxes** in the screenshot:
   - Look for squares (or similar shapes) that are visually distinct from text fields, switches, or radio buttons.
   - Confirm if they are checked (with a mark) or unchecked (empty).

2. **Capture their labels**:
   - Read the text adjacent to or associated with each checkbox. This text is the label describing the checkbox’s function or meaning.
   - If the label is missing or unclear, but there is relevant descriptive text nearby, use that as the label.

3. **Compare each checkbox’s label** with the given task description:
   - If a checkbox appears relevant to the task (i.e., it controls an option or setting that helps accomplish the task), include it in the output.

4. **Return the checkboxes** in **JSON** format:
   - Return an array of objects under the `"checkbox"` key.
   - Each object must have the following fields:
     - `"id"`: A numeric identifier (e.g., the order in which the checkbox appears in the screenshot).
     - `"label"`: The exact descriptive text associated with the checkbox.
     - `"value"`: The checkbox state, either `"checked"` or `"unchecked"`.

5. **If no relevant checkboxes exist**, return:
   ```json
   { "checkbox": [] }
"""
# System prompt mapping
SYSTEM_PROMPTS = {
    "Calendar": CALENDAR_SYSTEM_PROMPT,
    "Icon": ICON_SYSTEM_PROMPT,
    "Combobox": COMBOBOX_SYSTEM_PROMPT,
    "url": URL_SYSTEM_PROMPT,
    "button": BUTTON_SYSTEM_PROMPT,
    "textbox": TEXTBOX_SYSTEM_PROMPT,
    "Checkbox": CHECKBOX_SYSTEM_PROMPT,
    "Switch": SWITCH_SYSTEM_PROMPT,
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

# Function to convert image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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

# Define the state for our graph
class GraphState(TypedDict):
    image_path: str
    image_base64: str
    image_mime_type: str
    use_90b: bool
    classification: Dict[str, Any]
    element_analyses: List[Dict[str, Any]]
    processed_types: List[str]
    current_type_index: int
    final_response: Dict[str, Any]
    raw_router_response: str
    raw_specialized_responses: List[str]

# Router agent function
def router_agent(state: GraphState) -> GraphState:
    # Initialize the model for JSON response
    model = initialize_model(use_90b=state["use_90b"], json_response=True)
    
    # Create the messages for the model
    human_message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{state['image_mime_type']};base64,{state['image_base64']}"},
            }
        ],
    )
    system_message = SystemMessage(content=ROUTER_SYSTEM_PROMPT)
    
    # Get the response from the model
    response = model.invoke([system_message, human_message])
    
    # Save the raw response for debugging
    state["raw_router_response"] = response.content
    
    try:
        # Try to extract JSON from the response
        response_json = extract_json(response.content)
        
        # Ensure screenshot_types is a list
        if isinstance(response_json.get("screenshot_types"), str):
            response_json["screenshot_types"] = [response_json["screenshot_types"]]
        
        # Initialize element_analyses list and processed_types list
        state["element_analyses"] = []
        state["raw_specialized_responses"] = []
        state["processed_types"] = []
        state["current_type_index"] = 0
        
        # Update the state with classification
        state["classification"] = {
            "screenshot_types": response_json.get("screenshot_types", []),
            "platform": response_json.get("platform", "Unknown")
        }
        
    except Exception as e:
        # Handle any errors in parsing
        state["classification"] = {
            "screenshot_types": ["Unknown"],
            "platform": "Unknown",
            "error": str(e),
            "raw_response": response.content
        }
        state["element_analyses"] = []
        state["raw_specialized_responses"] = []
        state["processed_types"] = []
        state["current_type_index"] = 0
    
    return state

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
        
        # Save the uploaded image temporarily
        temp_path = "temp_image.png"
        image.save(temp_path)
        
        # Get the MIME type
        mime_type = f"image/{image.format.lower()}" if image.format else "image/png"
        
        # Convert the image to Base64
        image_base64 = encode_image_to_base64(temp_path)
        
        # Initialize the state
        initial_state: GraphState = {
            "image_path": temp_path,
            "image_base64": image_base64,
            "image_mime_type": mime_type,
            "use_90b": use_90b,
            "classification": {},
            "element_analyses": [],
            "processed_types": [],
            "current_type_index": 0,
            "final_response": {},
            "raw_router_response": "",
            "raw_specialized_responses": []
        }
        
        # Build and run the graph
        workflow = build_graph()
        final_state = workflow.invoke(initial_state)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        # Prepare debug info
        debug_info = {
            "raw_router_response": final_state["raw_router_response"],
            "raw_specialized_responses": final_state["raw_specialized_responses"]
        }
        
        return (
            json.dumps(final_state["classification"], indent=2),
            json.dumps(final_state["final_response"], indent=2),
            json.dumps(debug_info, indent=2)
        )
    
    except Exception as e:
        return f"Error: {str(e)}", "Error processing image", "{}"

# Build the LangGraph
def build_graph():
    # Define the graph
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("router_agent", router_agent)
    builder.add_node("specialized_agent", specialized_agent)
    
    # Add edges
    builder.add_conditional_edges(
        "router_agent",
        router_decider,
        {
            "specialized_agent": "specialized_agent",
            "end": END
        }
    )
    
    # Add conditional edges for the specialized agent to loop back to itself
    builder.add_conditional_edges(
        "specialized_agent",
        router_decider,
        {
            "specialized_agent": "specialized_agent",
            "end": END
        }
    )
    
    # Set the entry point
    builder.set_entry_point("router_agent")
    
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
            submit_button = gr.Button("Analyze UI")
        
        with gr.Column(scale=1):
            image_type_output = gr.JSON(label="Detected Element Types")
            analysis_output = gr.JSON(label="Detailed Analysis")
            debug_output = gr.JSON(label="Debug Information (Raw Responses)", visible=False)
    
    submit_button.click(
        fn=process_image_with_graph,
        inputs=[image_input, model_choice],
        outputs=[image_type_output, analysis_output, debug_output]
    )
    
    gr.Markdown("### LangGraph Workflow")
    gr.Markdown("""
    1. **Router Agent**: Classifies all UI element types and platform
    2. **Specialized Agents**: Provide in-depth analysis for each detected element type
    3. **Result Aggregation**: Combines all analyses into a single structured response
    """)
    
    # Add a visual representation of the flow
    gr.Markdown("""
    ```
    START → router_agent → specialized_agent (for type 1) 
                        → specialized_agent (for type 2)
                        → specialized_agent (for type 3)
                        → ...
                        → END
    ```
    """)

# Launch the interface
if __name__ == "__main__":
    demo.launch()