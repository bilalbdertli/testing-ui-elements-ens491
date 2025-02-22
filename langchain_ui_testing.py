import base64
import gradio as gr
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import getpass
import os

from dotenv import load_dotenv
load_dotenv()

calendar_prompt = """
Please analyze the labeled calendar image provided. The image contains UI elements with numeric labels and their corresponding text. Your task is to extract specific information about the calendar's current state.

Please provide the following details in JSON format:

[
    "state": "<open or closed>",
    "currently_selected_month": "<month-name>",
    "currently_selected_month_id": "<label id of UI element which captures the currently selected month>",
    "currently_selected_year": "<year in 4 digits>",
    "currently_selected_year_id": "<label id of UI element which captures the currently selected year>",
    "currently_selected_day": "<day in the range of 1 - 31>",
    "currently_selected_day_id": "<label id of UI element which captures the currently selected day>",
    "decrease_button_element_id": "<label id of decrease button>",
    "increase_button_element_id": "<label id of increase button>"
]
Notes:

Set "state" to "open" if the calendar is expanded, or "closed" if it is shrunk.
If the calendar is closed but still displays any of the open calendar properties (e.g., month, year), include those values in the JSON output.
Use the exact field names as provided.
Ensure that the values are accurately extracted from the labeled image.
The text within the UI elements may be in any language. Transcribe the text exactly as it appears, without translation or alteration.
The label IDs correspond to the numeric IDs in the image labels (e.g., "0", "1", etc.).
Include only the specified fields in the JSON output.
"""
combobox_prompt = """ 
You are given a webpage screenshot from Türkiye İş Bankasi’s “Hizli Kredi - Hesapla” page. Examine the screenshot closely and identify the visible icons and comboboxes related to selecting a loan type.

Your final answer must be a single JSON object with two top-level keys: "icon" and "combobox". Each key maps to an array. If you have no relevant icons or comboboxes, return empty arrays for them. Otherwise, list the ones that help accomplish the “select loan type” task.

For each icon you include, create an object with exactly two fields:
- "id": A numeric label identifying that icon as it appears in the screenshot. Use the actual numeric label from the screenshot.
- "label": The textual label visible for that icon exactly as shown in the screenshot. Do not invent, alter, or translate text. Reproduce it character-for-character, including spaces, punctuation, and any diacritical marks. If the icon’s label in the screenshot is, for example, “Türkiye İş Bankası Logo,” then use exactly “Türkiye İş Bankası Logo.”

For each combobox you include, create an object with exactly these fields:
- "id": The numeric label that identifies the combobox in the screenshot.
- "label": The visible textual label of the combobox exactly as shown in the screenshot. No changes, no additions.
- "selectedOption": The text of the currently selected option exactly as it appears in the screenshot. Again, no alterations or guesses.
- "state": Either "open" or "closed". Determine this by what you see in the screenshot. If the combobox is not visibly showing multiple options, set this to "closed".
- If "state" is "open", include a "visibleOptions" array listing all visible options. Each option must be exactly as seen on the screen. If "state" is "closed", do not include "visibleOptions" at all.

Do not include any additional fields, keys, or properties beyond those specified above. Do not provide any explanations, reasoning steps, or commentary in your final answer. Do not include numeric references for anything other than the "id" fields. Textual labels, selected options, and visible options must appear exactly as they do in the screenshot, without modifications.

Since the given task is “select loan type,” return only the icons and comboboxes that are necessary to perform this action. If, for example, there is a combobox labeled “Kredi Türü” that allows you to select a loan type, include it and its currently selected option exactly as shown. If there are other comboboxes or icons unrelated to choosing the loan type, do not include them.

Your final output must be the JSON object only, with no extra text, explanations, or commentary before or after it.
"""
# List of available models
available_models = [
    "llama3.2-vision:11b-instruct-q4_K_M",
    "llama3.2-vision:11b",
    "llava:latest",
    "janus-pro-7b"
]

# Default selected model
default_model = "llama3.2-vision:11b-instruct-q4_K_M"


# Mapping of model names to their initialization functions
model_initializers = {
    "llama3.2-vision:11b-instruct-q4_K_M": lambda: ChatOllama(model="llama3.2-vision:11b-instruct-q4_K_M", temperature=0),
    "llama3.2-vision:11b": lambda: ChatOllama(model="llama3.2-vision:11b", temperature=0),
    "llava:latest": lambda: ChatOllama(model="llava:latest", temperature=0),
    "janus-pro-7b": lambda: ChatOllama(model="erwan2/DeepSeek-Janus-Pro-7B", temperature=0)
}



def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Function to query the LLM
def query_llm(image_base64, model_name, system_prompt):
    llm = model_initializers[model_name]()

    def prompt_func(data):
        text = data["text"]
        image = data["image"]

        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image}",
        }

        content_parts = []

        text_part = {"type": "text", "text": text}

        content_parts.append(image_part)
        content_parts.append(text_part)

        return [HumanMessage(content=content_parts)]

    chain = prompt_func | llm | StrOutputParser()

    query_chain = chain.invoke(
        {"text": system_prompt, "image": image_base64}
    )
    return query_chain

# Function to handle the image input and return the LLM response
def process_image(image, model_name, task_type):
    image_base64 = convert_to_base64(image)
    # Select which prompt to use based on the task_type
    if task_type == "Calendar":
        chosen_prompt = calendar_prompt
    else:
        chosen_prompt = combobox_prompt
    description = query_llm(image_base64, model_name, chosen_prompt)
    return description

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=[
    gr.Image(type="pil"),
    gr.Radio(choices=available_models, label="Select Model", value=default_model),
    gr.Radio(choices=["Calendar", "Combobox"], label="Task Type", value="Calendar")
    ],
    outputs="text",
    title="UI Analysis with LLM",
    description="Upload an image and select a task type to extract UI information (Calendar or Combobox)."
)

# Launch the interface
iface.launch()
