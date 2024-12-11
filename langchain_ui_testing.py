import base64
import gradio as gr
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import getpass
import os

system_prompt = """
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

# List of available models
available_models = [
    "llama3.2-vision:11b-instruct-q4_K_M",
    "llama3.2-vision:11b",
    "llava:latest"
]

# Default selected model
default_model = "llama3.2-vision:11b-instruct-q4_K_M"


# Mapping of model names to their initialization functions
model_initializers = {
    "llama3.2-vision:11b-instruct-q4_K_M": lambda: ChatOllama(model="llama3.2-vision:11b-instruct-q4_K_M", temperature=0),
    "llama3.2-vision:11b": lambda: ChatOllama(model="llama3.2-vision:11b", temperature=0),
    "llava:latest": lambda: ChatOllama(model="llava:latest", temperature=0)
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
def query_llm(image_base64, model_name):
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
def process_image(image, model_name):
    image_base64 = convert_to_base64(image)
    description = query_llm(image_base64, model_name)
    return description

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="pil"), gr.Radio(choices=available_models, label="Select Model", value=default_model)],
    outputs="text",
    title="Calendar  using LLM",
    description="Upload an image to get a description from the LLM."
)

# Launch the interface
iface.launch()
