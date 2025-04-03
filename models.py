# Pydantic models should go here, we will add each expected JSON object as a Pydantic model to here.
# Firstly a single model subclass of BaseClass, then a model that is a nullable list of the single model
# Repeat for all of the prompts.

from typing import Literal, Optional, Union, Any, List, Dict
from pydantic import BaseModel, Field, conint, conlist
from typing_extensions import TypedDict, List
from langchain_core.messages import BaseMessage
class Button(BaseModel):

    id: conint(ge=0) = Field(... ,  # type: ignore
        description="The non-negative integer label ID from the input image.") 
    
    label: Optional[str] = Field(default= None, description="The detected text label on the button, if any.")  

    state: Literal["enabled", "disabled"]  = Field(...,
        description="The inferred state of the button (enabled or disabled)."
    )

class ButtonList(BaseModel):
    buttons: list[Button] = Field(
        default_factory=list, description="List of detected button objects.")

class Checkbox(BaseModel):
    id: conint(ge=0) = Field(..., # type: ignore
        description="The non-negative integer label ID from the input image.") 
    
    label: Optional[str] = Field(
        default=None,
        description="The detected text label associated with the checkbox, if any."
    )
    value: Literal["checked", "unchecked"] = Field(
        ...,
        description="The inferred value of the checkbox (checked or unchecked)."
    )

class CheckboxList(BaseModel):
    checkboxes: list[Checkbox] = Field(
        default_factory=list,
        description="List of detected checkbox objects."
    )

class Combobox(BaseModel):
    id: conint(ge=0) # type: ignore
    label: Optional[str] = None
    state: Literal["open", "closed"]
    visibleOptions: list[str] = Field(..., min_length=1)
    selectedOption: str

class ComboboxList(BaseModel):
    comboboxes: list[Combobox] = Field(default_factory=list)

class Icon(BaseModel):
    id: conint(ge=0) # type: ignore
    label: str

class IconList(BaseModel):
    icons: list[Icon] = Field(default_factory=list)

class Switch(BaseModel):
    id: conint(ge=0) # type: ignore
    label: str
    state: Literal["on", "off"]

class SwitchList(BaseModel):
    switches: list[Switch] = Field(default_factory=list)

class Textbox(BaseModel):
    id: conint(ge=0) # type: ignore
    label: str
    placeholder: Optional[str] = None
    value: Optional[str] = None

class TextboxList(BaseModel):
    textboxes: list[Textbox] = Field(default_factory=list)

class URL(BaseModel):
    id: conint(ge=0) # type: ignore
    url_text: str
    presentation: Literal["full url", "shortened", "descriptive text"]

class URLList(BaseModel):
    urls: list[URL] = Field(default_factory=list)

class ClosedCalendar(BaseModel):
    id: conint(ge=0) # type: ignore
    state: Literal["closed"]

class OpenCalendar(BaseModel):
    id: conint(ge=0) # type: ignore
    state: Literal["open"]
    selected_month: str
    selected_month_id: conint(ge=0) # type: ignore
    selected_year: conint(ge=1000, le=2100) # type: ignore
    selected_year_id: conint(ge=0) # type: ignore
    selected_day: conint(ge=1, le=31) # type: ignore
    selected_day_id: conint(ge=0) # type: ignore
    decrease_button_id: Optional[conint(ge=0)] = None # type: ignore
    increase_button_id: Optional[conint(ge=0)] = None # type: ignore

# Calendar can be either an OpenCalendar or a ClosedCalendar
Calendar = Union[OpenCalendar, ClosedCalendar]

class CalendarList(BaseModel):
    calendars: list[Calendar] = Field(default_factory=list)



class ButtonSubgraphState(TypedDict):
    button_analysis: ButtonList


class RouterInitialDecision(BaseModel):
    """
    Represents the decision made by the router agent based on the input image.
    """
    device: Literal["ios", "android"] = Field(
        ..., # Make it required, the router must decide
        description="The detected operating system of the device shown in the image."
    )
    analysis_required: List[Literal["Button", "Checkbox", "Calendar", "Icon", "Combobox", "Url", "Textbox", "Switch"]] = Field(
        ..., # Required, can be empty if the image is not related
        description="A list of UI element types identified in the image that require further analysis."
    )



class GraphState(TypedDict):
    image_base64: str # Base64 encoded representation of input UI screenshot.
    image_mime_type: str # The MIME type of the input image
    use_90b: bool # Config flag
    device: Optional[Literal["ios", "android"]]
    analysis_needed: List[str]
    button_analysis: ButtonList
    checkbox_analysis: CheckboxList
    messages: List[BaseMessage]
    final_response: Optional[Dict[str, Any]]


class RouterRealResponse(BaseModel):
    device: Literal["ios", "android"]
    buttons: ButtonList
    checkboxes: CheckboxList
    comboboxes: ComboboxList
    icons:IconList
    switches: SwitchList
    textboxes: TextboxList
    urls: URLList
    calendars:CalendarList


