# Pydantic models should go here, we will add each expected JSON object as a Pydantic model to here.
# Firstly a single model subclass of BaseClass, then a model that is a nullable list of the single model
# Repeat for all of the prompts.

from typing import Literal, Optional, Union, Any, List, Dict
from pydantic import BaseModel, Field, conint, conlist, field_validator, ValidationInfo, model_validator
from typing_extensions import TypedDict, List
from langchain_core.messages import BaseMessage
from datetime import date, datetime
# Define the list of known element types for reusability
ELEMENT_TYPES = Literal["Button", "Checkbox", "Calendar", "Icon", "Combobox", "Url", "Textbox", "Switch"]
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
    """Base class for combobox elements with common fields."""
    label: Optional[str] = Field(
        default=None,
        description="The text label associated with the combobox, if visible."
    )
    state: Literal["open", "closed"] = Field(
        ...,
        description="The current state of the combobox: 'open' (options visible) or 'closed'."
    )
    multiple_selection_allowed: bool = Field(
        ...,
        description="Whether multiple options can be selected (True) or only a single option (False)."
    )
    visibleOptions: Optional[List[str]] = Field(
        default=None,
        description="A list of the text of currently visible options. Should only be present if state is 'open'."
    )
    search_bar_id: Optional[int] = Field( 
        default=None,
        ge=0,
        description="The label ID of the search bar within the combobox, if present."
    )
    
    @model_validator(mode='after')
    def check_visible_options_state(self) -> 'Combobox':
        """Validate that visibleOptions is only present when state is 'open'."""
        if self.state == "closed" and self.visibleOptions is not None:
            raise ValueError("visibleOptions should be null or omitted when state is 'closed'")
        if self.state == "open" and self.visibleOptions is None:
            print("Warning: visibleOptions is None but state is 'open'.")
            raise ValueError("visibleOptions must be provided when state is 'open'")
        return self
    
    @classmethod
    def model_validate(cls, data: dict, *args, **kwargs) -> Union['SingleSelectCombobox', 'MultiSelectCombobox']:
        """Factory method to validate and instantiate the correct combobox type."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
            
        if "multiple_selection_allowed" not in data:
            raise ValueError("Data must contain 'multiple_selection_allowed' field")
            
        if data["multiple_selection_allowed"] is True:
            # Use BaseModel's model_validate to avoid recursion
            return MultiSelectCombobox.__pydantic_validator__.validate_python(data, *args, **kwargs)
        else:
            # Use BaseModel's model_validate to avoid recursion
            return SingleSelectCombobox.__pydantic_validator__.validate_python(data, *args, **kwargs)

        
class SingleSelectCombobox(Combobox):
    """Represents a combobox where only one option can be selected."""
    label: Optional[str] = Field(
        default=None,
        description="The text label associated with the combobox, if visible."
    )
    state: Literal["open", "closed"] = Field(
        ...,
        description="The current state of the combobox: 'open' (options visible) or 'closed'."
    )
    multiple_selection_allowed: Literal[False] = Field( # Fixed to False
        ...,
        description="Indicates that only a single option can be selected."
    )
    selectedOption: Optional[str] = Field(
        default=None,  # Use default=None instead of ...
        description="The text of the currently selected option, or null if none is selected.",
        exclude=False  # This ensures the field is always included in serialization
    )
    visibleOptions: Optional[List[str]] = Field(
        default=None,
        description="A list of the text of currently visible options. Should only be present if state is 'open'."
    )
    search_bar_id: Optional[int] = Field( 
        default=None,
        ge=0,
        description="The label ID of the search bar within the combobox, if present."
    )
    close_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of the button used to close the calendar (e.g., 'OK', 'Done', 'X'), if present."
    )
    cancel_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of the button used to cancel the selection and close the calendar, if present."
    )
    @model_validator(mode='after') # Run after standard validation
    def check_visible_options_state(self) -> 'SingleSelectCombobox':
        if self.state == "closed" and self.visibleOptions is not None:
            raise ValueError("visibleOptions should be null or omitted when state is 'closed'")
        if self.state == "open" and self.visibleOptions is None:
            # Could raise error, or just allow it if LLM might miss it
            print("Warning: visibleOptions is None but state is 'open'. LLM might have missed options.")
            raise ValueError("visibleOptions must be provided when state is 'open'")
        return self

class MultiSelectCombobox(BaseModel):
    """Represents a combobox where multiple options can be selected."""
    label: Optional[str] = Field(
        default=None,
        description="The text label associated with the combobox, if visible."
    )
    state: Literal["open", "closed"] = Field(
        ...,
        description="The current state of the combobox: 'open' (options visible) or 'closed'."
    )
    multiple_selection_allowed: Literal[True] = Field( # Fixed to True
        ...,
        description="Indicates that multiple options can be selected."
    )
    selectedOption: Optional[List[str]] = Field(
        default_factory=list,
        description="A list containing the text of all currently selected options. Can be empty.",
        exclude=False  # This ensures the field is always included in serialization
    )
    visibleOptions: Optional[List[str]] = Field(
        default=None,
        description="A list of the text of currently visible options. Should only be present if state is 'open'."
    )
    search_bar_id: Optional[int] = Field( # type: ignore # New optional field
        default=None,
        ge=0,
        description="The label ID of the search bar within the combobox, if present."
    )
    close_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of the button used to close the calendar (e.g., 'OK', 'Done', 'X'), if present."
    )
    cancel_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of the button used to cancel the selection and close the calendar, if present."
    )

    @model_validator(mode='after')
    def check_visible_options_state(self) -> 'MultiSelectCombobox':
        if self.state == "closed" and self.visibleOptions is not None:
            raise ValueError("visibleOptions should be null or omitted when state is 'closed'")
        if self.state == "open" and self.visibleOptions is None:
            print("Warning: visibleOptions is None but state is 'open'. LLM might have missed options.")
            raise ValueError("visibleOptions must be provided when state is 'open'")
        return self

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
    switch: list[Switch] = Field(default_factory=list)

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


class Calendar(BaseModel):
    """Base calendar class with common fields and discriminator logic."""
    state: str = Field(..., description="Indicates if the calendar is open or closed")
    selected_date: Optional[date] = Field(
        default=None,
        description="The selected date (as a date object or string depending on calendar type)."
    )
    selected_date_id: Optional[int] = Field( 
        default=None,
        ge=0,
        description="The label ID of the UI element displaying the full selected date, if applicable."
    )
    date_entry_field_id: Optional[int] = Field( 
        default=None,
        ge=0,
        description=(
            "The label ID of the UI element which should be interacted with to open "
            "the calendar or accomplish date-related tasks. May be the same as selected_date_id "
            "in cases where clicking the displayed date opens the calendar."
        ),
    )
    model_config = {
        "json_encoders": {
            date: lambda d: d.strftime("%d.%m.%Y") if d else None
        }
    }

    # Static method to validate from dict and determine the type
    @staticmethod
    def model_validate(data: dict) -> Union['OpenCalendar', 'ClosedCalendar', 'SpinnerCalendar']:
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
            
        if "state" not in data:
            raise ValueError("Data must contain 'state' field")
            
        if data["state"] == "open":
            return OpenCalendar.model_validate(data)
        elif data["state"] == "closed":
            return ClosedCalendar.model_validate(data)
        elif data["state"] == "spinner":
            return SpinnerCalendar.model_validate(data)
        else:
            raise ValueError(f"Invalid state value: {data['state']}")


class ClosedCalendar(BaseModel):
    """Represents a closed calendar element, potentially showing a selected date."""
    state: Literal["closed"] = Field(
        ..., description="Indicates the calendar is in a closed state."
    )
    selected_date: Optional[str] = Field(
        default=None,
        description=(
            "The date text displayed when closed. Can be any format, "
            "including abbreviations like '23 apr' or custom formats."
        ),
    )
    selected_date_id: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "The label ID of the UI element displaying the full selected date "
            "when closed, if applicable."
        ),
    )
    date_entry_field_id: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "The label ID of the UI element which should be interacted with to open "
            "the calendar or accomplish date-related tasks. May be the same as selected_date_id "
            "in cases where clicking the displayed date opens the calendar."
        ),
    )


class OpenCalendar(BaseModel):
    """Represents an open/expanded calendar element with detailed selectable parts."""
    state: Literal["open"] = Field(
        ...,
        description="Indicates the calendar is in an open/expanded state."
    )
    selected_month: str = Field(
        ...,
        description="The full name of the currently displayed/selected month (e.g., 'April', 'May')."
    )
    selected_month_id: int = Field( 
        ...,
        ge=0,
        description="The label ID of the UI element showing the selected month name."
    )
    selected_year: int = Field( 
        ...,
        ge=1000,
        le=3000,    
        description="The currently displayed/selected year (e.g., 2025)."
    )
    selected_year_id: int = Field( 
        ...,
        ge=0,
        description="The label ID of the UI element showing the selected year."
    )
    selected_day: int = Field( 
        ...,
        ge=1,
        le=31,  
        description="The currently selected day number within the month."
    )
    selected_day_id: int = Field( 
        ...,
        ge=0,
        description="The label ID of the UI element representing the selected day."
    )
    decrease_button_id: Optional[int] = Field( 
        default=None,
        ge=0,
        description="The label ID of the button used to navigate to the previous month/year, if present."
    )
    increase_button_id: Optional[int] = Field( 
        default=None,
        ge=0,
        description="The label ID of the button used to navigate to the next month/year, if present."
    )
    close_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of the button used to close the calendar (e.g., 'OK', 'Done', 'X'), if present."
    )
    cancel_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of the button used to cancel the selection and close the calendar, if present."
    )
    selected_date: Optional[date] = Field( 
        default=None,
        description="The full selected date (e.g., dd.mm.yyyy or mm.dd.yyyy) if represented as a single element, typically relevant when closed."
    )
    selected_date_id: Optional[int] = Field( 
        default=None,
        ge=0,
        description="The label ID of the UI element displaying the full selected date, if applicable as a single element."
    )
    date_entry_field_id: Optional[int] = Field( 
        default=None,
        ge=0,
        description=(
            "The label ID of the UI element which should be interacted with to open "
            "the calendar or accomplish date-related tasks. May be the same as selected_date_id "
            "in cases where clicking the displayed date opens the calendar."
        ),
    )

    @field_validator("selected_date", mode='before')
    @classmethod
    def _parse_selected_date(cls, v: Any) -> Optional[date]:
        # allow None or already-parsed date
        if v is None or isinstance(v, date):
            return v

        # Ensure input is a string before trying strptime
        if not isinstance(v, str):
             raise ValueError("selected_date must be a string for parsing")

        # try dd.mm.yyyy then mm.dd.yyyy
        for fmt in ("%d.%m.%Y", "%m.%d.%Y"):
            try:
                return datetime.strptime(v, fmt).date()
            except ValueError:
                continue
        # If loop finishes without returning, raise error
        raise ValueError(f"selected_date string '{v}' must be in dd.mm.yyyy or mm.dd.yyyy format")

    model_config = {
        "json_encoders": {
            date: lambda d: d.strftime("%d.%m.%Y") if d else None
        }
    }

class SpinnerItem(BaseModel):
    """Represents a single spinner component in a spinner date picker."""
    bounding_box_id: int = Field(
        ...,
        ge=0,
        description="The label ID of the bounding box for this specific spinner."
    )
    value: str = Field(
        ...,
        description="The currently selected value in this spinner (e.g., '07', 'HAZ', '2005')."
    )
    format: str = Field(
        ...,
        description="The format this spinner represents (e.g., '%d', '%b', '%Y') using Python's datetime format codes."
    )

class SpinnerCalendar(BaseModel):
    """Represents a spinner-style date picker with multiple rotating selectors."""
    state: Literal["spinner"] = Field(
        ...,
        description="Indicates the calendar is a spinner-type date picker."
    )
    spinners: List[SpinnerItem] = Field(
        ...,
        description="List of spinner components, each with its own bounding box, value, and format."
    )
    decrease_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of any button used to decrease values in spinners, if present."
    )
    increase_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of any button used to increase values in spinners, if present."
    )
    close_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of the button used to close and confirm the selection (e.g., 'OK', 'Save'), if present."
    )
    cancel_button_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="The label ID of the button used to cancel the selection (e.g., 'Cancel', 'İptal'), if present."
    )


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

class RouterDecision(BaseModel):
    """
    Represents the router's decision identifying the OS and the single
    most relevant UI element type for the user's request.
    """
    device: Literal["ios", "android"] = Field(
        ...,
        description="The detected operating system of the device shown in the image."
    )
    target_element_type: Optional[ELEMENT_TYPES] = Field(
        ..., # The key must be present, but value can be null
        description="The single UI element type most relevant to the user's request "
                    "and visible in the screenshot. Should be null if no single "
                    "relevant element type is identified or applicable."
    )

    # model_config = {
    #     "json_schema_extra": {
    #         "examples": [
    #             {"device": "android", "target_element_type": "Button"},
    #             {"device": "ios", "target_element_type": "Textbox"},
    #             {"device": "android", "target_element_type": None}, # Representing null
    #         ]
    #     }
    # }

class GraphState(TypedDict):
    image_base64: str # Base64 encoded representation of input UI screenshot.
    image_mime_type: str # The MIME type of the input image
    use_90b: bool # Config flag
    human_request: str # String that will be written.
    device: Optional[Literal["ios", "android"]]
    analysis_required: List[str]
    agent_analysis: Optional[str]
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


