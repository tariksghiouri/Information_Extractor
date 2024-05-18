import streamlit as st
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import urllib.parse
import json
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

# Define the Pydantic models
class NearbyPlace(BaseModel):
    name: str
    google_maps_link: str = None

class MonumentInfo(BaseModel):
    name: str
    location: str
    built: str
    architect: str
    Architectural_styles:str
    significance: str
    visitor_info: str
    nearby_places: List[NearbyPlace]

parser = JsonOutputParser(pydantic_object=MonumentInfo)

prompt = PromptTemplate(
    template="""system
    You are an information extractor focused on historical monuments and Buildings in morocco.
    If provided with the name of a specified historical monument,
    give back information about the monument.
    Give details about the place: 
    Provide these details as a JSON with these keys: 'name', 'location', 'built', 'architect', 'Architectural styles, 'significance', 'visitor_info', 'nearby places'.
    for the visitor_info give if there are rules to follow for visiting the monument things to do or to do not
    and no preamble or explanation.
    {format_instructions}

    user
    Here is the historical monument: {location}
    assistant""",
    input_variables=["location"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = HuggingFaceTextGenInference(
    inference_server_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
    server_kwargs={
        "headers": {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
    },
)

chain = prompt | llm | parser

# Function to get monument information
def get_monument_info(monument_name: str) -> MonumentInfo:
    response = chain.invoke({"location": monument_name})
    monument_info = MonumentInfo.parse_obj(response)

    # Add Google Maps links for nearby places with correct URL encoding
    starting_point = monument_info.name
    starting_point_encoded = urllib.parse.quote_plus(starting_point)
    for place in monument_info.nearby_places:
        destination_encoded = urllib.parse.quote_plus(place.name)
        place.google_maps_link = f"https://www.google.com/maps/dir/{starting_point_encoded}/{destination_encoded}"

    return monument_info

# Function to format the monument information as Markdown
def format_monument_info(monument_info: MonumentInfo) -> str:
    formatted_info = f"""
### {monument_info.name}

**Location:** {monument_info.location}

**Built:** {monument_info.built}

**Architect:** {monument_info.architect}

**Architectural style :** {monument_info.Architectural_styles}

**Significance:** {monument_info.significance}

**Visitor Info:** {monument_info.visitor_info}

**Nearby Places:**
"""
    for place in monument_info.nearby_places:
        formatted_info += f"- [{place.name}]({place.google_maps_link})\n"

    return formatted_info

# Streamlit app
st.title("Historical Monument Information Extractor")

monument_name = st.text_input("Enter the name of the monument:")
if st.button("Get Information"):
    if monument_name:
        info = get_monument_info(monument_name)
        formatted_info = format_monument_info(info)
        st.markdown(formatted_info)
    else:
        st.error("Please enter a monument name.")
