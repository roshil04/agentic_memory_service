import requests
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
# from vanilla_rag_agent.tools import search
from typing import Optional
from google.genai import types 

from google.adk.sessions.session import Session
from google.adk.sessions import InMemorySessionService
from google.adk.events.event import Event
from google.genai import types
from google.genai.types import Part

from my_agent.tools import forecasting_demand
from datetime import datetime, timezone, date
# from custom_memory_service.services.graphiti_logic import SearchResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import time

USER_ID = "user_003"
ENGINE_NAME = "roi_engine"
PROJECT_NAME = "project_two"
SESSION_ID = "session_002"
BASE_URL = "http://localhost:8005"


APP_NAME = "test_session"
USER_ID = "user_two"
session_id = "123"

class SearchResultEdge(BaseModel):
    fact: str
    uuid: str
    created_at: datetime | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None

class SearchResultEpisode(BaseModel):
    content: str
    created_at: datetime | None = None

class SearchResponse(BaseModel):
    edges: List[SearchResultEdge]
    episodes: List[SearchResultEpisode]


async def create_session():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id="123_session",
    )
    return session

async def create_episode(text):
    parts = []
    session = await create_session()
    event = Event(invocationId="e-cc44898b-e88b-4fb1-b48c-b59d2e6d1abb", author="user")
    part = Part(text=text)
    role = "user"
    parts.append(part)
    content = types.Content(parts = parts, role=role)
    event.content = content
    session.events.append(event)
    return session

def before_model_search_memory(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""


    agent_name = callback_context.agent_name
    print(f"Search Memory callback triggered.")

    # Inspect the last user message in the request contents
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    # Search the memory for relevant context
    search_query = {
        "user_id": "user_delta",
        "project_name": "project_beta",
        "engine_name": "engine_beta",
        "query": last_user_message
    }

    
    data = None
    edges = []
    episodes = []
    prefix = ""

    try:
        # Send the search request
        search_results = requests.post(f"{BASE_URL}/search", json=search_query)
        search_results.raise_for_status()  # Raises error for HTTP issues

        # Parse JSON response
        data = search_results.json()

        # Parse into Pydantic model
        search_response = SearchResponse(**data)
        edges = search_response.edges
        episodes = search_response.episodes

        # Create a prefix with all data values
        prefix = "Memory Search Result:\n"
        for key, value in data.items():
            prefix += f"{key}: {value}\n"

        # # Save edges to a file (optional)
        # filename = f"memory_result_{int(time.time())}.txt"
        # with open(filename, "w") as f:
        #     f.write(str(edges))

    except (requests.RequestException, ValueError, TypeError, KeyError) as e:
        print(f"An error occurred while fetching or processing search results: {e}")
        data = None
        edges = []
        episodes = []
        prefix = ""
        
    # --- FIX: Initialize original_instruction ---
    original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])

    if not isinstance(original_instruction, types.Content):
        original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])

    if not original_instruction.parts:
        original_instruction.parts.append(types.Part(text=""))

    # Modify the first part's text
    modified_text = prefix + (original_instruction.parts[0].text or "")
    original_instruction.parts[0].text = modified_text
    llm_request.config.system_instruction = original_instruction

    print(f"[Callback] Modified system instruction to: '{modified_text}'")

    return None

async def after_model_add_memory(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """
    After-model callback that stores the user message and LLM response
    in memory using the add_memory API.
    """
    # Extract AI response text

    agent_name = callback_context.agent_name
    # Extract User Message
    user_message = callback_context.user_content.parts[0].text
    print("User message:", user_message)
    print("Add memory callback triggered.")

    # --- Extract LLM Response ---
    original_text = ""
    print(llm_response.content)
    if llm_response.content and llm_response.content.parts:
        # Assuming simple text response for this example
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            print(f"[Callback] Inspected original response text: '{original_text[:100]}...'") # Log snippet
        else:
             print("[Callback] Inspected response: No text content found.")
             return None
    elif llm_response.error_message:
        print(f"[Callback] Inspected response: Contains error '{llm_response.error_message}'. No modification.")
        return None
    else:
        print("[Callback] Inspected response: Empty LlmResponse.")
        return None
    

    episode = await create_episode(f"User:{user_message}\n AI: {original_text}")

    # Build Chat Conversation Payload
    episode_data = {
        "user_id": "user_gamma",
        "project_name": "project_gamma",
        "engine_name": "engine_gamma",
        "session": episode.model_dump()
    }

    response = requests.post(f"{BASE_URL}/add_episode", json=episode_data)


    # Save edges to a file (optional)
    filename = f"memory_add_{int(time.time())}.txt"
    with open(filename, "w") as f:
        f.write(str(response))

    if response.status_code == 200:
        print("[Callback] Successfully added memory via API.")
    else:
        print(f"[Callback] Failed to add memory. Status code: {response.status_code}, Response: {response.text}")

    return None


from google.adk.agents import Agent

agent = Agent(
    name="VanillaRAGMemAgent",
    model="gemini-2.0-flash",
    description=(
        "An intelligent retrieval-augmented agent designed to store and recall chat memories. "
        "It can add new memory entries and search through previous conversations to provide concise, "
        "context-aware answers."
    ),
    instruction=(
        f"The user ID is '{USER_ID}' and the application name is '{ENGINE_NAME}'. "

        "You have access to two key sources of information: "
        "1) Memory search results (prepended in the system prompt), and "
        "2) The `forecasting_demand` tool for forecasting tasks. "

        "Use memory search results to answer factual questions about past data, "
        "such as price per unit, revenue, quantity, brand, SKU, region, date, or category. "
        "Answer directly from memory search results whenever possible. "

        "If the user asks about future predictions, forecasting, or estimates, "
        "do NOT call the `forecasting_demand` tool automatically. "
        "Instead, ask the user first if they want a forecast to be performed. "
        "Only call the tool if the user confirms. "
        "When calling the tool, extract the forecast period and any SKU, Region, Brand, or Category mentioned in the question and pass them as filters. "

        "If memory search results contain multiple matching rows, choose the most relevant one."
    ),
    before_model_callback=before_model_search_memory,
    tools=[forecasting_demand]
)

root_agent = agent

