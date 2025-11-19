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

from my_agent.tools import forcasting_demand

USER_ID = "user_003"
ENGINE_NAME = "roi_engine"
PROJECT_NAME = "project_two"
SESSION_ID = "session_002"
BASE_URL = "http://localhost:8000"


APP_NAME = "test_session"
USER_ID = "user_two"
session_id = "123"

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
        "user_id": "user_gamma",
        "project_name": "project_gamma",
        "engine_name": "engine_gamma",
        "query": last_user_message
    }


    search_results = requests.post(f"{BASE_URL}/search", json=search_query)
    # if search_results.status_code == 200:
    #     memories = search_results.json().get("message", [])
    #     print(f"[Callback] Retrieved {len(memories)} memories from search API.")
    # else:
    #     print(f"[Callback] Failed to retrieve memories. Status code: {search_results.status_code}")
    #     memories = []

    # --- Modification Example ---
    # Add a prefix to the system instruction
    original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])
    prefix = f"Memory Search Result: {search_results.json()}\n\n"

    
    # Ensure system_instruction is Content and parts list exists
    if not isinstance(original_instruction, types.Content):
         # Handle case where it might be a string (though config expects Content)
         original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])
    if not original_instruction.parts:
        original_instruction.parts.append(types.Part(text="")) # Add an empty part if none exist

    # Modify the text of the first part
    modified_text = prefix + (original_instruction.parts[0].text or "")
    original_instruction.parts[0].text = modified_text
    llm_request.config.system_instruction = original_instruction
    print(f"[Callback] Modified system instruction to: '{modified_text}'")

    return None # Proceed with the (modified) request

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
    if response.status_code == 200:
        print("[Callback] Successfully added memory via API.")
    else:
        print(f"[Callback] Failed to add memory. Status code: {response.status_code}, Response: {response.text}")

    return None


from google.adk.agents import Agent

# Define your agent
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
        "Ask the user for the forecasting period if needed and then use the `forcasting_demand` tool to respond."
    ),
    after_model_callback=after_model_add_memory,
    before_model_callback=before_model_search_memory,
    tools=[forcasting_demand]
)



root_agent = agent

