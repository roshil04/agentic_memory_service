from google.adk.sessions.session import Session
from google.adk.sessions import InMemorySessionService
from google.adk.events.event import Event
from google.genai import types
from google.genai.types import Part
import uuid

APP_NAME = "test_session"
USER_ID = "user_two"

async def create_session(session_id):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=session_id,
    )
    return session

async def create_episode(text):
    parts = []
    session_id = uuid.uuid4().hex
    session = await create_session(session_id)
    event = Event(invocationId="e-cc44898b-e88b-4fb1-b48c-b59d2e6d1abb", author="user")
    part = Part(text=text)
    role = "user"
    parts.append(part)
    content = types.Content(parts = parts, role=role)
    event.content = content
    session.events.append(event)
    
    return session

# async def create_episode(texts):
#     """
#     Creates an episode session using a list of texts.

#     Args:
#         texts (list[str]): A list of strings, where each string is a part of the episode content.

#     Returns:
#         Session: The created session object with the content added.
#     """
#     parts = []
#     session_id = uuid.uuid4().hex
#     session = await create_session(session_id) 
#     event = Event(invocationId="e-cc44898b-e88b-4fb1-b48c-b59d2e6d1abb", author="user")
    
#     # Iterate over the list of texts and create a Part for each one
#     for text in texts:
#         part = Part(text=text)
#         parts.append(part)
        
#     role = "user"
#     content = types.Content(parts = parts, role=role)
#     event.content = content
#     session.events.append(event)
#     return session