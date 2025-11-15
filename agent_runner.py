#!/usr/bin/env python3
"""
PostgreSQL Memory Demo for Google ADK Agents with embeddings and cross-session memory by USER_ID
"""

import asyncio
import logging
import os
import uuid
import psycopg2
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import Client, types

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

USER_ID = "Postgres_Session_User"
DB_URL = "postgresql://postgres:newpassword@localhost:5432/postgres"
MEMORY_TABLE = "chat_history"

# Initialize Google GenAI Client
client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# Database & Memory Utilities
# -----------------------------

def init_memory_table():
    """Ensure the chat_history table exists with embedding column."""
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="newpassword", host="localhost")
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {MEMORY_TABLE} (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            session_id TEXT,
            role TEXT,
            message TEXT,
            embedding FLOAT8[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def get_embedding(text: str) -> list[float]:
    """Generate embedding vector for a given text as a plain Python list."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    # Convert ContentEmbedding object to a plain Python list
    return list(response.embeddings[0].values)

def save_message(user_id, session_id, role, message):
    """Save a message and its embedding to the memory table."""
    embedding = get_embedding(message)
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="newpassword", host="localhost")
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO {MEMORY_TABLE} (user_id, session_id, role, message, embedding)
        VALUES (%s, %s, %s, %s, %s)
    """, (user_id, session_id, role, message, embedding))
    conn.commit()
    cur.close()
    conn.close()

def load_user_memory(user_id):
    """Load all previous messages for a USER_ID."""
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="newpassword", host="localhost")
    cur = conn.cursor()
    cur.execute(f"""
        SELECT role, message FROM {MEMORY_TABLE}
        WHERE user_id = %s
        ORDER BY created_at
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return "\n".join([f"{role}: {msg}" for role, msg in rows])

# -----------------------------
# Agent Setup
# -----------------------------

def setup_agent_environment():
    print("Initializing PostgreSQL-backed session service...")
    session_service = DatabaseSessionService(db_url=DB_URL)
    artifact_service = InMemoryArtifactService()
    agent = Agent(
        name="PostgresKnowledgeAgent",
        model="gemini-2.0-flash",
        description="An AI assistant that remembers conversations across sessions.",
        instruction=(
            "You are a friendly AI assistant. "
            "Use the memory of past conversations provided to answer naturally. "
            "Do NOT output code or SQL."
        ),
    )
    runner = Runner(
        app_name="PostgresMemoryDemoApp",
        agent=agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )
    print(" PostgreSQL session system initialized successfully!\n")
    return runner, session_service

def display_message(role: str, text: str):
    icons = {"User": "💬", "Agent": "🤖"}
    prefix = icons.get(role, "")
    name = "PostgresKnowledgeAgent" if role == "Agent" else "User"
    print(f"{prefix} {name}: {text or '(No response)'}")

# -----------------------------
# Chat Loop
# -----------------------------

def generate_agent_reply(runner, session, user_input, memory_text=""):
    display_message("User", user_input)
    prompt_text = f"""Memory from previous sessions:
{memory_text}

User: {user_input}
Agent:"""

    user_msg = types.Content(role="user", parts=[types.Part(text=prompt_text)])
    response_text = None

    try:
        for event in runner.run(
            user_id=session.user_id,
            session_id=session.id,
            new_message=user_msg,
        ):
            if event.author == runner.agent.name and event.is_final_response():
                if event.content and event.content.parts:
                    text_parts = [p.text for p in event.content.parts if p.text]
                    response_text = "".join(text_parts).strip()
                break
        display_message("Agent", response_text)
        # Save conversation to memory with embeddings
        save_message(USER_ID, session.id, "user", user_input)
        save_message(USER_ID, session.id, "agent", response_text)
        return response_text
    except Exception as e:
        print(f" Error while getting agent response: {e}")
        return None

async def chat_loop(runner, session_service):
    session_id = f"postgres_session_{uuid.uuid4().hex[:8]}"
    session = await session_service.get_session(
        app_name="PostgresMemoryDemoApp",
        user_id=USER_ID,
        session_id=session_id,
    )

    if not session:
        session = await session_service.create_session(
            app_name="PostgresMemoryDemoApp",
            user_id=USER_ID,
            session_id=session_id,
        )

    print("--------------------------------------------------")
    print(f" USER ID     : {session.user_id}")
    print(f" SESSION ID  : {session.id}")
    print("--------------------------------------------------")
    print("Welcome to the PostgreSQL Chat CLI!")
    print("Type 'exit' or 'quit' to end the session.\n")

    memory_text = load_user_memory(USER_ID)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Goodbye!")
            break
        generate_agent_reply(runner, session, user_input, memory_text)

# -----------------------------
# Main Entry Point
# -----------------------------

async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: Missing GOOGLE_API_KEY")
        return
    init_memory_table()
    runner, session_service = setup_agent_environment()
    await chat_loop(runner, session_service)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Chat session interrupted by user.")
