#!/usr/bin/env python3
"""
PostgreSQL Memory Demo for Google ADK Agents
Enhanced version:
- Only provides dates/times for day/time questions.
- Answers in third person when a name is mentioned (e.g., 'Roshil').
- Avoids 'as we discussed earlier' references.
"""

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime
import psycopg2
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

# --------------------- Load Environment ---------------------
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

# --------------------- Config ---------------------
USER_ID = "Postgres_Session_Memory_User"
DB_URL = "postgresql://postgres:newpassword@localhost:5432/postgres"
MEMORY_TABLE = "chat_history"

# --------------------- Database ---------------------
def init_memory_table():
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="newpassword", host="localhost")
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {MEMORY_TABLE} (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            session_id TEXT,
            role TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_message(user_id, session_id, role, message, speaker_name=None):
    """Save message with optional speaker name."""
    speaker = speaker_name if speaker_name else ("User" if role == "user" else "Agent")
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="newpassword", host="localhost")
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO {MEMORY_TABLE} (user_id, session_id, role, message)
        VALUES (%s, %s, %s, %s)
    """, (user_id, session_id, speaker, message))
    conn.commit()
    cur.close()
    conn.close()

# --------------------- Timestamp Helper ---------------------
def relative_day_with_date(created_at):
    """Return a string like 'today [2025-10-29]' or 'yesterday [2025-10-28]'."""
    now = datetime.now()
    delta_days = (now.date() - created_at.date()).days
    date_str = created_at.strftime("%Y-%m-%d")
    if delta_days == 0:
        return f"today [{date_str}]"
    elif delta_days == 1:
        return f"yesterday [{date_str}]"
    elif delta_days < 7:
        return f"{delta_days} days ago [{date_str}]"
    else:
        return date_str

def load_user_memory(user_id):
    """Load all previous messages for a user."""
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="newpassword", host="localhost")
    cur = conn.cursor()
    cur.execute(f"""
        SELECT role, message, created_at FROM {MEMORY_TABLE}
        WHERE user_id = %s
        ORDER BY created_at
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return "\n".join([
        f"[{relative_day_with_date(created_at)}] {role}: {msg}"
        for role, msg, created_at in rows
    ])

# --------------------- Agent Setup ---------------------
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
            "Use the memory of past conversations to answer naturally. "
            "Do NOT say 'as we discussed earlier' or 'as you mentioned earlier'. "
            "Only include dates or times (e.g., [2025-10-29], 'yesterday') when the user explicitly asks "
            "about when something happened (e.g., 'when', 'day', 'date', 'time'). "
            "If a question includes a person's name (e.g., 'Roshil'), respond about that person in third person — "
            "do NOT refer to them as 'you'. "
            "When the user refers to themselves without using a name, you may answer in second person ('you'). "
            "Respond clearly, politely, and naturally. Do NOT output code or SQL."
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

# --------------------- Display ---------------------
def display_message(role: str, text: str):
    icons = {"User": "💬", "Agent": "🤖"}
    prefix = icons.get(role, "")
    name = "PostgresKnowledgeAgent" if role == "Agent" else "User"
    print(f"{prefix} {name}: {text or '(No response)'}")

# --------------------- Agent Reply ---------------------
def generate_agent_reply(runner, session, user_input):
    display_message("User", user_input)

    # Check if user asked about date/time
    day_keywords = ["when", "day", "date", "time", "today", "yesterday"]
    include_dates = any(word in user_input.lower() for word in day_keywords)

    # Load memory
    memory_text = load_user_memory(USER_ID)
    if not include_dates:
        memory_text = re.sub(r"\[\d{4}-\d{2}-\d{2}\]", "", memory_text)

# --------------------- Detect if the user mentioned a person's name ---------------------
    name_pattern = r"\b(Roshil|Buddy|Aayush|Muskan)\b" 

    if re.search(name_pattern, user_input, flags=re.IGNORECASE):
        prompt_text = (
            "[Note: Answer about the person(s) mentioned, do not assume 'you']\n"
            f"Memory:\n{memory_text}\n"
            f"User: {user_input}\n"
            "Agent:"
        )
    else:
        prompt_text = (
            f"Memory:\n{memory_text}\n"
            f"User: {user_input}\n"
            "Agent:"
        )

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
        save_message(USER_ID, session.id, "user", user_input)
        save_message(USER_ID, session.id, "agent", response_text)
        return response_text
    except Exception as e:
        print(f" Error while getting agent response: {e}")
        return None

# --------------------- Chat Loop ---------------------
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
    print(f" APP NAME    : {'PostgresMemoryDemoApp'}")
    print(f" USER ID     : {session.user_id}")
    print(f" SESSION ID  : {session.id}")
    print("--------------------------------------------------")
    print("Welcome to the PostgreSQL Chat CLI!")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Goodbye!")
            break
        generate_agent_reply(runner, session, user_input)

# --------------------- Main ---------------------
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
