import os
import asyncio
import time
import re

import discord
import requests

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_AGENT_API_URL = os.getenv("DISCORD_AGENT_API_URL")

if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN not found in environment.")

#if not DISCORD_AGENT_API_URL:
    #raise ValueError("DISCORD_AGENT_API_URL not found in environment.")

#Checks if backend is enabled, if not we just say that it isnt enabled but can still send msgs
BACKEND_ENABLED = bool(DISCORD_AGENT_API_URL and DISCORD_AGENT_API_URL.strip())

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

SESSION_TIMEOUT_SECONDS = 5 * 60
WARNING_BEFORE_SECONDS = 60

if WARNING_BEFORE_SECONDS >= SESSION_TIMEOUT_SECONDS:
    raise ValueError("WARNING_BEFORE_SECONDS must be less than SESSION_TIMEOUT_SECONDS.")

# user_id -> {"chat_history": [...], "last_message_ts": float}
user_sessions: dict[int, dict] = {}

# user_id -> asyncio.Task
warning_tasks: dict[int, asyncio.Task] = {}
expire_tasks: dict[int, asyncio.Task] = {}

def strip_markdown_links(text: str) -> str:
    """
    Convert markdown links like [label](https://example.com)
    into plain URLs like https://example.com.
    """
    pattern = r"\[([^\]]+)\]\((https?://[^\s)]+)\)"
    return re.sub(pattern, r"\2", text)

def get_user_session(user_id: int) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "chat_history": [],
            "last_message_ts": 0.0,
        }
    return user_sessions[user_id]


def cancel_user_timers(user_id: int) -> None:
    warning_task = warning_tasks.pop(user_id, None)
    if warning_task:
        warning_task.cancel()

    expire_task = expire_tasks.pop(user_id, None)
    if expire_task:
        expire_task.cancel()


def clear_user_session(user_id: int) -> None:
    cancel_user_timers(user_id)
    user_sessions.pop(user_id, None)


def build_question_with_history(user_input: str, chat_history: list[dict]) -> str:
    """
    Flatten prior Discord chat history into the question string so the
    existing Flask route can remain unchanged.
    """
    if not chat_history:
        return user_input

    lines = [
        "Use the previous conversation for context when answering the current question.",
        "",
        "Previous conversation:"
    ]

    for msg in chat_history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if role == "human":
            lines.append(f"User: {content}")
        elif role == "ai":
            lines.append(f"Assistant: {content}")

    lines.append("")
    lines.append(f"Current question: {user_input}")

    return "\n".join(lines)


def ask_backend(question: str) -> str:
    if not BACKEND_ENABLED:
        return (
            "I’m online, but I’m not connected to the backend yet, so I can’t answer questions right now. Please start the website and try again. https://pinto-beans2.onrender.com/"
        )
    response = requests.post(
        DISCORD_AGENT_API_URL,
        json={"question": question},
        timeout=(5, 60),
    )
    response.raise_for_status()

    data = response.json()
    answer = data.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Backend response missing valid 'answer'.")

    return answer


async def send_inactivity_warning(user_id: int, channel: discord.DMChannel) -> None:
    try:
        await asyncio.sleep(SESSION_TIMEOUT_SECONDS - WARNING_BEFORE_SECONDS)

        session = user_sessions.get(user_id)
        if not session:
            return

        if not session.get("chat_history"):
            return

        last_message_ts = session.get("last_message_ts", 0.0)
        if not last_message_ts:
            return

        elapsed = time.time() - last_message_ts
        remaining = SESSION_TIMEOUT_SECONDS - elapsed

        # Only send the warning if the user is still close to timeout.
        if remaining <= WARNING_BEFORE_SECONDS + 1:
            await channel.send(
                f"Your conversation history will reset in about {WARNING_BEFORE_SECONDS} seconds due to inactivity."
            )

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Warning task error for user {user_id}: {e}")


async def expire_user_session(user_id: int, channel: discord.DMChannel) -> None:
    try:
        await asyncio.sleep(SESSION_TIMEOUT_SECONDS)

        session = user_sessions.get(user_id)
        if not session:
            return

        if not session.get("chat_history"):
            return

        last_message_ts = session.get("last_message_ts", 0.0)
        if not last_message_ts:
            return

        elapsed = time.time() - last_message_ts
        if elapsed >= SESSION_TIMEOUT_SECONDS:
            session["chat_history"] = []
            session["last_message_ts"] = 0.0

            warning_tasks.pop(user_id, None)
            expire_tasks.pop(user_id, None)

            await channel.send(
                "Your previous conversation expired due to inactivity. Send a question to start a new conversation."
            )

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Expire task error for user {user_id}: {e}")


def restart_session_timers(user_id: int, channel: discord.DMChannel) -> None:
    cancel_user_timers(user_id)

    warning_tasks[user_id] = asyncio.create_task(
        send_inactivity_warning(user_id, channel)
    )
    expire_tasks[user_id] = asyncio.create_task(
        expire_user_session(user_id, channel)
    )


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    if BACKEND_ENABLED:
        print("Bot is ready and listening for DMs.")
    else:
        print("Bot is ready, but backend API URL is not configured.")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    if message.guild is not None:
        return

    if not isinstance(message.channel, discord.DMChannel):
        return

    user_id = message.author.id
    user_input = message.content.strip()

    if not user_input:
        return

    if user_input.lower() == "!reset":
        clear_user_session(user_id)
        await message.channel.send("Your conversation history has been reset.")
        return

    session = get_user_session(user_id)
    chat_history = session["chat_history"]
    full_question = build_question_with_history(user_input, chat_history)

    async with message.channel.typing():
        try:
            bot_output = await asyncio.to_thread(ask_backend, full_question)

            chat_history.append({"role": "human", "content": user_input})
            chat_history.append({"role": "ai", "content": bot_output})
            session["last_message_ts"] = time.time()

            restart_session_timers(user_id, message.channel)

            clean_output = strip_markdown_links(bot_output)
            await message.channel.send(clean_output)

        except requests.Timeout:
            print(f"Timeout while calling backend for user {user_id}")
            await message.channel.send(
                "Sorry, the backend took too long to respond. Please try again in a moment. Make sure the website is running, then try again: https://pinto-beans2.onrender.com/"
            )

        except requests.RequestException as e:
            print(f"HTTP error while calling backend for user {user_id}: {e}")
            await message.channel.send(
                "Sorry, the backend took too long to respond. Please try again in a moment. Make sure the website is running, then try again: https://pinto-beans2.onrender.com/"
            )

        except Exception as e:
            print(f"Unexpected error while handling message from user {user_id}: {e}")
            await message.channel.send(
                "Sorry, something went wrong while processing your message."
            )


def main():
    client.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()