import os
import asyncio

import discord
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
import os
from agent.agent import create_agent

# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / "agent" / ".env"

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN not found in .env")

# Discord intents
# message_content is required so the bot can read DM message text.
intents = discord.Intents.default()
intents.message_content = True


# Create the Discord client
client = discord.Client(intents=intents)


# ------------------------------------------------------------------
# In-memory per-user chat history
# ------------------------------------------------------------------
# Key: Discord user ID
# Value: list of LangChain messages [HumanMessage, AIMessage, ...]
# This exists only while the bot is running.
# If the bot restarts, everything is forgotten.
user_histories = {}


def get_user_history(user_id: int):
    """Get or create a user's in-memory chat history."""
    if user_id not in user_histories:
        user_histories[user_id] = []
    return user_histories[user_id]


def run_agent(user_input: str, chat_history: list):
    """
    Synchronous helper function that creates a new agent and invokes it.

    We run this inside asyncio.to_thread(...) so the Discord bot
    does not block while the agent is working.
    """
    agent_executor = create_agent()

    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    return response


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    print("Bot is ready and listening for DMs.")


@client.event
async def on_message(message: discord.Message):
    """
    Handle incoming messages.

    Rules:
    - Ignore bot messages
    - Ignore server/guild messages
    - Only respond in DMs
    - Support !reset command
    """
    # Ignore messages from bots, including itself
    if message.author.bot:
        return

    # Only respond to DMs and ignore server messages
    if message.guild is not None:
        return

    # Check to make sure we only respond in DM channels
    if not isinstance(message.channel, discord.DMChannel):
        return

    user_id = message.author.id
    user_input = message.content.strip()

    if not user_input:
        return

    # Reset this user's history
    if user_input.lower() == "!reset":
        user_histories[user_id] = []
        await message.channel.send("Your conversation history has been reset.")
        return

    # Get this user's private history
    chat_history = get_user_history(user_id)

    # typing indicator so the user sees the bot is working
    async with message.channel.typing():
        try:
            # Run the synchronous LangChain invoke call in a background thread
            response = await asyncio.to_thread(
                run_agent,
                user_input,
                chat_history
            )

            bot_output = response["output"]

            # Save conversation history only in memory
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=bot_output))

            # Send response back to the user
            await message.channel.send(bot_output)

        except Exception as e:
            print(f"Error while handling message from {message.author}: {e}")
            await message.channel.send(
                "Sorry, something went wrong while processing your message."
            )


def main():
    client.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()