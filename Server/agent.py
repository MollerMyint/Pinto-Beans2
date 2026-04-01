# file to create chat agent to help students navigate website

import os
from dotenv import load_dotenv
import json
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv() # load environment variables from .env file

# ============================================================================
# LANGCHAIN TOOLS (TBD: IMPLEMENT THESE)
# ============================================================================
# These functions are decorated with @tool to make them available to the
# LangChain agent. The agent can call these tools to perform specific tasks.


def create_agent() -> AgentExecutor:
    """Create LangChain agent with tools."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key
    )

    # Define the tools
    tools = []

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for Cal Poly Pomona students."""),

        # NEED TO IMPLEMENT TOOLS TO USE PROMPT BELOW
        # ("system", """You are a helpful assistant for Cal Poly Pomona students.
        # Use the search_corpus tool to find facts in the scraped website corpus (markdown under test_corpus).
        # Base answers only on tool results; if searches find nothing relevant, say the answer cannot be found in the corpus.
        # Keep a conversational tone and cite source URLs from the tool output when you use them."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create the agent (uses the tools and prompt)
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create executor (runs the agent)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False, # does not print verbose output
        handle_parsing_errors=True
    )

    return agent_executor

def main():
    """Main function to run the agent."""
    try: # create the agent and check for API key
        agent_executor = create_agent()
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set your OPENAI_API_KEY in a .env file.")
        return
    
    chat_history = [] # store chat history

    while True: # main chat loop
        try:
            user_input = input("\nYou: ").strip() # get user input
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Run the agent (invoke the agent executor)
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            # Display the response (from the agent executor)
            print(f"\nAssistant: {response['output']}")

            # Update chat history (store the user input and the agent response)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response['output']))

        except KeyboardInterrupt: # handle keyboard interrupt
            print("\n\nInterrupted by user.")
            break

        except Exception as e: # handle other exceptions
            print(f"\nError: {e}")
            print("Please try again.")
            continue

if __name__ == "__main__": # run the main function
    main()