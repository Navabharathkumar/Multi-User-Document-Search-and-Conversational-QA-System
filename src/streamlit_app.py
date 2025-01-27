import asyncio
import os
from collections.abc import AsyncGenerator

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:
# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.
# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.

APP_TITLE = "Multi-User Document Search and Conversational Q&A System"
APP_ICON = "🤖"

async def main() -> None:
    """Main function to set up the Streamlit app and high-level structure."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )
            # User access control
    user_access = {
            "user1@example.com": ["Amazon.pdf", "IBM.pdf", "IBM2.pdf"],
            "user2@example.com": ["JPMC.pdf", "Wells.pdf"],
            "user3@example.com": ["Alphabet.pdf", "JPMC.pdf", "IBM.pdf","IBM2.pdf"],
        }

    # Hide the Streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        agent_url = os.getenv("AGENT_URL", "http://localhost")
        st.session_state.agent_client = AgentClient(agent_url)
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            history: ChatHistory = agent_client.get_history(thread_id=thread_id)
            messages = history.messages
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id
    models = {
            
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Groq Llama 3.1 8B": "groq-llama-3.1-8b",
            "Groq Llama 3.3 70B": "groq-llama-3.3-70b",
            "Mixtral 8x7B Instruct V0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }
    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        with st.popover(":material/settings: Settings", use_container_width=True):
            m = st.radio("LLM to use", options=models.keys())
            model = models[m]
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=["chatbot"],
            )

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        st.markdown(
            f"Thread ID: **{st.session_state.thread_id}**",
            help=f"Set URL query parameter ?thread_id={st.session_state.thread_id} to continue this conversation",
        )


        username = st.text_input("Enter your username to access the chat:")
        if username in user_access:
            st.success(f"Welcome {username}! You have access to the following documents: {', '.join(user_access[username])}")
        else:
            st.error("Invalid username. Please try again.")
            st.stop()

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = f"Hello {username}! I'm an AI-powered research assistant with web search and a calculator. I may take a few seconds to boot up when you send your first message. Ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        response = await agent_client.ainvoke(
            message=user_input,
            model=model,
            thread_id=st.session_state.thread_id,
            custom_data=[{"user": username}]
        )
        messages.append(response)
        with st.chat_message("ai"):
            st.write(response.content)
            # Display metadata below the generated content
            if response.custom_data:
                cols = st.columns(len(response.custom_data))
                for idx, meta in enumerate(response.custom_data):
                    source = meta["metadata"]
                    content = meta["page_content"]
                    # Shorten long content
                    short_content = content if len(content) <= 100 else content[:100] + "..."
                    with cols[idx]:
                        with st.expander(f"Source: {source}", expanded=False):
                            st.markdown(f"**Content:** {short_content}")

async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages or new ones.

    Args:
        messages_agen: An async iterator over messages to draw
        is_new: Whether the messages are new or not
    """
    # Keep track of the last message container  
    last_message_type = None
    st.session_state.last_message = None

    # Iterate over the messages and draw them
    async for msg in messages_agen:
        match msg.type:
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)
            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    st.write(msg.content)

async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        normalized_score = (feedback + 1) / 5.0
        agent_client: AgentClient = st.session_state.agent_client
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs={"comment": "In-line human feedback"},
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")

if __name__ == "__main__":
    asyncio.run(main())