from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEmbeddings
from core import get_model, settings
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


embeddings_ = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vectorstore = Chroma(embeddings,collection_name="rag-chroma",persist_directory="./chroma_langchain_db")
vectorstore = Chroma(collection_name='rag-chroma', persist_directory="./chroma_langchain_db", embedding_function = embeddings_)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str | None = None  # Make query optional with default None
    retrieved_docs: list[str] | None = None  

async def retrieve(state: AgentState, config: RunnableConfig) -> AgentState:
    """Retrieve relevant content based on user query."""
    try:
        query = state["messages"][-1].content
        state["query"] = query
    
        # Safely get user from custom_data
        data = config["configurable"].get("custom_data", [])
        user = data[0].get("user") if data else None
       
        if not user:
            # Return empty state if no user provided
            state["retrieved_docs"] = []
            return state
            
        # Perform search with user filter
        search_kwargs = {"filter": {"user": user}} if user else {}
        results = await vectorstore.asimilarity_search(
            query, 
            k=5,
            filter = {"user":user}
        )
        state["retrieved_docs"] = results
      
        
        return state
        
    except Exception as e:
        # Return state with empty results on error
        state["retrieved_docs"] = []
        return state
    

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate answer"""
    try:
        docs = []
        source = []
        messages = state["messages"]
        question = state.get("query", messages[-1].content)  # Fallback to last message
        history = "\n".join([m.content for m in messages[:-1]])
        
        # Safely handle retrieved docs
        if state.get("retrieved_docs"):
            for doc in state["retrieved_docs"]:
                docs.append(doc.page_content)
                source.append(doc.dict())
        
        # Rest of your existing code...
                system_prompt = """
                You are an AI assistant specialized in answering questions based on provided documents context and conversation history. 
                Your goal is to provide accurate, concise, and contextually relevant answers. 
                When answering, consider the following guidelines:
                *** INSTRUCTIONS ***
                Think step-by-step before generating content. Follow below steps:
                1. Identify the company, what financial details the user is asking about.
                2. Check if that company details are present in the context or relevant documents.
                3. DO NOT ANSWER the question, if the relevant documents are not found for that company. Quote the numerical values if present in relevant documents.
                4. Do not answer the question based on assumptions using model internal memory.
                5. Always back your responses with citations or references from the provided documents.
                6. Maintain the context of the conversation. Do not provide irrelevant information.
                7. Be polite and professional in your responses. Its okay if you do not know he answer but dont give incorrect response.

                """
        user_prompt = """
                Conversation history:
                {history}

                User question:
                {question}

                Relevant documents:
                {context}

                Please provide a answer based on the above information.
                """
        prompt = ChatPromptTemplate([("system", system_prompt), ("user", user_prompt)])
        
        llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        rag_chain = prompt | llm
        
        response = await rag_chain.ainvoke(
            {
                "context": "\n\n".join(docs) if docs else "No relevant documents found.",
                "question": question,
                "history": history
            }, 
            config
        )
        
        
        response.custom_data = source
        return {"messages": [response]}
        
    except Exception as e:
        return {"messages": [AIMessage(content="I apologize, but I encountered an error processing your request.")]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("Generator", acall_model)
agent.add_node("Retriever", retrieve)
agent.add_edge(START, "Retriever")
agent.add_edge("Retriever", "Generator")
agent.add_edge("Generator", END)

chatbot = agent.compile(checkpointer=MemorySaver())