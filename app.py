import os
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ----------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# ----------------------------------------------------
# LLM
# ----------------------------------------------------

llm = ChatGroq(model="llama-3.1-8b-instant")


# ----------------------------------------------------
# TOOLS
# ----------------------------------------------------

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=300)
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
)

tavily = TavilySearch(max_results=3)

tools = [arxiv, wiki, tavily]

llm_with_tools = llm.bind_tools(tools)


# ----------------------------------------------------
# STATE
# ----------------------------------------------------

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ----------------------------------------------------
# AGENTS
# ----------------------------------------------------

def planner_agent(state: State):

    system_prompt = """
    You are a research planner.

    Break the user query into smaller research questions.
    Decide what information is needed.
    """

    response = llm.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"][-4:]
    )

    return {"messages": [response]}


def research_agent(state: State):

    system_prompt = """
    You are a research agent.

    Use tools when needed:
    - Arxiv for academic papers
    - Wikipedia for background
    - Tavily for latest web info
    """

    response = llm_with_tools.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"][-4:]
    )

    return {"messages": [response]}


def writer_agent(state: State):

    system_prompt = """
    You are a research writer.

    Combine gathered information into a concise structured summary.

    Output format:

    Topic
    Key Concepts
    Recent Research
    Applications
    References
    """

    response = llm.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )

    return {"messages": [response]}


# ----------------------------------------------------
# BUILD GRAPH
# ----------------------------------------------------

builder = StateGraph(State)

builder.add_node("planner", planner_agent)
builder.add_node("research", research_agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("writer", writer_agent)

# Start
builder.add_edge(START, "planner")

# Planner -> Research
builder.add_edge("planner", "research")


# Research decides:
# If tool needed -> tools
# Else -> writer
builder.add_conditional_edges(
    "research",
    tools_condition,
    {
        "tools": "tools",
        END: "writer",
    },
)

# After tools, go back to research
builder.add_edge("tools", "research")

# Writer ends
builder.add_edge("writer", END)

graph = builder.compile()


# ----------------------------------------------------
# FUNCTION TO RUN AGENT
# ----------------------------------------------------

def run_agent(query):
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]}
        )
        return result["messages"][-1].content

    except Exception as e:
        return f"Error occurred: {str(e)}"


# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------

st.title("AI Research Assistant")
st.write("Ask any research question. The agent will gather information from Arxiv, Wikipedia, and the web.")

user_query = st.text_input("Enter your question:")

if st.button("Search"):

    if user_query:

        with st.spinner("Researching..."):

            answer = run_agent(user_query)

        st.subheader("Result")
        st.write(answer)