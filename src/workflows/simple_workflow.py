from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)


# Graph State
class SimpleState(TypedDict):
    query: str
    response: str
    task: str


# -----------------------------
# Router Node
# -----------------------------
def router_node(state: SimpleState):

    query = state["query"].lower()

    if "summarize" in query:
        state["task"] = "summarize"
    elif "explain" in query:
        state["task"] = "explain"
    else:
        state["task"] = "qa"

    return state


# -----------------------------
# QA Agent Node
# -----------------------------
def qa_agent_node(state: SimpleState) -> SimpleState:

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer questions clearly."),
        ("human", "{query}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "query": state["query"]
    })

    state["response"] = response.content

    return state


# -----------------------------
# Summarizer Node
# -----------------------------
def summarizer_node(state: SimpleState):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert summarizer."),
        ("human", "Summarize the following text:\n\n{query}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "query": state["query"]
    })

    state["response"] = response.content

    return state


# -----------------------------
# Explainer Node
# -----------------------------
def explain_node(state: SimpleState):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You explain concepts in simple terms."),
        ("human", "Explain the following concept simply:\n\n{query}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "query": state["query"]
    })

    state["response"] = response.content

    return state


# -----------------------------
# Graph Definition
# -----------------------------
workflow = StateGraph(SimpleState)

workflow.add_node("router", router_node)
workflow.add_node("qa_agent", qa_agent_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("explainer", explain_node)


# Entry point
workflow.set_entry_point("router")


# Conditional routing
def route_decision(state: SimpleState):

    if state["task"] == "summarize":
        return "summarizer"

    elif state["task"] == "explain":
        return "explainer"

    else:
        return "qa_agent"


workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "qa_agent": "qa_agent",
        "summarizer": "summarizer",
        "explainer": "explainer"
    }
)

workflow.add_edge("qa_agent", END)
workflow.add_edge("summarizer", END)
workflow.add_edge("explainer", END)


# Compile Graph
multi_agent = workflow.compile()
