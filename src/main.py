from fastapi import FastAPI
from pydantic import BaseModel

from src.workflows.simple_workflow import multi_agent

app = FastAPI(title="Simple Agentic RAG")


# -----------------------------
# Request Schema
# -----------------------------
class QueryRequest(BaseModel):
    query: str


# -----------------------------
# Query Endpoint
# -----------------------------
@app.post("/query")
def query_agent(request: QueryRequest):

    result = multi_agent.invoke({
        "query": request.query
    })

    return {
        "response": result["response"]
    }


# -----------------------------
# Summarize Endpoint
# -----------------------------
@app.post("/summarize")
def summarize_text(request: QueryRequest):

    result = multi_agent.invoke({
        "query": f"summarize: {request.query}"
    })

    return {
        "summary": result["response"]
    }


# -----------------------------
# Explain Endpoint
# -----------------------------
@app.post("/explain")
def explain_concept(request: QueryRequest):

    result = multi_agent.invoke({
        "query": f"explain: {request.query}"
    })

    return {
        "explanation": result["response"]
    }


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }
