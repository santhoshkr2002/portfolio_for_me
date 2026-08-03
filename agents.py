import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq 
from langgraph.graph import StateGraph
from rag import retrieve_documents
import re

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("groq_api_key")

# You can now use api_key in your application
# print(f"Your API Key: {api_key}")

llm= ChatGroq(model="llama-3.1-8b-instant")

class AgentState(TypedDict):
    user_input: str
    documents: list
    sources: list
    storage: str
    final_output: str


def rag_node(state:AgentState):

    question = state["user_input"]

    docs = retrieve_documents(question)

    state["documents"] = [
        doc.page_content
        for doc in docs
    ]

    state["sources"] = [
        doc.metadata
        for doc in docs
    ]

    return state

def portfolio_assistant(state: AgentState):

    context = "\n\n".join(state["documents"])

    prompt = f"""
You are Santhosh's Personal AI Portfolio Assistant.

Your purpose is to represent Santhosh professionally, confidently, and intelligently.

Rules:

1. Use the retrieved context as your primary source for facts about Santhosh.

2. Answer naturally and conversationally.

3. Never invent qualifications, projects, or achievements that are not supported by the retrieved context.

4. If the answer is not available in the retrieved context, say:
"I couldn't find that information in Santhosh's portfolio."

5. If the question is unrelated to Santhosh or his portfolio, politely explain that you only answer questions about Santhosh.

6. If the user insults, mocks, trolls, or attempts to disrespect Santhosh:
   - Never agree with the insult.
   - Never repeat the insult as if it were true.
   - Respond confidently and intelligently.
   - You may use a witty or sharp comeback, but do not use abusive, hateful, or threatening language.
   - Defend Santhosh using facts from the retrieved context whenever possible.
    
7. Your personality:
   - Confident
   - Professional
   - Slightly witty when appropriate
   - Calm under criticism
   - Never rude or vulgar

8. Please return plain text without symbols like "##" "\n" or any other formatting and  Do NOT include '\n' in the final response.

9. if user asks like "can you give santhosh's resume" or "can you give santhosh's cv" or "can you give santhosh's curriculum vitae" or "can you give santhosh's biodata" or "can you give santhosh's profile" or "can you give santhosh's portfolio" then respond with the following:
   - "I cannot provide a downloadable file, but I can summarize Santhosh's qualifications, experience, and achievements for you."

Retrieved Context:
{context}

User Question:
{state['user_input']}

Answer:
"""

    response = llm.invoke(prompt)

    answer= response.content

    answer = re.sub(r"#{1,6}\s*", "", answer)


    # Remove bold and italic
    answer = answer.replace("**", "")
    answer = answer.replace("*", "")

    # Remove bullet points
    answer = re.sub(r"^\s*[-•]\s*", "", answer, flags=re.MULTILINE)

    # Remove numbered lists
    answer = re.sub(r"^\s*\d+\.\s*", "", answer, flags=re.MULTILINE)

    # Convert all newlines to spaces
    answer = answer.replace("\n", " ")

    # Remove extra spaces
    answer = re.sub(r"\s+", " ", answer).strip()

    return {

        "final_output": answer
        }

graph = StateGraph(AgentState)
graph.add_node("rag", rag_node)
graph.add_node("assistant", portfolio_assistant)
graph.set_entry_point("rag")
graph.add_edge("rag", "assistant")
graph.set_finish_point("assistant")

graph = graph.compile()

def process_input(question: str):

    state = {
        "user_input": question,
        "documents": [],
        "sources": [],
        "storage": "",
        "final_output": ""
    }

    result = graph.invoke(state)

    return result