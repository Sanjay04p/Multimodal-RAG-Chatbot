from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv('API_KEY')

from langchain_groq import ChatGroq


llm=ChatGroq(api_key=api_key,model='llama-3.1-8b-instant')

def build_history_text(history):
    lines = []
    for msg in history:
        lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
    return "\n".join(lines)

def retrieve_context(query, vectorstore, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n".join([r.page_content for r in results])

def rag_chat(query, vectorstore, memory):
    history_text = build_history_text(memory["messages"])
    context = retrieve_context(query, vectorstore)

    prompt = f"""
You are a helpful assistant using RAG and conversation memory.

Conversation so far:
{history_text}

Relevant context from documents:
{context}

User question:
{query}

Respond naturally and clearly.
"""

    response = llm.invoke(prompt).content
    return response