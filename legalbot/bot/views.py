import os
import pandas as pd
import faiss
from django.shortcuts import render
from django.http import JsonResponse
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -------------------------------
# Paths and File Checks
# -------------------------------
INDEX_PATH = "law_faiss.index"
CSV_PATH = "law_mapping.csv"

if not os.path.exists(INDEX_PATH) or not os.path.exists(CSV_PATH):
    raise FileNotFoundError("❌ FAISS index or mapping CSV not found.")

# -------------------------------
# Load FAISS index and mapping
# -------------------------------
faiss_index = faiss.read_index(INDEX_PATH)   # <-- renamed here
df = pd.read_csv(CSV_PATH)

# -------------------------------
# Load embedding model
# -------------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# Setup Gemini LLM
# -------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBx8fBsl3rEsveG3-v7uHLi6Eupq-2lcJ0")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------
# FAISS search function
# -------------------------------
def search_faiss(query, k=3):
    """Search FAISS index for top-k matches."""
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_vec, k)  # <-- uses faiss_index
    return [df.iloc[i]["text"] for i in indices[0] if 0 <= i < len(df)]

# -------------------------------
# Views
# -------------------------------
def index(request):
    """Render the home page."""
    return render(request, "index.html")

def ask(request):
    """Handle user query and return JSON answer."""
    query = request.GET.get("q", "").strip()
    if not query:
        return JsonResponse({"answer": "Please enter a query."})

    try:
        # Search FAISS
        matches = search_faiss(query)
        if not matches:
            results = df[df['text'].str.contains(query, case=False, na=False)]
            if not results.empty:
                matches = results['text'].tolist()[:3]
            else:
                matches = ["No relevant documents found."]

        context = "\n".join(matches)

        # Construct prompt for Gemini
        prompt = f"""
You are a legal assistant for Indian law.
Use the following context to answer the user's question.

Context:
{context}

Question: {query}
Answer in clear, concise terms:
"""

        # Generate answer with Gemini
        response = model.generate_content(prompt)
        answer_text = getattr(response, "text", str(response))

        return JsonResponse({"answer": answer_text})

    except Exception as e:
        return JsonResponse({"answer": f"Error occurred: {str(e)}"})

