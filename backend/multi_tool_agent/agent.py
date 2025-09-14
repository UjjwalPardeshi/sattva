import os
from dotenv import load_dotenv
from multi_tool_agent.rag_utils import build_pipeline, query_faiss
import google.generativeai as genai  # Gemini API client
import json

# Load .env file if present
load_dotenv()

# Prepare RAG pipeline
pdf_path = "/home/diagla/code/backend/data/ayurveda_book.pdf"
chunks, faiss_index, emb_model = build_pipeline(pdf_path, model_name="all-MiniLM-L6-v2")

# Load Gemini API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment")
genai.configure(api_key=GOOGLE_API_KEY)

def generate_report(data: str) -> dict:
    # Retrieve relevant context
    context_chunks = query_faiss(data, chunks, faiss_index, emb_model, top_k=3)
    context = "\n".join(context_chunks)

    # Compose the prompt
    prompt = (
        f"User input: {data}\n"
        f"Relevant context from Ayurveda book:\n{context}\n"
        "Respond only with a JSON object with four keys exactly as below (no extra formatting, no markdown):\n"
        "1. \"user_food_dosha_mapping\" - add food item name as the heading , numerical mapping (0-1) of each dosha (Vata, Pitta, Kapha) for the food item the user mentioned,\n"
        "2. \"user_food_description\" - a concise Ayurvedic description for that food,\n"

        "3. \"suggested_food_dosha_mapping\" - add food name as the heading, numerical dosha mapping (0-1) for a healthier alternate dish or recipe,\n"
        "4. \"suggested_food_description\" - Ayurvedic description for the suggested food or dish.\n"
        "Use only JSON notation in the response.\n"
        "You are an Ayurveda AI expert. Provide clear, simple Ayurvedic insights focusing on doshas and dietary effects.\n"
    )

    # Call Gemini 2.5 flash model
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)


    # Get response text (adjust attribute if different)
    result_text = response.text if hasattr(response, "text") else str(response)

    # Parse output JSON or fallback
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {"status": "success", "report": result_text}
