import os
from dotenv import load_dotenv
from multi_tool_agent.rag_utils import build_pipeline, query_faiss
import google.generativeai as genai  # Gemini API client
import json

# Load .env file if present
load_dotenv()

# Prepare RAG pipeline
pdf_path = "/home/diagla/code/sattva/backend/data/ayurveda_book.pdf"
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
        "Respond only with a valid JSON object. Do NOT include markdown, code blocks, backticks, or any extra formatting.\n"
        "If user messages something random reply with pls specify what you ate/drank ? "
        "Your JSON response must include exactly these keys, no more, no less:\n"
        "Numerical mapping should be consistent and should not change again and again"
        "1. \"user_food_dosha_mapping\" - map the food item/drink user mentioned to dosha values (Vata, Pitta, Kapha) each between 0 and 1.\n"
        "2. \"user_food_description\" - a concise Ayurvedic description for that food item.\n"
        "3. \"suggested_food_dosha_mapping\" - map a healthier suggested alternate dish or recipe to dosha values along with quantity of the food (0-1).\n"
        "4. \"suggested_food_description\" - an Ayurvedic description of the suggested alternate. with quantity\n"
        "Additionally, add a key \"rasa_panchaka_explanation\" which is an object holding one-sentence descriptions for each of the five Rasa Panchaka properties (Rasa, Guna, Virya, Vipaka, Prabhava) as they relate to the food mentioned.\n"
        "Return the response strictly as a JSON object with these keys. No commentary or extra text.\n"
        "Dont give every suggested food as moong daal khichdi"
         "Example response format:\n"
        "{\n"
        "  \"user_food_dosha_mapping\": {\"Daal and Roti\": {\"Vata\": 0.4, \"Pitta\": 0.3, \"Kapha\": 0.5}},\n"
        "  \"user_food_description\": \"Daal and Roti is a nourishing meal combining light and heavy qualities, generally balancing but slightly increasing Kapha.\",\n"
        "  \"suggested_food_dosha_mapping\": {\"Mung Daal Khichdi\": {\"Vata\": 0.2, \"Pitta\": 0.2, \"Kapha\": 0.3}},\n"
        "  \"suggested_food_description\": \"Mung Daal Khichdi is light, digestive, and suitable for all dosha types.\",\n"
        "  \"rasa_panchaka_explanation\": {\n"
        "    \"Rasa\": \"Daal is astringent or sweet; roti is predominantly sweet.\",\n"
        "    \"Guna\": \"Combines dry and light qualities of daal with heavy and unctuous qualities of roti.\",\n"
        "    \"Virya\": \"Both have cooling (sheeta) potency.\",\n"
        "    \"Vipaka\": \"Post-digestive tastes are pungent or sweet, nourishing the body.\",\n"
        "    \"Prabhava\": \"Properly prepared meal strengthens and nourishes the body.\"\n"
        "  }\n"
        "}\n"
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
