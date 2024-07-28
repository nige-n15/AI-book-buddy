from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import json
import logging
import anthropic

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "book-brain"
index = pc.Index(index_name)

# Initialize the sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Load the sentence storage
sentence_storage_path = 'sentence_storage.json'
logger.info(f"Attempting to load sentence storage from {os.path.abspath(sentence_storage_path)}")
if os.path.exists(sentence_storage_path):
    with open(sentence_storage_path, 'r') as f:
        sentence_storage = json.load(f)
    logger.info(f"Loaded {len(sentence_storage)} paragraphs from sentence storage")
    logger.info(f"First 5 keys in sentence_storage: {list(sentence_storage.keys())[:5]}")
    logger.info(f"Sample paragraph: {next(iter(sentence_storage.values()))[:200]}")
else:
    logger.error(f"sentence_storage.json not found at {os.path.abspath(sentence_storage_path)}")
    sentence_storage = {}


def process_with_anthropic(query, raw_results):
    prompt = f"{HUMAN_PROMPT} Based on the following query and raw results, provide a concise and informative answer. Synthesize the information and present it clearly. Ignore any irrelevant metadata or bibliographic information.\n\nQuery: {query}\n\nRaw Results:\n{raw_results}\n\n{AI_PROMPT}"

    response = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=prompt
    )

    return response.completion

@app.route('/query', methods=['POST'])
def query_books():
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', 5)

        if not query:
            return jsonify({"error": "No query provided"}), 400

        query_embedding = model.encode([query])[0].tolist()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        logger.info(f"Raw results from Pinecone: {results}")

        processed_results = []
        for match in results['matches']:
            paragraph_id = match['id']
            logger.info(f"Looking for paragraph_id: {paragraph_id}")
            paragraph = sentence_storage.get(paragraph_id, "Paragraph not found")
            if paragraph == "Paragraph not found":
                logger.warning(f"Paragraph not found for id: {paragraph_id}")

            processed_results.append({
                'paragraph': paragraph,
                'score': float(match['score']),
                'book': match['metadata'].get('book', 'Unknown book')
            })

        raw_results = "\n\n".join([f"From '{r['book']}':\n{r['paragraph']}" for r in processed_results])
        anthropic_response = process_with_anthropic(query, raw_results)

        return jsonify({
            "query": query,
            "anthropic_response": anthropic_response,
            "raw_results": processed_results
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "An error occurred while processing the query"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200



@app.route('/debug_paragraph/<paragraph_id>')
def debug_paragraph(paragraph_id):
    paragraph = sentence_storage.get(paragraph_id, "Paragraph not found")
    return jsonify({
        "paragraph_id": paragraph_id,
        "paragraph": paragraph[:200] if paragraph != "Paragraph not found" else paragraph
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)