from flask import Flask, request, jsonify, current_app
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import re
from dotenv import load_dotenv
import json
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
else:
    logger.error(f"sentence_storage.json not found at {os.path.abspath(sentence_storage_path)}")
    sentence_storage = {}

@app.route('/debug_paragraph/<paragraph_id>')
def debug_paragraph(paragraph_id):
    paragraph = sentence_storage.get(paragraph_id, "Paragraph not found")
    return jsonify({
        "paragraph_id": paragraph_id,
        "paragraph": paragraph[:200] if paragraph != "Paragraph not found" else paragraph
    })

def query_books():
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', 5)

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Generate embedding for the query
        query_embedding = model.encode([query])[0].tolist()

        # Perform the similarity search
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        logger.info(f"Raw results from Pinecone: {results}")

        # Process and return the results
        processed_results = []
        for match in results['matches']:
            paragraph_id = match['id']
            paragraph = sentence_storage.get(paragraph_id, "Paragraph not found")
            score = float(match['score'])

            if score > 0.5:  # Adjust this threshold as needed
                processed_paragraph = post_process_paragraph(paragraph)
                processed_results.append({
                    'paragraph': processed_paragraph,
                    'score': score,
                    'book': match['metadata'].get('book', 'Unknown book')
                })

        # Combine results into a single response
        combined_response = "\n\n".join([f"From '{r['book']}':\n{r['paragraph']}" for r in processed_results])

        return jsonify({
            "query": query,
            "response": combined_response,
            "individual_results": processed_results
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "An error occurred while processing the query"}), 500


def post_process_paragraph(paragraph):
    # Remove remaining artifacts
    paragraph = re.sub(r'_book\.indb\s+\d+', '', paragraph)
    # Remove reference numbers
    paragraph = re.sub(r'\[\d+\]', '', paragraph)
    # Capitalize first letter of sentences
    paragraph = '. '.join(s.capitalize() for s in paragraph.split('. '))
    return paragraph.strip()


def health_check():
    return jsonify({"status": "healthy"}), 200