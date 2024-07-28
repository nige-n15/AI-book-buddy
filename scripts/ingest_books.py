import os
from dotenv import load_dotenv
import PyPDF2
import ebooklib
from ebooklib import epub
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import json
import textwrap
import re

#load_dotenv()

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Print the directory of the script being run
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
print(f"Looking for .env file at: {env_path}")

# Check if the .env file exists
if os.path.exists(env_path):
    print(f".env file found at {env_path}")
    # Load the .env file
    load_dotenv(dotenv_path=env_path)
else:
    print(f".env file not found at {env_path}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print(os.getenv("PINECONE_API_KEY"))
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "book-brain"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Make sure this matches your model's output dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-west-2")
        )
    )

# Get the index
index = pc.Index(index_name)
# At the beginning of your ingest_books.py script, after initializing the Pinecone index
#index.delete(delete_all=True)
print("Cleared all vectors from the Pinecone index.")
if os.path.exists('sentence_storage.json'):
    os.remove('sentence_storage.json')
    print("Removed existing sentence_storage.json")
# Initialize the sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

logger = logging.getLogger(__name__)
def clean_text(text):
    # Remove PDF artifacts
    text = re.sub(r'_book\.indb\s+\d+', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def extract_text(file_path):
    logger.info(f"Attempting to extract text from: {file_path}")

    if file_path.endswith('.pdf'):
        logger.info("Detected PDF file")
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                logger.info(f"PDF has {len(reader.pages)} pages")
                text = ''
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    logger.info(f"Extracted {len(page_text)} characters from page {i + 1}")
                    text += page_text + '\n'
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ''
    elif file_path.endswith('.epub'):
        logger.info("Detected EPUB file")
        try:
            book = epub.read_epub(file_path)
            text = ''
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                item_text = item.get_content().decode('utf-8')
                logger.info(f"Extracted {len(item_text)} characters from EPUB item")
                text += item_text + '\n\n'
        except Exception as e:
            logger.error(f"Error extracting text from EPUB: {str(e)}")
            return ''
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        return ''

    cleaned_text = clean_text(text)
    logger.info(f"Extracted and cleaned text length: {len(cleaned_text)} characters")
    logger.info(f"First 500 characters of extracted text: {cleaned_text[:500]}")
    return cleaned_text

def preprocess_text(text):
    # Remove PDF artifacts and non-content elements
    text = re.sub(r'\d+_Book\.indb\s+\d+', '', text)
    text = re.sub(r'\d+/\d+/\d+\s+\d+:\d+\s+[AP]M', '', text)

    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)

    processed_paragraphs = []
    for paragraph in paragraphs:
        # Basic cleaning
        paragraph = paragraph.strip().lower()
        # Remove excessive whitespace
        paragraph = ' '.join(paragraph.split())
        # Remove very short paragraphs and likely headers
        if len(paragraph) > 100 and not paragraph.startswith('chapter') and not paragraph.isnumeric():
            processed_paragraphs.append(paragraph)

    # Further split long paragraphs
    final_paragraphs = []
    for paragraph in processed_paragraphs:
        if len(paragraph) > 1000:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_paragraph = ''
            for sentence in sentences:
                if len(current_paragraph) + len(sentence) < 1000:
                    current_paragraph += ' ' + sentence
                else:
                    if current_paragraph:
                        final_paragraphs.append(current_paragraph.strip())
                    current_paragraph = sentence
            if current_paragraph:
                final_paragraphs.append(current_paragraph.strip())
        else:
            final_paragraphs.append(paragraph)

    logger.info(f"Number of processed paragraphs: {len(final_paragraphs)}")
    if final_paragraphs:
        logger.info(f"First processed paragraph: {final_paragraphs[0][:200]}...")
        logger.info(f"Last processed paragraph: {final_paragraphs[-1][:200]}...")

    return final_paragraphs

def process_books(directory, max_books=None, batch_size=100):
    processed_books = []
    total_vectors = 0
    sentence_storage = {}

    files = [f for f in os.listdir(directory) if f.endswith(('.pdf', '.epub'))]
    files = files[:max_books] if max_books else files

    for file in tqdm(files, desc="Processing books"):
        file_path = os.path.join(directory, file)
        logger.info(f"Processing: {file_path}")

        text = extract_text(file_path)
        if not text:
            logger.warning(f"No text extracted from {file}")
            continue

        logger.info(f"Extracted text length: {len(text)} characters")
        logger.info(f"First 500 characters of extracted text: {text[:500]}")

        processed_paragraphs = preprocess_text(text)
        logger.info(f"Processed {len(processed_paragraphs)} paragraphs from {file}")

        if processed_paragraphs:
            logger.info(f"First paragraph: {processed_paragraphs[0]}")
            logger.info(f"Last paragraph: {processed_paragraphs[-1]}")

        vectors = []
        for i, paragraph in enumerate(processed_paragraphs):
            embedding = model.encode([paragraph])[0].tolist()
            paragraph_id = f"{file}_{i}"  # Consistent ID format without 'p'
            vectors.append((paragraph_id, embedding, {"book": file, "paragraph_id": paragraph_id}))
            sentence_storage[paragraph_id] = paragraph  # Use the same ID format for sentence_storage

            if len(vectors) >= batch_size:
                try:
                    index.upsert(vectors=vectors)
                    total_vectors += len(vectors)
                    vectors = []
                except Exception as e:
                    logger.error(f"Error upserting to Pinecone: {str(e)}")

        if vectors:
            try:
                index.upsert(vectors=vectors)
                total_vectors += len(vectors)
            except Exception as e:
                logger.error(f"Error upserting to Pinecone: {str(e)}")

        processed_books.append({
            'file_name': file,
            'paragraphs_processed': len(processed_paragraphs)
        })

    logger.info(f"Sample of paragraph IDs being stored: {list(sentence_storage.keys())[:5]}")
    with open('sentence_storage.json', 'w') as f:
        json.dump(sentence_storage, f)
    logger.info(f"Wrote {len(sentence_storage)} paragraphs to sentence_storage.json")

    return processed_books, total_vectors
#  Main execution
books_directory = '/Users/nige.karus/Documents/Books/'
max_books_to_process = 1  # Set to None to process all books

processed_books, total_vectors = process_books(books_directory, max_books_to_process)

logger.info(f"Processed {len(processed_books)} books.")
logger.info(f"Total vectors stored in Pinecone: {total_vectors}")
for book in processed_books:
    logger.info(f"File: {book['file_name']}")
    logger.info(f"Paragraphs processed: {book['paragraphs_processed']}")
    logger.info("---")