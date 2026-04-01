import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
import io
from pypdf import PdfReader
import numpy as np

load_dotenv()

app = Flask(__name__)
# Enable CORS for all domains so local testing works
CORS(app)

# Initialize the new google-genai client
# It automatically picks up GEMINI_API_KEY from the environment variable
client = genai.Client()

# In-memory storage for PDF text chunks and their embeddings
pdf_chunks = []
pdf_embeddings = []

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    global pdf_chunks, pdf_embeddings
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({'error': 'No selected PDF file'}), 400
    
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += text + extracted + "\n"
        
        if not text.strip():
            return jsonify({'error': 'No extractable text found in PDF'}), 400
            
        chunks = chunk_text(text)
        
        response = client.models.embed_content(
            model='text-embedding-004',
            contents=chunks
        )
        
        embeddings = [np.array(e.values) for e in response.embeddings]
        
        pdf_chunks = chunks
        pdf_embeddings = embeddings
        
        return jsonify({'message': f'Successfully processed PDF into {len(chunks)} chunks.'})
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400

    prompt = data['prompt']

    try:
        if not pdf_chunks or not pdf_embeddings:
            return jsonify({'response': 'Please upload a PDF document first. I am configured to only answer questions based on an uploaded PDF.'})
            
        prompt_embed_response = client.models.embed_content(
            model='text-embedding-004',
            contents=prompt
        )
        prompt_embed = np.array(prompt_embed_response.embeddings[0].values)
        
        similarities = []
        for emb in pdf_embeddings:
            sim = np.dot(prompt_embed, emb) / (np.linalg.norm(prompt_embed) * np.linalg.norm(emb))
            similarities.append(sim)
            
        top_k = min(3, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        context_text = "\n\n".join([pdf_chunks[i] for i in top_indices])
        
        augmented_prompt = f"You are a helpful assistant. Use ONLY the following extracted text from a PDF document to answer the user's question. If the answer cannot be found in the provided text, respond with 'The provided document does not contain this information.' Do NOT use your general knowledge.\n\n--- PDF Context ---\n{context_text}\n\n--- User Question ---\n{prompt}"
        
        # Use gemini-2.5-flash as requested
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=augmented_prompt,
        )
        return jsonify({'response': response.text})
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_pdf():
    global pdf_chunks, pdf_embeddings
    pdf_chunks = []
    pdf_embeddings = []
    return jsonify({'message': 'PDF context cleared successfully'})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
