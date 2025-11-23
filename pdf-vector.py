import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import PyPDF2
import numpy as np
import pickle

# Set your Google API key
genai.configure(api_key="AIzaSyCDCkbJyJsDK20K3kKcclVjwjVMiBZRqAc")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good quality embeddings


def pdf_to_vectors(pdf_path):
    # Read PDF
    print(f"[PDF] Reading PDF: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)

        # Extract text from each page separately
        page_texts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            page_texts.append({
                'text': page_text,
                'page_number': page_num + 1
            })

        # Combine all text for chunking
        text = ''.join([p['text'] for p in page_texts])

    print(f"[STATS] Total pages: {total_pages}")
    print(f"[STATS] Total text length: {len(text):,} characters")
    print(f"[STATS] Average characters per page: {len(text) // total_pages:,}")

    # Create chunks with page tracking
    chunks = []
    chunk_metadata = []

    for i in range(0, len(text), 400):
        chunk_text = text[i:i + 500]
        chunks.append(chunk_text)

        # Estimate which page this chunk belongs to
        estimated_page = min((i // (len(text) // total_pages)) + 1, total_pages)
        chunk_metadata.append({
            'start_pos': i,
            'estimated_page': estimated_page
        })

    print(f"[CHUNKS] Created {len(chunks)} chunks")

    # Get embeddings using Sentence Transformers
    print("[EMBED] Getting embeddings using Sentence Transformers...")
    print(f"[MODEL] Using model: {embedding_model.get_sentence_embedding_dimension()} dimensions")

    # Generate embeddings in batches for efficiency
    batch_size = 32
    embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)

    # Create FAISS index
    print("[INDEX] Creating FAISS index...")
    embeddings = np.array(embeddings)
    embedding_dim = embeddings.shape[1]  # Get actual embedding dimensions
    print(f"[INDEX] Using {embedding_dim} dimensions")
    index = faiss.IndexFlatIP(embedding_dim)  # Use actual embedding dimensions
    index.add(embeddings.astype('float32'))

    # Save to files
    print("[SAVE] Saving to files...")
    faiss.write_index(index, "vectors.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            'chunks': chunks,
            'metadata': chunk_metadata,
            'total_pages': total_pages
        }, f)

    print("[SUCCESS] Vector database created successfully!")
    print(f"[FILES] Files saved: vectors.index, chunks.pkl")
    print(f"[STATS] Vector shape: {embeddings.shape}")
    print(f"[SAMPLE] Sample vector (first 5 dims): {embeddings[0][:5]}")

    return embeddings, chunks


# Usage
if __name__ == "__main__":
    # Convert PDF to vectors (run this once)
    pdf_file = "RESPONSIVE QUIZ PORTAL FOR ACADEMIC TESTING final report.pdf"  # Your actual PDF file
    embeddings, chunks = pdf_to_vectors(pdf_file)

    print("\n[COMPLETE] Setup complete! Now you can run 'question-vector.py' to chat with your PDF!")