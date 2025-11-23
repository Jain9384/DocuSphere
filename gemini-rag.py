import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

# Set your Google API key
genai.configure(api_key="AIzaSyCDCkbJyJsDK20K3kKcclVjwjVMiBZRqAc")

# Initialize the embedding model (same as used in pdf-vector.py)
print("[INIT] Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"[INIT] Model loaded: {embedding_model.get_sentence_embedding_dimension()} dimensions")

def ask_question(question):
    """Ask a question using Gemini RAG system"""
    
    # Check if vector files exist
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("[ERROR] Vector database not found!")
        print("[INFO] Please run 'pdf-vector.py' first to create the database.")
        return None

    try:
        # Load saved data
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']

        # Get question embedding using Sentence Transformers
        query_embedding = embedding_model.encode([question], convert_to_numpy=True)
        query_vector = query_embedding.reshape(1, -1)

        # Search similar chunks
        scores, indices = index.search(query_vector.astype('float32'), 3)

        # Show similarity scores and page info for debugging
        print(f"[SEARCH] Found {len(indices[0])} relevant chunks:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            page_num = metadata[idx]['estimated_page']
            print(f"   Chunk {i + 1}: Score {score:.3f} (~Page {page_num})")

        # Build context with page information
        context_parts = []
        for idx in indices[0]:
            chunk_text = chunks[idx]
            page_num = metadata[idx]['estimated_page']
            context_parts.append(f"[Page {page_num}]: {chunk_text}")

        context = '\n\n'.join(context_parts)

        # Get answer from Gemini AI with page context
        print("[AI] Generating answer with Gemini...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""You are answering questions about a {total_pages}-page document. When providing answers, mention page numbers when relevant.

Context from the document:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above:"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"[ERROR] Error processing question: {str(e)}")
        return None

def main():
    """Main interactive loop"""
    
    # Check if database exists
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("[ERROR] Vector database not found!")
        print("[INFO] Please run 'pdf-vector.py' first to create the database.")
        return

    # Load database info
    try:
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        total_pages = data['total_pages']

        print(f"[SUCCESS] Database loaded: {len(chunks)} chunks from {total_pages} pages")
    except Exception as e:
        print(f"[ERROR] Error loading database: {str(e)}")
        return

    # Interactive question loop
    print("\n" + "=" * 60)
    print("[RAG] Gemini AI RAG System Ready!")
    print("[TIP] Type 'bye', 'quit', 'exit', or 'q' to exit")
    print("[TIP] Type 'info' for database statistics")
    print("=" * 60)

    while True:
        question = input("\n[Q] Your question: ").strip()

        # Check for exit commands
        if question.lower() in ['bye', 'quit', 'exit', 'q']:
            print("[BYE] Goodbye! Thanks for using the Gemini RAG system!")
            break

        # Show database info
        if question.lower() == 'info':
            print(f"[INFO] Database Info:")
            print(f"   - Total pages: {total_pages}")
            print(f"   - Total chunks: {len(chunks)}")
            print(f"   - Vector dimensions: {index.d}")
            print(f"   - Embedding model: all-MiniLM-L6-v2")
            print(f"   - AI model: Gemini 2.5 Flash")
            print(f"   - Average chunks per page: {len(chunks) / total_pages:.1f}")
            continue

        # Skip empty questions
        if not question:
            print("[WARNING] Please enter a question!")
            continue

        print("[SEARCH] Searching and generating answer...")
        answer = ask_question(question)

        if answer:
            print(f"\n[ANSWER] {answer}")
        else:
            print("[ERROR] Sorry, I couldn't generate an answer. Please try a different question.")

if __name__ == "__main__":
    main()
