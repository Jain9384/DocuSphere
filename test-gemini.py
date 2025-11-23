import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

# Set your Google API key
genai.configure(api_key="AIzaSyCDCkbJyJsDK20K3kKcclVjwjVMiBZRqAc")

def test_gemini_rag():
    """Test the Gemini RAG system"""
    
    print("[TEST] Testing Gemini RAG System")
    print("=" * 50)
    
    # Test 1: Load database
    try:
        print("[1] Loading vector database...")
        if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
            print("[ERROR] Vector database not found!")
            return False
            
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)
        
        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']
        
        print(f"[SUCCESS] Database loaded: {len(chunks)} chunks, {total_pages} pages")
        print(f"[INFO] Vector dimensions: {index.d}")
        
    except Exception as e:
        print(f"[ERROR] Database loading failed: {e}")
        return False
    
    # Test 2: Initialize embedding model
    try:
        print("\n[2] Initializing embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"[SUCCESS] Embedding model loaded: {embedding_model.get_sentence_embedding_dimension()} dimensions")
        
    except Exception as e:
        print(f"[ERROR] Embedding model failed: {e}")
        return False
    
    # Test 3: Test embedding generation
    try:
        print("\n[3] Testing embedding generation...")
        test_question = "What is this document about?"
        query_embedding = embedding_model.encode([test_question], convert_to_numpy=True)
        query_vector = query_embedding.reshape(1, -1)
        print(f"[SUCCESS] Query embedding generated: shape {query_vector.shape}")
        
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        return False
    
    # Test 4: Test vector search
    try:
        print("\n[4] Testing vector search...")
        scores, indices = index.search(query_vector.astype('float32'), 3)
        print(f"[SUCCESS] Found {len(indices[0])} similar chunks")
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            page_num = metadata[idx]['estimated_page']
            chunk_preview = chunks[idx][:100] + "..." if len(chunks[idx]) > 100 else chunks[idx]
            print(f"   [{i+1}] Score: {score:.3f}, Page: {page_num}")
            print(f"       Preview: {chunk_preview}")
            
    except Exception as e:
        print(f"[ERROR] Vector search failed: {e}")
        return False
    
    # Test 5: Test Gemini AI
    try:
        print("\n[5] Testing Gemini AI...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build context
        context_parts = []
        for idx in indices[0]:
            chunk_text = chunks[idx]
            page_num = metadata[idx]['estimated_page']
            context_parts.append(f"[Page {page_num}]: {chunk_text}")
        context = '\n\n'.join(context_parts)
        
        prompt = f"""You are answering questions about a {total_pages}-page document. When providing answers, mention page numbers when relevant.

Context from the document:
{context}

Question: {test_question}

Please provide a comprehensive answer based on the context above:"""

        response = model.generate_content(prompt)
        print(f"[SUCCESS] Gemini response generated")
        print(f"[ANSWER] {response.text}")
        
    except Exception as e:
        print(f"[ERROR] Gemini AI failed: {e}")
        return False
    
    print("\n[COMPLETE] All tests passed! ✅")
    return True

if __name__ == "__main__":
    test_gemini_rag()
