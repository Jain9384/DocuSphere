import faiss
import pickle
import os

def test_database():
    """Test if the vector database loads correctly without needing OpenAI API"""
    
    # Check if vector files exist
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("[ERROR] Vector database not found!")
        print("[INFO] Files needed: vectors.index, chunks.pkl")
        return False

    try:
        # Load saved data
        print("[LOADING] Loading vector database...")
        index = faiss.read_index("vectors.index")
        
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']

        print(f"[SUCCESS] Database loaded successfully!")
        print(f"[STATS] Total pages: {total_pages}")
        print(f"[STATS] Total chunks: {len(chunks)}")
        print(f"[STATS] Vector dimensions: {index.d}")
        print(f"[STATS] Total vectors: {index.ntotal}")
        print(f"[STATS] Average chunks per page: {len(chunks) / total_pages:.1f}")
        
        # Show sample chunk
        if chunks:
            sample_chunk = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
            print(f"[SAMPLE] First chunk: {sample_chunk}")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading database: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("[TEST] Testing RAG Database")
    print("=" * 60)
    
    success = test_database()
    
    if success:
        print("\n[READY] Database is ready! To use the full RAG system:")
        print("1. Get an OpenAI API key from: https://platform.openai.com/api-keys")
        print("2. Replace 'YOUR_OPENAI_API_KEY_HERE' in question-vector.py")
        print("3. Run: python question-vector.py")
    else:
        print("\n[FAILED] Database test failed. You may need to recreate it.")
        print("Run: python pdf-vector.py")
