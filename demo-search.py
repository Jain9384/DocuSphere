import faiss
import numpy as np
import pickle
import os

def demo_search():
    """Demo version that shows vector search without OpenAI API"""
    
    # Load database
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("[ERROR] Vector database not found!")
        return

    try:
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']

        print(f"[SUCCESS] Database loaded: {len(chunks)} chunks from {total_pages} pages")
        
    except Exception as e:
        print(f"[ERROR] Error loading database: {str(e)}")
        return

    # Interactive demo
    print("\n" + "=" * 60)
    print("[DEMO] RAG System Demo (Vector Search Only)")
    print("[INFO] This demo shows chunk retrieval without AI answers")
    print("[TIP] Type 'quit' to exit, 'info' for database stats")
    print("=" * 60)

    while True:
        query = input("\n[Q] Enter search terms: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("[BYE] Goodbye!")
            break
            
        if query.lower() == 'info':
            print(f"[INFO] Database Statistics:")
            print(f"   - Total pages: {total_pages}")
            print(f"   - Total chunks: {len(chunks)}")
            print(f"   - Vector dimensions: {index.d}")
            print(f"   - Sample chunk: {chunks[0][:100]}...")
            continue
            
        if not query:
            print("[WARNING] Please enter search terms!")
            continue

        # Simple keyword search in chunks (since we don't have OpenAI for embeddings)
        print(f"[SEARCH] Searching for: '{query}'")
        
        # Find chunks containing the search terms
        matching_chunks = []
        for i, chunk in enumerate(chunks):
            if any(term.lower() in chunk.lower() for term in query.split()):
                page_num = metadata[i]['estimated_page']
                matching_chunks.append((i, chunk, page_num))
        
        if matching_chunks:
            print(f"[FOUND] {len(matching_chunks)} matching chunks:")
            for i, (chunk_idx, chunk_text, page_num) in enumerate(matching_chunks[:3]):  # Show top 3
                preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                print(f"\n   [{i+1}] Page {page_num}:")
                print(f"       {preview}")
        else:
            print("[NO MATCH] No chunks found containing those terms.")
            print("[TIP] Try different keywords or check spelling.")

if __name__ == "__main__":
    demo_search()
