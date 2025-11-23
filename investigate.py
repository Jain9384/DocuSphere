import pickle
import os
import sqlite3
import glob

def analyze_chunks():
    print("--- Analyzing chunks.pkl ---")
    if not os.path.exists("chunks.pkl"):
        print("chunks.pkl not found.")
        return

    try:
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)
        
        chunks = data['chunks']
        print(f"Loaded {len(chunks)} chunks.")
        
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            if "faiss" in chunk_lower:
                print(f"\n[CHUNK {i}] contains FAISS:")
                print(chunk[:500])
            
            if "improvement" in chunk_lower or "percent" in chunk_lower or "%" in chunk_lower:
                print(f"\n[CHUNK {i}] contains stats:")
                print(chunk[:500])

            if "streamlit" in chunk_lower:
                print(f"\n[CHUNK {i}] contains Streamlit:")
                print(chunk[:500])

    except Exception as e:
        print(f"Error reading chunks.pkl: {e}")

def search_files_for_query():
    print("\n--- Searching files for '1234' and 'query' ---")
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "investigate.py": continue
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "1234" in content:
                        print(f"Found '1234' in: {path}")
                    if "QID" in content:
                        print(f"Found 'QID' in: {path}")
            except Exception as e:
                pass

def check_all_files():
    print("\n--- Listing all files ---")
    for root, dirs, files in os.walk("."):
        for file in files:
            print(os.path.join(root, file))

if __name__ == "__main__":
    analyze_chunks()
    search_files_for_query()
    check_all_files()
