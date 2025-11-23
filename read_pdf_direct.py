import PyPDF2
import os

def read_pdf():
    pdf_path = "RESPONSIVE QUIZ PORTAL FOR ACADEMIC TESTING final report.pdf"
    if not os.path.exists(pdf_path):
        print("PDF not found.")
        return

    try:
        print(f"Reading {pdf_path}...")
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        print(f"Total text length: {len(text)}")
        
        # Search for keywords
        keywords = ["FAISS", "vector", "improvement", "percent", "%", "Streamlit", "real-time", "1234", "QID"]
        
        for kw in keywords:
            if kw.lower() in text.lower():
                print(f"\n--- Found '{kw}' ---")
                # Find all occurrences
                start = 0
                while True:
                    idx = text.lower().find(kw.lower(), start)
                    if idx == -1: break
                    
                    # Print context
                    ctx_start = max(0, idx - 100)
                    ctx_end = min(len(text), idx + 100)
                    print(f"...{text[ctx_start:ctx_end]}...")
                    
                    start = idx + 1
                    
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    read_pdf()
