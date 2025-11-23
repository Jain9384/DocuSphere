import streamlit as st
import os
import importlib.util
import sys
import database

# Helper to import modules with hyphens
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import existing scripts
try:
    pdf_vector = import_module_from_path("pdf_vector", "pdf-vector.py")
    gemini_rag = import_module_from_path("gemini_rag", "gemini-rag.py")
except Exception as e:
    st.error(f"Error importing backend scripts: {e}")
    st.stop()

# Initialize DB
database.init_db()

st.set_page_config(page_title="DocuSphere", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #e8f0fe;
    }
</style>
""", unsafe_allow_html=True)

# Tabs for Chat and History
tab1, tab2 = st.tabs(["💬 Chat", "📜 History Management"])

with tab1:
    with st.sidebar:
        st.title("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Processing PDF... This may take a minute."):
                    # Save temp file
                    temp_path = "temp_uploaded.pdf"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process
                    try:
                        pdf_vector.pdf_to_vectors(temp_path)
                        st.success("✅ Document processed successfully!")
                        # Cleanup
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error processing document: {e}")

        st.divider()
        st.markdown("### Status")
        if os.path.exists("vectors.index") and os.path.exists("chunks.pkl"):
            st.success("🟢 Database Ready")
        else:
            st.warning("🔴 Database Not Found")

    st.title("DocuSphere")
    st.caption("Powered by Gemini AI & FAISS")

    # Initialize chat history in session state for display
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = gemini_rag.ask_question(prompt)
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Save to Database
                        database.add_record(prompt, response)
                    else:
                        st.error("Could not generate a response. Please check the database.")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.header("History Management")
    st.caption("View, Edit, or Delete past conversations.")
    
    # Refresh button
    if st.button("🔄 Refresh History"):
        st.rerun()

    records = database.get_all_records()
    
    if not records:
        st.info("No history found.")
    
    for record in records:
        id, timestamp, question, answer = record
        
        with st.expander(f"[{timestamp}] {question[:50]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Edit Form
                with st.form(key=f"edit_{id}"):
                    new_q = st.text_input("Question", value=question)
                    new_a = st.text_area("Answer", value=answer)
                    if st.form_submit_button("💾 Save Changes"):
                        database.update_record(id, new_q, new_a)
                        st.success("Record updated!")
                        st.rerun()
            
            with col2:
                st.write("") # Spacer
                st.write("")
                if st.button("🗑️ Delete", key=f"del_{id}"):
                    database.delete_record(id)
                    st.warning("Record deleted.")
                    st.rerun()
