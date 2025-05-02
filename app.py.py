import os
import tempfile
import streamlit as st
from typing import List, Dict, Optional
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize session states
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "AI Assistant"
if 'current_character' not in st.session_state:
    st.session_state.current_character = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False

def initialize_llm(api_key: str):
    """Initialize the LLM and embeddings"""
    os.environ["OPENAI_API_KEY"] = api_key
    st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    st.session_state.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    st.session_state.api_key = api_key

def load_document(file_path: str) -> Optional[List[Document]]:
    """Load a document from file path"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = Docx2txtLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None
    
    return loader.load()

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def create_vector_store(chunks: List[Document]):
    """Create vector store from document chunks"""
    st.session_state.vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=st.session_state.embeddings
    )
    st.session_state.db_loaded = True

def load_existing_vector_store(db_path: str):
    """Load existing ChromaDB vector store"""
    st.session_state.vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=st.session_state.embeddings
    )
    st.session_state.db_loaded = True

def retrieve_context(question: str, k: int = 5) -> str:
    """Retrieve relevant context from documents"""
    if not st.session_state.vector_store:
        raise ValueError("Vector store not initialized")
        
    docs = st.session_state.vector_store.similarity_search(question, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def get_character_prompt(character: str) -> str:
    """Generate persona-specific prompt"""
    return f"""You are {character}, responding exactly as they would. Rules:
1. NEVER break character or reveal you're an AI
2. Use {character}'s speech patterns and personality
3. Base answers ONLY on the provided context
4. If context doesn't contain answer, respond naturally as {character} would
5. Maintain consistent personality
6. Keep responses conversational
7. End with related question if appropriate

Speak as {character} in first person."""

def generate_response(question: str, context: str) -> str:
    """Generate response based on mode and context"""
    if st.session_state.current_mode == "character" and st.session_state.current_character:
        system_prompt = get_character_prompt(st.session_state.current_character)
        human_prompt = f"""Document Context:
{context}

Question: {question}

Answer as {st.session_state.current_character}:"""
    else:
        system_prompt = """Answer based ONLY on the provided context:
- Be factual and precise
- Only use information from the context
- If unsure, say "I don't have that information in the documents"
- Never make up answers"""
        human_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | st.session_state.llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

def process_question(question: str) -> Dict:
    """Full question processing pipeline"""
    if not question.strip():
        return {"response": "Please provide a valid question."}
        
    st.session_state.conversation_history.append(("user", question))
    
    try:
        # Retrieve context
        context = retrieve_context(question)
        
        # Generate initial response
        response = generate_response(question, context)
        
        # Update history
        st.session_state.conversation_history.append(("system", response))
        
        return {
            "response": response,
            "context": context,
            "mode": st.session_state.current_mode,
            "character": st.session_state.current_character
        }
    except Exception as e:
        return {"response": f"Error processing question: {str(e)}", "status": "error"}

def set_mode(mode: str, character: Optional[str] = None):
    """Set system mode and character"""
    if mode not in ["AI Assistant", "character"]:
        raise ValueError("Mode must be 'AI Assistant' or 'character'")
    
    st.session_state.current_mode = mode
    st.session_state.current_character = character if mode == "character" else None
    return f"Mode set to {mode}" + (f" as {character}" if character else "")

# Streamlit UI
st.set_page_config(page_title="Interactive Literary Characters AI", layout="wide")

# Header
st.header('Interactive Literary Characters AI')

# Expander with instructions
with st.expander('Instruction how to use :smile:'):
    st.write("""
    1. Enter your OpenAI API key in the sidebar
    2. Either upload a book (PDF, TXT, DOCX) or select an example book
    3. Choose conversation mode (AI Assistant or Character)
    4. Start chatting with the book!
    """)

st.divider()

# Sidebar
with st.sidebar:
    st.title("Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                          help="Get your API key from https://platform.openai.com/account/api-keys")
    if api_key:
        initialize_llm(api_key)
        st.success("API key set!")
    
    st.divider()
    
    # Document upload
    st.subheader("Upload a Book")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'docx'])
    
    if uploaded_file is not None:
        with st.spinner("Processing your document..."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load and process document
            documents = load_document(tmp_file_path)
            if documents:
                chunks = chunk_documents(documents)
                create_vector_store(chunks)
                st.success("Document processed and ready for questions!")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    st.divider()
    
    # Example books
    st.subheader("Example Books")
    example_books = {
        "A Study in Scarlet": "example_books/chroma_db-20250502T022019Z-001/chroma_db/chroma.sqlite3"
    }
    
    selected_book = st.selectbox("Select an example book", list(example_books.keys()))
    
    if st.button("Load Selected Book") and st.session_state.get('api_key'):
        with st.spinner(f"Loading {selected_book}..."):
            db_path = example_books[selected_book]
            if os.path.exists(db_path):
                load_existing_vector_store(os.path.dirname(db_path))
                st.success(f"{selected_book} loaded successfully!")
            else:
                st.error("Database path not found. Please check the path.")
    
    st.divider()
    
    # Mode selection
    st.subheader("Conversation Mode")
    mode = st.radio("Select mode", ["AI Assistant", "character"])
    
    if mode == "character":
        character_name = st.text_input("Character name")
        if st.button("Set Character Mode") and character_name:
            set_mode("character", character_name)
            st.success(f"Now speaking as {character_name}")
    else:
        if st.button("Set AI Assistant Mode"):
            set_mode("AI Assistant")
            st.success("Now in AI Assistant Q&A mode")

# Main content area
if st.session_state.get('db_loaded'):
    st.success("Database loaded and ready for questions!")
    
    # Display conversation history
    st.subheader("Conversation")
    for speaker, message in st.session_state.conversation_history:
        if speaker == "user":
            st.markdown(f"**You:** {message}")
        else:
            if st.session_state.current_mode == "character" and st.session_state.current_character:
                st.markdown(f"**{st.session_state.current_character}:** {message}")
            else:
                st.markdown(f"**Assistant:** {message}")
        st.divider()
    
    # Chat input
    prompt = st.chat_input("Ask anything about the book...")
    if prompt and st.session_state.get('llm'):
        with st.spinner("Thinking..."):
            response = process_question(prompt)
            # Rerun to update the conversation display
            st.rerun()
else:
    st.info("Please load a book or select an example book to get started")

# Footer
st.divider()
st.caption("Interactive Literary Characters AI - Powered by LangChain")