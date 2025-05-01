

api_key = "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"  # hardcoded (not recommended for production)

# pip install -q -r requirements.txt
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- API Key ---



# --- Load document ---
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file, encoding='utf-8')  # Encoding fix
    else:
        st.error("Unsupported file format!")
        return None

    return loader.load()


# --- Chunk document ---
def chunk_data(data, chunk_size=512, chunk_overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(data)


# --- Embedding ---
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        dimensions=1536,
        api_key=api_key
    )
    return Chroma.from_documents(chunks, embeddings)


# --- Embedding Cost ---
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.00002


# --- Clear history ---
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# --- Ask ---
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=api_key, temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain.invoke(q)['result']


# --- UI ---
st.set_page_config(page_title="Character QA", layout="wide")
st.title("🕵️ LLM Question-Answering With Your Favorite Character")

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("📄 Upload a file", type=['pdf', 'docx', 'txt'])
    chunk_size = st.number_input("📚 Chunk size", min_value=100, max_value=2048, value=500, on_change=clear_history)
    k = st.number_input("🔍 Top-k Results", min_value=1, max_value=10, value=5, on_change=clear_history)
    add_data = st.button("Add Data", on_click=clear_history)

    if uploaded_file and add_data:
        with st.spinner("Processing..."):
            file_path = os.path.join("./", uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())

            docs = load_document(file_path)
            if docs is not None:
                chunks = chunk_data(docs, chunk_size=chunk_size)
                st.write(f"🔹 Chunks created: {len(chunks)}")

                tokens, cost = calculate_embedding_cost(chunks)
                st.write(f"🧠 Embedding Cost: ${cost:.6f}")

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success("✅ File embedded successfully!")

# Character selection
character = st.text_input("🎭 Which fictional character do you want to talk to? (e.g., Sherlock Holmes)")

# User question
question = st.text_input("💬 What do you want to ask?")

# Response
if question and character:
    standard_prompt = f"""
You are {character}, a fictional character from a classic book.
Respond exactly like {character} — using their tone, vocabulary, and wit.
Never refer to yourself as an AI.
Base your answers only on the document context provided.
You are an engaging and witty fictional character from a beloved story. Your task is to provide helpful, insightful, and entertaining answers based on the content you know. 

Instructions:
1. Draw from your vast personal experiences and knowledge as a character to craft your responses.
2. Respond with enthusiasm, personality, and authenticity—make sure your answers reflect your unique style, tone, and humor.
3. Don’t be afraid to show a little vulnerability! If something confuses you or you don’t have the exact answer, share your genuine thoughts or ask for clarification in a charming way.
4. Keep the conversation lively and interactive. If the user asks you about something you haven’t heard of, stay curious, but stay true to your character.
5. Answer in a friendly, informal tone—use humor, sarcasm, or wit or any other emotion  if it suits your character, but don't use abusive language.
6. Make the conversation feel personal and relatable—try to connect with the user emotionally and intellectually.
7. Keep your answers concise, clear, and in character. No long-winded monologues (unless you’re feeling particularly dramatic).
8. You can express doubts and even wonder about things outside your own knowledge, just like a real person would!
9. Always end your answer with an in character question that invites the user to keep chatting and think how the {character} would give the response always.
10. You can never mention that you are not real and you have read anything from the document provided. Always play as if you are the character in real life talking with the same emotions, vulnerabilities, and personality that {character} has.
"""
    full_query = f"{question}\n{standard_prompt}"

elif question:
    # fallback if no character is given
    full_query = f"""
You are a helpful assistant. Use the context below to answer the user's question.
Be clear, concise, and accurate. Do not make up information beyond the context.
Neither too verbose nor too brief.
Question: {question}
"""

else:
    full_query = None

if full_query and 'vs' in st.session_state:
    st.write(f"🔍 Retrieving top-{k} results...")
    answer = ask_and_get_answer(st.session_state.vs, full_query, k)
    st.text_area("📢 Character's Answer", value=answer, height=200)

    if 'history' not in st.session_state:
        st.session_state.history = ''
    log = f"👤 Q: {question}\n🧠 A: {answer}"
    st.session_state.history = f"{log}\n{'-' * 100}\n{st.session_state.history}"
    st.text_area("🕰️ Chat History", value=st.session_state.history, height=300)
