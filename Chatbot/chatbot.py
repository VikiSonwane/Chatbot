import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from gtts import gTTS
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_AI_API_KEY = os.getenv("GEMINI_AI_API_KEY")

# Check if the API key is set
if not GEMINI_AI_API_KEY:
    st.error("GEMINI_AI_API_KEY is not set. Please add it to your .env file or Streamlit secrets.")
    st.stop()

# Set up the Streamlit app
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.header('My First Chatbot 💬')

# --- Chatbot logic in the sidebar ---
with st.sidebar:
    st.title('Your Documents')
    # Use st.file_uploader to allow the user to upload a PDF file.
    file = st.file_uploader('Please upload your PDF and ask questions', type="pdf")

    # Only process the file if it's been uploaded.
    if file:
        with st.spinner("Processing document..."):
            # Read the PDF file using PdfReader.
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Split the text into manageable chunks.
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n'],
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Use a pre-trained sentence transformer model for embeddings.
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Create a vector store from the text chunks and embeddings.
            vector_store = FAISS.from_texts(chunks, embeddings)
            st.session_state.vector_store = vector_store
            st.success("Document processed and ready for questions!")

# --- Main chat interface ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Function to get the conversational response.
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the document." Do not provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """

    # Initialize the ChatGoogleGenerativeAI model.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_AI_API_KEY, temperature=0.3)

    # Create a LangChain QA chain with the specified prompt template.
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain

def speak_text(text):
    # Generate speech
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


# Process the user's question and generate a response.
def user_input(user_question):
    if 'vector_store' not in st.session_state:
        st.warning("Please upload a PDF document first.")
        return

    # Use the vector store to find the most similar documents.
    docs = st.session_state.vector_store.similarity_search(user_question)

    # Get the conversational chain and run it with the retrieved documents and user question.
    chain = get_conversational_chain()

    # Use the invoke method with the correct parameters
    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]


# Get user input from the chat box.
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = user_input(prompt)
            st.write(response)

            audio_file = speak_text(response)
            st.audio(audio_file, format="audio/mp3")


    st.session_state.messages.append({"role": "assistant", "content": response})
