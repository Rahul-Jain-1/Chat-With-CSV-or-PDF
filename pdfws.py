import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import PyPDF2

load_dotenv()
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = ChatOpenAI()
    return llm

def pdf_to_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def filepdf(data):
    st.title("Chat with PDF using OpenAi ")  
    uploaded_file =data  
    st.write("Uploaded PDF:", uploaded_file.name)

    # Reset chat history when a new file is uploaded
    st.session_state['history'] = []
    st.session_state['generated'] = ["Hello ! Ask me anything about  🤗"]
    st.session_state['past'] = ["Hey ! 👋"]

    if uploaded_file:
        # Use tempfile because PDFLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Extract text from PDF
        text = pdf_to_text([tmp_file_path])

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        db.save_local(DB_FAISS_PATH)
        
        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        # Container for the chat history
        response_container = st.container()
        # Container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your PDF data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
