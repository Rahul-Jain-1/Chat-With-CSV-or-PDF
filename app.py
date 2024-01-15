import streamlit as st 
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from chat import filecsv
from pdfws import filepdf

def main():
    
    data = st.sidebar.file_uploader("Upload your Data",type=["csv","pdf"])
    if data is not None:
            # Get the file extension
        file_extension = os.path.splitext(data.name)[1].lower()
            # st.write(f"File extension: {file_extension}")
            # Check the file extension
        if file_extension == ".csv":
            filecsv(data)
        elif file_extension == ".pdf":
            filepdf(data)


if __name__=="__main__":
    main() 