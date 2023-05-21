import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## How to use this app
    1. Enter your OpenAI API key below
    2. Upload a pdf, docx, or txt file
    3. Ask a question about your document
    ''')

    api_key = st.text_input('Enter your API key', type='password')

    st.markdown('''
    ## FAQ

    **1. How does this app work?** 

    When you upload a document, it will be divided into smaller chunks and stored in a special type of database called a vector index that allows for semantic search and retrieval.

    When you ask a question, the app uses an AI model called **pdfGPT** to search through the document chunks and find the most relevant ones using the vector index. Then, it uses another AI model, **GPT3**, to generate a final answer.
    
    **1. Is my data safe?** 

    Yes, your data is safe. pdfGPT does not store your documents or questions. All uploaded data is deleted after you close the browser tab.
    ''')

    add_vertical_space(1)
    st.write('Made with ❤️ by Urim Aliu')
    # Ask the user to enter their API key

def main():
    st.header("Chat with your PDFs 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        if not api_key:  # if the user has not entered an API key
            st.error("API key is required to proceed. Get your OpenAI API key here: https://platform.openai.com/account/api-keys.")
            return

        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(api_key=api_key)  # use the provided API key
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(api_key=api_key)  # use the provided API key
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
