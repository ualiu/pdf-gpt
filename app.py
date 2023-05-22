import streamlit as st
from dotenv import load_dotenv
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
        add_vertical_space(1)
        st.write('Created by Urim Aliu')
        st.markdown('''
        ## How to use this app
        1. Upload a pdf, docx, or txt file 📄
        2. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) 🔑
        3. Using natural language, ask a question about your document 💬
        
        ## FAQ


        **1. How does this app work?**


        When you upload a document, it will be divided into smaller chunks and stored in a special type of database.


        When you ask a question using natural lanauge, the app uses an AI model called **pdfGPT** to search through the document chunks and find the most relevant information. Then, it uses another AI model, **GPT3**, to generate a final answer.
       
        **1. Is my data safe?**


        Yes, your data is safe. **pdfGPT** does not store your documents or questions. All uploaded data is deleted after you close the browser tab.
        ''')

load_dotenv()
 
def main():
    st.header("Chat with your PDFs 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        # Ask the user for the API key
        api_key = st.text_input('Enter your API key (sk-...).', type='password')

        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        
        if api_key:
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
    
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            query = st.text_input("Ask questions about your PDF file:")
    
            if query:
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.write(response)
        else:
            st.write("Please enter your API key to use the app.\n You can get your API key from https://platform.openai.com/account/api-keys.")
 
if __name__ == '__main__':
    main()
