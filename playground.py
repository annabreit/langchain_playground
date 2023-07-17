import logging
import os

import pinecone
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All as lang_GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone


def download_model(model_name: str, target_folder: str):
    import requests

    from pathlib import Path
    from tqdm import tqdm

    target_path = Path(target_folder) / model_name
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    url = 'http://gpt4all.io/models/' + model_name

    response = requests.get(url, stream=True)

    with open(target_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)


def get_llm(model_name, model_path="./models"):

    local_path = (model_path + "/" + model_name)

    if not os.path.exists(local_path):
        download_model(model_name, model_path)

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    return lang_GPT4All(model=local_path, callbacks=callbacks)


def load_docs_and_generate_chunks(docs_dir='./data',
                                  docs_ending="txt", ):
    loader = DirectoryLoader(docs_dir,
                             glob=f"**/*.{docs_ending}",
                             show_progress=True,
                             loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def get_vectorstore(index_name="langchain-demo",
                    embeddings_cls=HuggingFaceEmbeddings,
                    add_docs=False,
                    docs_dir='./data',
                    docs_ending="txt",
                    pinecone_env=None,
                    pinecone_key=None
                    ):

    embeddings = embeddings_cls()

    pinecone.init(
        api_key=pinecone_key,
        environment=pinecone_env
    )

    if not index_name in pinecone.list_indexes():
        embed_dim = len(embeddings.embed_documents(["Test"])[0])
        pinecone.create_index(index_name,
                              dimension=embed_dim,
                              metric="cosine",
                              pods=1,
                              pod_type="p1.x1")
        chunks = load_docs_and_generate_chunks(docs_dir=docs_dir,
                                               docs_ending=docs_ending)
        if add_docs:
            vectorstore = Pinecone.from_documents(chunks,
                                                  embeddings,
                                                  index_name=index_name)
        else:
            index = pinecone.Index(index_name)
            vectorstore = Pinecone(index, embeddings.embed_query, "text")
    else:
        index = pinecone.Index(index_name)
        vectorstore = Pinecone(index, embeddings.embed_query, "text")

    return vectorstore


if __name__ == '__main__':
    logging.basicConfig(encoding='utf-8', level=logging.INFO)



    st.title('PP help')
    st.write("""
    I will have an answer to all of your questions. They might not be correct though."""
             )

    load_dotenv(find_dotenv())
    pinecone_env = os.getenv("PINECONE_ENV")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    pinecone_env = st.text_input("pinecone environment",
                                 value=pinecone_env if pinecone_env else "")
    pinecone_api_key = st.text_input("pinecone api key",
                                     type="password",
                                     value=pinecone_api_key if pinecone_api_key else "")

    model_name = st.text_input("GPT4All model name (available models can be found here : https://gpt4all.io/models/models.json)",
                              value="ggml-model-gpt4all-falcon-q4_0.bin")
    query = st.text_area("Ask your PoolParty Question:")

    #todo this loads the model and the db every time a request is set
    if st.button("Run"):
        # pinecone vectorstore
        vectorstore = get_vectorstore(index_name="langchain-demo",
                                      embeddings_cls=HuggingFaceEmbeddings,  # Embeddings / Document retrieval model
                                      add_docs=False,
                                      # change here for True if your documents are not yet in the vector store
                                      pinecone_env=pinecone_env,
                                      pinecone_key=pinecone_api_key)
        # generative model
        llm = get_llm("ggml-model-gpt4all-falcon-q4_0.bin")

        # qa chain: retrieves
        qa_chain = RetrievalQA.from_chain_type(llm,
                                               retriever=vectorstore.as_retriever(),
                                               return_source_documents=True)

        generated_text = qa_chain({"query": query})
        st.write(generated_text)
