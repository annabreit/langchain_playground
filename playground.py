import copy
import logging
import os
from typing import Type, Iterable, List, Optional

import pinecone
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All as lang_GPT4All
from langchain.schema import Document
from langchain.vectorstores import Pinecone, VectorStore
from langchain.vectorstores.pgvector import PGVector


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
    logging.info(f"using model located at {local_path}")

    if not os.path.exists(local_path):
        logging.info(f"downloading model to {local_path}")
        download_model(model_name, model_path)

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    return lang_GPT4All(model=local_path, callbacks=callbacks)


def create_documents(
        splitter, texts: List[str], metadatas: Optional[List[dict]] = None
) -> List[Document]:
    _metadatas = metadatas or [{}] * len(texts)
    documents = []
    for i, text in enumerate(texts):
        index = -1
        for doc in splitter.split_text(text):
            metadata = copy.deepcopy(_metadatas[i])
            metadata.update(doc.metadata)
            # if splitter._add_start_index:
            if True:
                index = text.find(doc.page_content, index + 1)
                metadata["start_index"] = index
            new_doc = Document(page_content=doc.page_content, metadata=metadata)
            documents.append(new_doc)
    return documents


def split_documents(splitter, documents: Iterable[Document]) -> List[Document]:
    texts, metadatas = [], []
    for doc in documents:
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)
    return create_documents(splitter, texts, metadatas=metadatas)


def load_docs_and_generate_chunks(docs_dir='./data',
                                  docs_ending="txt", ):
    loader = DirectoryLoader(docs_dir,
                             glob=f"**/*.{docs_ending}",
                             show_progress=True,
                             loader_cls=TextLoader)
    documents = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(
    #    chunk_size=1000,
    #    chunk_overlap=20,
    #    length_function=len,
    #    add_start_index=True,
    # )
    # return text_splitter.split_documents(documents)

    headers_to_split_on = [
        ('#', "Header 1"),
        ('##', "Header 2"),
        ('###', "Header 3")
    ]

    from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = split_documents(markdown_splitter, documents)

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(md_header_splits)


def _init_pinecone(embeddings,
                   index_name,
                   pinecone_env: str,
                   pinecone_api_key: str,
                   add_docs: bool = False,
                   docs_dir: str = './data',
                   docs_ending: str = "txt"
                   ):
    index_name = index_name

    pinecone.init(
        api_key=pinecone_api_key,
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


def _init_pgvector(embeddings,
                   collection_name: str,
                   pgvector_connection_string: str,
                   add_docs: bool = False,
                   pre_delete_collection: bool = True,
                   docs_dir: str = './data',
                   docs_ending: str = "txt", ):
    if add_docs:
        chunks = load_docs_and_generate_chunks(docs_dir=docs_dir,
                                               docs_ending=docs_ending)

        vectorstore = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=pgvector_connection_string,
            pre_delete_collection=pre_delete_collection,
        )
    else:
        vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=pgvector_connection_string,
            embedding_function=embeddings,
        )
    return vectorstore


def get_vectorstore(vector_db_cls: Type[VectorStore],
                    vector_db_params: dict,
                    embeddings_cls=HuggingFaceEmbeddings,
                    embedding_model_name="models/sentence_transformer"
                    ):
    embeddings = embeddings_cls(model_name=embedding_model_name)

    if vector_db_cls == Pinecone:
        vectorstore = _init_pinecone(embeddings, **vector_db_params)
    elif vector_db_cls == PGVector:
        vectorstore = _init_pgvector(embeddings, **vector_db_params)
    else:
        raise ValueError("supported vector stores: Pinecone, PGVector")

    return vectorstore


if __name__ == '__main__':
    logging.basicConfig(encoding='utf-8', level=logging.INFO)

    st.title('Local help')
    st.write("""
    I will have an answer to all of your questions. They might not be correct though."""
             )

    # pinecone_env = st.text_input("pinecone environment",
    #                             value=pinecone_env if pinecone_env else "")
    # pinecone_api_key = st.text_input("pinecone api key",
    #                                 type="password",
    #                                 value=pinecone_api_key if pinecone_api_key else "")

    model_name = st.text_input(
        "GPT4All model name (available models can be found here : https://gpt4all.io/models/models.json)",
        value="ggml-llama-2-7b.ggmlv3.q4_0.bin")
    query = st.text_area("Ask your Question:")

    # todo this loads the model and the db every time a request is set
    if st.button("Run"):
        # pinecone vectorstore
        load_dotenv(find_dotenv())
        pinecone_env = os.getenv("PINECONE_ENV")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_params = {
            "index_name": "langchain-demo",
            "add_docs": False,  # change here for True if your documents are not yet in the vector store
            "pinecone_env": pinecone_env,
            "pinecone_api_key": pinecone_api_key
        }
        ## pgvector vectorstore
        # pgvector_connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")
        # pgvector_params = {
        #    "collection_name": "langchain-demo",
        #    "add_docs": True,  # change here for True if your documents are not yet in the vector store
        #    "pgvector_connection_string": pgvector_connection_string
        # }
        vectorstore = get_vectorstore(
            vector_db_cls=Pinecone,
            vector_db_params=pinecone_params,
            embeddings_cls=HuggingFaceEmbeddings)  # Embeddings / Document retrieval model
        # generative model
        llm = get_llm(model_name)

        # qa chain: retrieves
        qa_chain = RetrievalQA.from_chain_type(llm,
                                               retriever=vectorstore.as_retriever(),
                                               return_source_documents=True)

        generated_text = qa_chain({"query": query})
        st.write(generated_text)
