from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore

from config import PINECONE_API_KEY, PINECONE_INDEX

from ingestion.embedder import get_embedding_model


def upload(chunks, reset=False):

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index = pc.Index(PINECONE_INDEX)

    if reset:
        # remove vectors from older data / older chunking so they cannot be retrieved
        index.delete(delete_all=True)
        print("Pinecone index cleared")

    embeddings = get_embedding_model()

    PineconeVectorStore.from_documents(

        chunks,

        embedding=embeddings,

        index_name=PINECONE_INDEX

    )

    print("Uploaded Successfully")
