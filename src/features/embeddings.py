import time
import tqdm

import chromadb
from chromadb.config import Settings

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class ChromaDBSearcher:
    def __init__(self, name: str = 'doc_collection') -> None:
        # get chromadb client
        self.client = chromadb.PersistentClient(
            '.chromadb/'
        )

        # get or create collection
        self.collection = self.client.get_or_create_collection(
            name=name
        )

    def add_embeddings(self, docs: list, emb: list = None):
        tags = [f"doc_{idx}" for idx in range(len(docs))]

        # in case embeddings are not provided, chromadb will handle encoding
        if emb is not None:
            self.collection.add(
                embeddings=emb,
                documents=docs,
                ids=tags
            )

        else: 
            self.collection.add(
                documents=docs,
                ids=tags
            )

    def search_similar_documents(self, query: list, n_docs: int = 3):
        results = self.collection.query(
            query_texts=query,
            n_results=n_docs
        )
        return results
    
    @classmethod
    def merge_documents(self, docs: list, join_tag: str = '\n\n'):
        return join_tag.join(docs)


class OpenAIEncoder:
    def __init__(self, model_name: str = 'text-embedding-ada-002') -> None:
        self.emb = OpenAIEmbeddings(model=model_name, show_progress_bar=True, max_retries=1, chunk_size=1)
    
    def get_embeddings(self, documents: list, async_emb: bool = True, **kwargs):
        if async_emb:
            embed = self.emb.aembed_documents(
                texts=documents,
                **kwargs
            )
        else:
            embed = self.emb.embed_documents(
                texts=documents,
                **kwargs
            )
        return embed
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embeddings_with_backoff(self, documents: list, async_emb: bool = True, sleep_time: int = 5, **kwargs):
        embeds = []
        for doc in tqdm.tqdm(documents, desc='Documents'):
            aux_embed = self.get_embeddings([doc], async_emb)
            embeds += aux_embed

            time.sleep(sleep_time)
    