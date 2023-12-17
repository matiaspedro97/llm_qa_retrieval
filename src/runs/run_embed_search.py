import os
import glob

from src.models.llm import LLMOpenAIChain
from src.data.split import DocSplitter
from src.features.embeddings import ChromaDBSearcher, OpenAIEncoder

query = 'O que se passa na guerra entre Palestina e Israel?'

docs_paths = glob.glob('data/raw/*.txt')

docs = []
for p in docs_paths:
    with open(p, 'r', encoding="ISO-8859-1") as f:
        lines = "".join(f.readlines())
        docs.append(lines)


split = DocSplitter(chunk_size=500, chunk_overlap=80)

docs_split = split.split_documents(docs)

searcher = ChromaDBSearcher()

searcher.add_embeddings(docs_split)

docs_sim = searcher.search_similar_documents(query, n_docs=5)





