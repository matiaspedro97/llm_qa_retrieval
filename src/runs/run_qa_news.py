import os
import glob

from src.models.llm import LLMOpenAIChain
from src.data.split import DocSplitter
from src.features.embeddings import ChromaDBSearcher, OpenAIEncoder
from src.data.parse import DocLoader
from src.models.llm import LLMHFHubChain, PTGenerator


# LOAD DOCUMENTS
doc_loader = DocLoader(docs_path='data/raw', fmt='txt')

# SPLIT AND SEARCH DOCUMENTS
searcher = ChromaDBSearcher(name='qa_news')

split = DocSplitter(chunk_size=500, chunk_overlap=80)

docs_split = split.split_documents(doc_loader.docs)

# COMPUTE EMBEDDINGS
searcher.add_embeddings(docs_split)

# PROMPT TEMPLATE
template = "Use the following pieces of context to answer the question at the end." \
           "If you don't know the answer, just say that you don't know, don't try to"\
           "make up an answer when you don't have information to do so.\n{context}\nQuestion: "\
           "{question}\nHelpful Answer:"
tags = ['context', 'question']

prompt_template = PTGenerator(prompt_template=template, tags=tags)

# LLM CHAIN
chain = LLMHFHubChain(model_name="internlm/internlm-chat-7b", model_kwargs={"temperature": 0.8, "max_length": 128})
chain.setup(prompt_template)

# GET SIMILAR CONTEXT
question=['Notícias acerca do futebol do Benfica?', 'O que se passa entre Israel e Palestina?']

docs_sim = searcher.search_similar_documents(question, n_docs=2)

context=["\n".join(docs) for docs in docs_sim['documents']]

response = chain.generate_response(question=question, context=context)

print(response)

