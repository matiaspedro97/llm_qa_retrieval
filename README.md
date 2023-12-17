llm_qa_retrieval
==============================

Building an LLM-based QA Retrieval including: (1) zero-shot and (2) few shot learning. Open Source foundation models will be used for demonstration purposes.

<h2>📌 Project Structure</h2>

```
root
|-- config
|   |-- pipeline
|   |   |-- qa_pipe01.json
|   |   |-- qa_pipe02.json
|-- data
|   |-- external
|   |   |-- .gitkeep
|   |-- interim
|   |   |-- .gitkeep
|   |-- processed
|   |   |-- .gitkeep
|   |-- raw
|       |-- abola.txt
|       |-- bbc.txt
|       |-- ...
|       |-- sicnoticias.txt
|-- docs
|   |-- commands.rst
|   |-- conf.py
|   |-- getting-started.rst
|   |-- index.rst
|       |-- make.bat
|       |-- Makefile
|-- models
|   |-- .gitkeep
|-- notebooks
|   |-- .gitkeep
|-- references
|   |-- .gitkeep
|-- reports
|   |-- figures
|       |-- .gitkeep
|-- requirements
|   |-- requirements-dev.txt
|   |-- requirements-prod.txt
|-- src
|   |-- config.py
|   |-- constants.py
|   |-- __init__.py
|   |-- data
|   |   |-- parse.py
|   |   |-- split.py
|   |   `-- __init__.py
|   |-- features
|   |   |-- embeddings.py
|   |   `-- __init__.py
|   |-- models
|   |   |-- llm.py
|   |   `-- __init__.py
|   |-- pipeline
|   |   |-- pipe_qa.py
|   |   |-- __init__.py
|   |-- runs
|   |   |-- run_embed_search.py
|   |   |-- run_experiment_qa.py
|   |   |-- run_fetch_news.py
|   |   |-- run_qa_news.py
|   |-- sync
|   |   |-- wandb.py
|   |   |-- __init__.py
|   |-- visualization
|   |   |-- visualize.py
|   |   |-- __init__.py
|
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```
---

<h2>⚒️ 📰 Fetching News</h2>

World news were fetched using Newspaper3k python library, who parsed the latest news from multiple news websites given in advance.

**Source**: [Fetch documents](src/data/parse.py)

```python
# Fetch the latest news from websites given as argument
class NewsFetcher:
    def __init__(self, websites: list = []) -> None:
        self.websites = websites


        self.names = [w.split('.')[-2].split('//')[-1] for w in self.websites]


    @classmethod
    def fetch_website_news(self, website: str, max_news: int = 30):
        web_parse = newspaper.build(website)


        web_text = []


        for article in tqdm.tqdm(web_parse.articles[:max_news], desc="Fetch articles"):
            try:
                # parse web article
                article.download()
                article.parse()
                article.nlp()


                # map important field
                article = {
                    "title": str(article.title),
                    "text": str(article.text),
                    "published_date": str(article.publish_date),
                    "keywords": article.keywords,
                }


                # dict to text
                text = "\n\n".join([f"** {k} **\n{it}" for k, it in article.items()])


                # assign
                web_text.append(text)
            except Exception as e:
                logger.debug(f"Error:\n{e}")
            
            time.sleep(5)
        
        web_text = "\n\n".join(web_text)
        return web_text
    
    def fetch_all(self, max_news: int = 30):
        for name, url in zip(self.names, self.websites):
            # fetch news
            web_text = self.fetch_website_news(url, max_news)


            # save to txt
            self.save_website_to_txt(web_text, name, dir_to_save=None)
        return None
    
    @classmethod
    def save_website_to_txt(self, text: str, source: str, dir_to_save: str = None):
        # get path to save
        dir_ = dir_to_save if isinstance(dir_to_save, str) else data_raw_path
        path_ = os.path.join(dir_, f"{source}.txt")


        # save to file
        try:
            with open(path_, "a", encoding="utf-8") as f:
                text = text.replace("\u201c", "").\
                    replace("\u201d", "").\
                        replace("\u2019", "").\
                            replace("\u2018", "")
                f.write(text)
                f.close()
        except Exception as e:
            logger.info(f"Error when writing into the file:\n{e}")

```

<h2>✂️📄 Chunking documents</h2>

After parsing, documents obtained were split into equally-sized chunks (with 70% overlap).

**Source**: [Split documents](src/data/split.py)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Split the documents into chunks of equal size. Overlap is used to prevent the loss of information
class DocSplitter:
    def __init__(
            self,
            chunk_size: int = 100, 
            chunk_overlap: int = 20, 
            length_function: callable = len, 
            add_start_index: bool = True
    ) -> None:
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=length_function, 
            add_start_index=add_start_index
        )


    def split_documents(self, documents: list):
        # with page content
        texts = self.text_splitter.create_documents(documents)
        
        # string-only
        texts_ = [txt.page_content for txt in texts]
        return texts_

```


<h2>🔎📄Semantic Search</h2>

ChromaDB was used to compute semantic embeddings over each chunk.

Then, the X closest chunks to our query served as a context for the LLM template prepared.

**Source**: [Embedding Search](src/features/)


```python
import time
import tqdm


import chromadb
from chromadb.config import Settings


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


# ChromaDB searcher is meant to compute and store the embeddings 
# that the provided query will be compared with.
# This will result in the best possible context to our query to be corretly answered.
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
```

<h2>✍ Text Generation</h2>

After the query and the context are defined, the template can be completed and fed into the chosen LLM for prediction. The user can ask anything about any news he want to know about. In case the template doesn't have enough information to answer, it won't provide an hypothetical answer.

Source: [Causal LLMs](src/models/)


```python
from loguru import logger


from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.chains import LLMChain


# It generates the template that will serve as a basis for the QA system.
class PTGenerator:
    def __init__(self, prompt_template: str = None, tags: list = ["context", "question"]) -> None:
        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=tags
        )
        
    def get_full_prompt(self, **kwargs):
        return self.PROMPT.format(**kwargs)


# LLM Chain based on OpenAI API calls
class LLMOpenAIChain:
    def __init__(self, model_name: str = 'text-davinci-003', **kwargs) -> None:


        self.llm = OpenAI(model_name=model_name)


        self.chain = LLMChain(prompt=self.template.prompt, llm=self.llm)


        self.template = None
        self.chain = None


        self.setup_done = False


    def setup(self, template: PTGenerator):
        self.template = template
        self.chain = LLMChain(prompt=self.template.prompt, llm=self.llm)
        self.setup_done = True


    def generate_response_from_llm(self, **kwargs):
        if self.setup_done:
            prompt = self.template.get_full_prompt(**kwargs)


            response = self.llm.predict(prompt)
        else:
            response = None


        return response
    
    def generate_response(self, **kwargs):
        if self.setup_done:
            response = self.chain.run(**kwargs)
        else:
            response = None


        return response


# LLM Chain based on HuggingFaceHub API calls
class LLMHFHubChain:
    def __init__(self, model_name: str = 'tiiuae/falcon-7b', **kwargs) -> None:


        self.llm = HuggingFaceHub(
            repo_id=model_name,
            task='text-generation',
            model_kwargs=kwargs
        )


        self.template = None
        self.chain = None


        self.setup_done = False


    def setup(self, template: PTGenerator):
        self.template = template
        self.chain = LLMChain(prompt=self.template.prompt, llm=self.llm)
        self.setup_done = True


    def generate_response_from_llm(self, **kwargs):
        if self.setup_done:
            prompt = self.template.get_full_prompt(**kwargs)


            response = self.llm.predict(prompt)
        else:
            response = None


        return response
    
    def generate_response(self, **kwargs):
        if self.setup_done:
            response = self.chain.run(**kwargs)
        else:
            logger.info("Chain Setup not performed yet. Please run setup() method first.")
            response = None


        return response
    
# LLM Chain based on calls to models stored locally
class LLMHFLocalChain:
    def __init__(self, model_name: str = 'tiiuae/falcon-7b', device: int = -1, batch_size: int = 2, **kwargs) -> None:


        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            device=device, 
            batch_size=batch_size,
            task='text-generation',
            model_kwargs=kwargs
        )


        self.template = None


        self.chain = None


    def setup(self, template: PTGenerator):
        self.template = template
        self.chain = LLMChain(prompt=template.prompt, llm=self.llm)


    def generate_response_from_llm(self, **kwargs):
        prompt = self.template.get_full_prompt(**kwargs)


        response = self.llm.predict(prompt)


        return response
    
    def generate_response(self, **kwargs):


        response = self.chain.run(**kwargs)


        return response

```

------

The template for this task can be checked down below:

```python
"""
Use the following pieces of context to answer the question at the end. 
Answer strictly to the question asked and only based on the context given, not giving to much background information. 
If you don't know the answer, just say that you don't know, don't try to make up an answer when you don't have information 
to do so. Your answer will be evaluated based on its (1) truthfulness, (2) conciseness, and (3) realism.\n


{context}\n


Question: {question}\n


Helpful Answer:
"""
```

**Source**: [Configs](config/pipeline/)

------

P.S.: Content may be in either English/Portuguese, since the websites provided contain information in such idioms.

------

<h2>🕹️ Example Run</h2>

This is an example of how to run a QA experiment from (1) **fetching** website news to (2) **answering** to the questions asked.

Source: [Runs](src/runs/)

```python
from src.data.parse import NewsFetcher


fetcher = NewsFetcher(
    websites=[
        "https://www.jn.pt/ultimas/",
        "https://www.publico.pt/",
        "https://expresso.pt/",
        "https://observador.pt/", 
        "https://www.dn.pt/",
        "https://sicnoticias.pt/",
        "https://www.rtp.pt/noticias/",
        "https://www.sapo.pt/",
        "https://www.abola.pt/",
        "https://www.bbc.com/news/world",
    ]
)

fetcher.fetch_all(max_news=80)

# instantiate QA pipeline loader
pipe_qa = PipelineQA(config_path=config_pth)

# queries
questions = [
    'Qual a opinião do partido Bloco de Esquerda acerca do novo aeroporto de Lisboa?', 
    'O que é o coro musical de Aveiro?', 
    'What was the result in the match between Frankfurt and Bayern Munich?',
    'Qual foi o resultado do jogo do Sporting frente ao Vitória de Guimaraes?',
    'Qual foi o resultado do jogo de hoje do Porto frente ao Casa Pia?',
    'Em que ano é que as duas gémeas brasileiras receberam o medicamento mais caro do mundo?'
]

# responses
responses = pipe_qa.run_pipe(queries=questions, n_sim_docs=4)

```




<h2>📣 Report</h2>

This report was generated using **Weights & Biases** API, which linked LLM zero-shot interaction on specific Question-Answering runs. Please check it out: [QA-Report](https://wandb.ai/matiaspedro97/LLM_QA/reports/LLM-QA-Report--Vmlldzo2MjE4NjM3?accessToken=nm9vvqj9u1zrft6560sullxyqmbr8zwaf1np7o81c3aiafgb5srtnxkx3heb41jq)


--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
