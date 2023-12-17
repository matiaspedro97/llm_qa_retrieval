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


<h2>✂️📄 Chunking documents</h2>

After parsing, documents obtained were split into equally-sized chunks (with 70% overlap).

**Source**: [Split documents](src/data/split.py)


<h2>🔎📄Semantic Search</h2>

ChromaDB was used to compute semantic embeddings over each chunk.

Then, the X closest chunks to our query served as context for the LLM template prepared.

**Source**: [Embedding Search](src/features/)


<h2>✍ Text Generation</h2>

After both the query and the context are defined, the template can be completed and fed into the chosen LLM for prediction. The user can ask anything about any news he want to know about. In case the template doesn't have enough information to answer, it won't provide an hypothetical answer.

**Source**: [Causal LLMs](src/models/)


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
