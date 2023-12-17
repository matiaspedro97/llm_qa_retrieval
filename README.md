llm_qa_retrieval
==============================

Building an LLM-based QA Retrieval including: (1) zero-shot and (2) few shot learning. Open Source foundation models will be used for demonstration purposes.

Project Organization
------------

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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
