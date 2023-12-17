import os

from src.pipeline.pipe_qa import PipelineQA
from src.constants import CONFIG_PATH

# pipeline QA config
config_pth = os.path.join(CONFIG_PATH, 'pipeline', 'qa_pipe01.json')

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

print(responses)