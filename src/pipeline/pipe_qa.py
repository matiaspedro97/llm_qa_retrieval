import json
import pydoc
import wandb

from typing import List
from loguru import logger

from src.pipeline import PipelineGen
from src.data.parse import DocLoader


class PipelineQA(PipelineGen):
    def __init__(self, config_path: str = None, config_dict: dict = None) -> None:
        # Config path
        self.config_path = config_path

        # load modules
        if isinstance(config_dict, dict):
            config_args = self.load_modules_from_dict(config_dict)
        else:
            config_args = self.load_modules_from_json(config_path)

        # load gen attributes
        super().__init__(**config_args)

        # setup modules
        self.setup_modules()

        # add embeddings
        docs_split = self.setup_embeddings(need_embeddings=True)

        # log dataset 
        self.syncher.log_dataset(docs_split)

    def load_modules_from_json(self, json_path: str):
        config = json.load(open(json_path, 'r'))
        return self.load_modules_from_dict(config)

    def load_modules_from_dict(self, config: dict):    
        # Kwargs
        config_gen_args = {
            k: v 
            for k, v in config.items() 
            if k != 'modules'
        }

        # Loading class modules
        for module_name, module in config['modules'].items():
            class_ = pydoc.locate(f"src.{module['class_']}")
            params_ = module['params_']

            try:
                obj = class_(**params_)
                logger.info(f"Module {module_name} successfully loaded")
            except Exception as e:
                logger.info(f"Module {module_name} not loaded correctly." 
                             f"Please check the error:\n{e}")
                obj = None

            # assign to class attribute
            config_gen_args[module_name] = obj
            #exec(f"self.{module_name} = obj")
            

        return config_gen_args
    
    def setup_modules(self):
        # add template and build chain
        self.generator.setup(self.template)

    def setup_embeddings(self, need_embeddings: bool = True):
        # load documents
        doc_loader = DocLoader(docs_path='data/raw', fmt='txt')

        # split 
        docs_split = self.splitter.split_documents(doc_loader.docs)

        # compute embeddings
        self.searcher.add_embeddings(docs_split)

        return docs_split

    def run_pipe(self, queries: List[dict], n_sim_docs: int = 2):
        # questions
        questions = queries

        # most similar documents
        docs_sim = self.searcher.search_similar_documents(
            questions,
            n_docs=n_sim_docs
        )

        # context
        context = ["\n".join(docs) for docs in docs_sim['documents']]

        # response
        response = []
        for quest, cnt in zip(questions, context):
            try:
                output = self.generator.generate_response(question=quest, context=cnt)
                response.append(output)
            except Exception as e:
                logger.info(f"Error. Generation not completed. Check:\n{e}")
                response.append('No response generated')            

        return response        
        

