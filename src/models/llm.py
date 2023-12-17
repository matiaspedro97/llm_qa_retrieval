from loguru import logger

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.chains import LLMChain


class PTGenerator:
    def __init__(self, prompt_template: str = None, tags: list = ["context", "question"]) -> None:
        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=tags
        )
        
    def get_full_prompt(self, **kwargs):
        return self.PROMPT.format(**kwargs)


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