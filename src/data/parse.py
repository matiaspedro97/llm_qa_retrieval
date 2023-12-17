import os
import newspaper
import json
import tqdm
import glob
import time

import pandas as pd

from loguru import logger

from src.config import data_raw_path


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


class DocLoader:
    def __init__(self, docs_path: str = 'data/raw', fmt='txt') -> None:
        self.docs_paths = glob.glob(os.path.join(docs_path, f'*{fmt}'))

        if fmt == 'txt':
            self.docs = self.merge_txt()
        elif fmt == 'csv':
            self.docs = self.merge_csv()
    
    def merge_csv(self):
        docs = "".join(["\n".join(pd.read_csv(path).to_list()) for path in self.docs_paths])
        return docs

    def merge_txt(self):
        docs = []
        for p in self.docs_paths:
            with open(p, 'r', encoding="latin-1") as f:
                lines = "".join(f.readlines())
                docs.append(lines)
        return docs
