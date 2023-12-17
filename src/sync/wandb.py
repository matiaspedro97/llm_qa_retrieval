import wandb

from typing import List

from langchain.docstore.document import Document


class WBSyncher:
    def __init__(self, project_name: str, run_name: str, config_args: dict) -> None:
        self.run = wandb.init(project=project_name, name=run_name, config=config_args)

    def log_dataset(self, documents: List[Document]):
        """Log a dataset to wandb

        Args:
            documents (List[Document]): A list of documents to log to a wandb artifact
            run (wandb.run): The wandb run to log the artifact to.
        """
        document_artifact = wandb.Artifact(name="documentation_dataset", type="dataset")
        with document_artifact.new_file("documents.json", mode='w', encoding='latin-1') as f:
            for idx, document in enumerate(documents):
                f.write(document)

        self.run.log_artifact(document_artifact)

    