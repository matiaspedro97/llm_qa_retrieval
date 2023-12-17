import wandb

class PipelineGen:
    def __init__(
            self,
            project_name: str,
            run_name: str,
            run_description: str,

            syncher,
            searcher,
            splitter,
            template,
            generator,
            **kwargs
    ) -> None:
        
        # Run details
        self.project_name = project_name
        self.run_name = run_name
        self.run_description = run_description

        # Modules
        self.syncher = syncher
        self.searcher = searcher
        self.splitter = splitter
        self.template = template
        self.generator = generator

    def finish_run(self, **kwargs):
        wandb.finish()
