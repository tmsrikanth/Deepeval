# FILE: src/deepeval/synthesizer_handler.py

import pandas as pd
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

class SynthesizerHandler:
    def __init__(self, model):
        self.synthesizer = Synthesizer()
        self.model = model

    def generate_synthetic_goldens(self, document_paths):
        self.synthesizer.generate_goldens_from_docs(
            document_paths=document_paths,
            context_construction_config=ContextConstructionConfig(critic_model=self.model,chunk_size=300)
        )
        return self.synthesizer.synthetic_goldens

    def save_synthetic_goldens_to_csv(self, file_path):
        goldens_dataframe = self.synthesizer.to_pandas()
        goldens_dataframe.to_csv(file_path, index=False)
        return goldens_dataframe