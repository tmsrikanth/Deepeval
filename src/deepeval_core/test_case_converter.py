# FILE: src/deepeval/test_case_converter.py

from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden
from typing import List
from deepeval.dataset import EvaluationDataset

class TestCaseConverter:
    def __init__(self, model):
        self.model = model
        self.data_set = EvaluationDataset()

    def convert_goldens_to_test_cases(self, goldens: List[Golden]) -> List[LLMTestCase]:
        test_cases = []
        for golden in goldens:
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=self.model.generate(golden.input),
                expected_output=golden.expected_output,
                context=golden.context,
            )
            test_cases.append(test_case)
        return test_cases
    
    def add_goldens_from_csv(self, goldens_csv_path: str, input_col_name: str, expected_output_col_name: str, context_col_name: str, retrieval_context_col_name: str, additional_metadata_col_name: str, actual_output_col_name: str) -> List[LLMTestCase]:
        return self.data_set.add_goldens_from_csv_file(
            file_path=goldens_csv_path,
            input_col_name=input_col_name,
            expected_output_col_name=expected_output_col_name,
            context_col_name=context_col_name,
            retrieval_context_col_name=retrieval_context_col_name,
            additional_metadata_col_name=additional_metadata_col_name,
            actual_output_col_name=actual_output_col_name,
        )
    
    def add_actual_output_to_test_cases(self,test_cases: List[LLMTestCase]) -> List[LLMTestCase]:
        for test_case in test_cases:
            test_case.actual_output = self.model.generate(test_case.input,test_case.context)
        return test_cases
