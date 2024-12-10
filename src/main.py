# FILE: main.py

import os
from AzureCustomModel import AzureCustomModel
from deepeval.dataset import EvaluationDataset
from deepeval_core.synthesizer_handler import SynthesizerHandler
from deepeval_core.test_case_converter import TestCaseConverter
from deepeval_core.evaluator import Evaluator
from deepeval_core.metricmanger import MetricManager

def main():
    # Initialize the AzureCustomModel with the correct parameters
    deepeval_model = AzureCustomModel(
        openai_api_version="<api version>",
        azure_deployment="<deployments>",
        azure_endpoint="<your llm endpoint>",
        openai_api_key="<your api key>"
    )

    # Initialize handlers
    synthesizer_handler = SynthesizerHandler(deepeval_model)
    test_case_converter = TestCaseConverter(deepeval_model)
    
    # Define the metrics to be used
    metric_names = ["hallucination", "answer_relevancy"]
    metric_manager = MetricManager(deepeval_model)
    metrics = metric_manager.get_list_of_metrics(metric_names,threshold=0.3)
   

    # Generate synthetic goldens
    document_paths = ['C:\\Azure_python\\Deepeval\\guideToScrum.pdf'] # Example document paths
    synthetic_goldens = synthesizer_handler.generate_synthetic_goldens(document_paths)

    # Save synthetic goldens to CSV
    goldens_csv_path = 'synthetic_golden.csv'
    synthesizer_handler.save_synthetic_goldens_to_csv(goldens_csv_path)

    # Load the CSV file into an EvaluationDataset
    data_set = EvaluationDataset()
    data_set.add_goldens_from_csv_file(
        file_path=goldens_csv_path,
        input_col_name="input",
        expected_output_col_name="expected_output",
        context_col_name="context",
        retrieval_context_col_name="retrieval_context",
        additional_metadata_col_name="additional_metadata",
        actual_output_col_name="actual_output",
    )

    # Convert synthetic goldens to test cases
    test_cases_eval = test_case_converter.convert_goldens_to_test_cases(data_set.goldens)

     # Initialize the evaluator
    evaluator = Evaluator(deepeval_model, metrics)
    # Evaluate the test cases
    eval_result = evaluator.evaluate_test_cases(test_cases_eval)

    # Save the evaluation results to a CSV file
    eval_result_csv_path = 'evaluation_result.csv'
    eval_result_df = evaluator.save_evaluation_results_to_csv(eval_result, eval_result_csv_path)

    # Print the evaluation results
    print(eval_result_df)

if __name__ == "__main__":
    main()
