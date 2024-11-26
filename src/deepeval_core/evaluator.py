# FILE: src/deepeval/evaluator.py

import pandas as pd
from deepeval import evaluate

class Evaluator:
    def __init__(self, model, metrics:list,threshold=0.3):
        self.model =model
        self.metrics = metrics
      

    def evaluate_test_cases(self, test_cases):
        eval_result = evaluate(test_cases, self.metrics, write_cache=True, print_results=True)
        return eval_result

    def save_evaluation_results_to_csv(self, evaluation_results, file_path):
        data = []
        for test_result in evaluation_results.test_results:
            for metric_data in test_result.metrics_data:
                data.append({
                    "Test Case Name": test_result.name,
                    "Input": test_result.input,
                    "Actual Output": test_result.actual_output,
                    "Expected Output": test_result.expected_output,
                    "Context": test_result.context,
                    "Metric Name": metric_data.name,
                    "Metric Score": metric_data.score,
                    "Metric Threshold": metric_data.threshold,
                    "Metric reason": metric_data.reason,
                    "Metric Verdict": "Pass" if metric_data.success else "Fail"
                })
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return df