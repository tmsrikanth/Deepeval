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

    def save_evaluation_results_to_csv(self, eval_result, file_path):
        data = []
        for result in eval_result:
            for metric_name, metric_data in result.metrics.items():
                data.append({
                    "Input": result.input,
                    "Expected Output": result.expected_output,
                    "Actual Output": result.actual_output,
                    "Metric": metric_name,
                    "Metric Score": metric_data.score,
                    "Metric Threshold": metric_data.threshold,
                    "Metric Verdict": "Pass" if metric_data.success else "Fail"
                })
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return df