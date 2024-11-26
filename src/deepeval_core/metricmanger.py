from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics  import HallucinationMetric


class MetricManager:
    """
    A class to manage and retrieve supported metrics from the DeepEval framework.
    """
    def __init__(self, model):
        """
        Initialize the MetricManager with a model instance.
        
        Args:
            model: The model instance (e.g., GPTModel) to be used with metrics.
        """
        self.model = model
        self.supported_metrics = {
            "answer_relevancy": AnswerRelevancyMetric,
            "hallucination": HallucinationMetric,
          
        }

    def get_metric(self, metric_name, **kwargs):
        """
        Retrieve a specific metric instance.
        
        Args:
            metric_name (str): Name of the metric to retrieve (e.g., 'answer_relevancy').
            kwargs: Additional parameters required for metric initialization.
        
        Returns:
            Metric instance if the metric is supported.
        
        Raises:
            ValueError: If the requested metric is not supported.
        """
        if metric_name not in self.supported_metrics:
            raise ValueError(f"Metric '{metric_name}' is not supported. Supported metrics: {list(self.supported_metrics.keys())}")
        
        # Initialize the metric with the provided model and additional arguments
        return self.supported_metrics[metric_name](model=self.model, **kwargs)
    
    def get_list_of_metrics(self, metric_names, **kwargs):
        """
        Retrieve a list of metric instances.
        
        Args:
            metric_names (list): Names of the metrics to retrieve (e.g., ['answer_relevancy', 'coherence']).
            kwargs: Additional parameters required for metric initialization.
        
        Returns:
            list: List of metric instances if the metrics are supported.
        
        Raises:
            ValueError: If any of the requested metrics are not supported.
        """
        return [self.get_metric(metric_name, **kwargs) for metric_name in metric_names]

    def list_supported_metrics(self):
        """
        List all supported metrics.
        
        Returns:
            list: Names of all supported metrics.
        """
        return list(self.supported_metrics.keys())
