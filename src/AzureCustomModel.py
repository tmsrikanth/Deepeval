import os
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM


class AzureCustomModel(DeepEvalBaseLLM):
    def __init__(self, openai_api_version, azure_deployment, azure_endpoint, openai_api_key):
        self.model = AzureChatOpenAI(
            openai_api_version=openai_api_version,
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            openai_api_key=openai_api_key,
        )

    def load_model(self):
        return self.model

    def generate(self, input):
        chat_model = self.load_model()
        return chat_model.invoke(input).content

    async def a_generate(self, input):
        chat_model = self.load_model()
        response = await chat_model.ainvoke(input)
        return response.content

    def get_model_name(self):
        return "Custom Azure Model for DeepEval"
