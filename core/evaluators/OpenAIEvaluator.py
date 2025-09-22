from core.evaluators.BaseEvaluator import BaseEvaluator
from openai import OpenAI
import json

class OpenAIEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs
        self.client = OpenAI(
            api_key = self.config.get("api_key", "")
        )
        self.model = self.config.get("model", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0.7)
        self.answer_system_prompt = self.config.get(
            "answer_system_prompt",
            ""
        )
        self.answer_prompt_template = self.config.get(
            "answer_prompt_template",
            "Is the answer to the question based on the context? Give me the answer in only a JSON, without ```json fences:\nContext: {retrieved_docs}\nQuery: {question}\nAnswer: {generated_answer}"
        )

    def evaluate_answer(self, results):
        prompt = self.answer_prompt_template.format(
            question=results["query"],
            retrieved_docs=results["retrieved_texts"],
            generated_answer=results["response"]["answer"]
        )
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {
                    "role": "system",
                    "content": self.answer_system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature = self.temperature,
        ).choices[0].message.content
        return response