from core.generators.BaseGenerator import BaseGenerator
from openai import OpenAI

class OpenAIGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs
        self.client = OpenAI(
            api_key = self.config.get("api_key", "")
        )
        self.model = self.config.get("model", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0.7)
        self.system_prompt = self.config.get(
            "system_prompt",
            ""
        )
        self.prompt_template = self.config.get(
            "prompt_template",
            "Respond to the query based on the context:\nContext: {context}\nQuery: {query}\nResponse:"
        )

    def generate(self, query, context) -> str:
        prompt = self.prompt_template.format(
            context=context,
            query=query
        )
        response = self.client.chat.completions.create(  
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt}, # These system messages prepare the agent to tailor responses
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature # This generates much better results
        )

        answer = response.choices[0].message.content
        return answer