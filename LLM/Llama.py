from openai import OpenAI

class Llama:
    def __init__(self):
        # Define your API key and base URL here
        self.api_key = "LL-UDjso5HXDSwlkYlnYPOsLMPZLHiy1ZzaziebWauVwDGKb2tb0htu8U2CWbgP3ORE"
        self.base_url = "https://api.llama-api.com"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def ask(self, prompt):
        response = self.client.chat.completions.create(
            model="llama-13b-chat",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "There are n humans and 1 robot in a 2d map"},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content