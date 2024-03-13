from openai import OpenAI
import LLM.My_Key as My_Key

class GPT:
    def __init__(self):
        self.client = OpenAI(api_key=My_Key.key)

    def ask(self, prompt):
        response = self.client.chat.completions.create(
            model="text-davinci-002",  # Specify LLMA model here
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "There are N humans and 1 robot in a 2D map... (Your prompt)",
                },
                {"role": "user", "content": prompt},
            ],
        )
        print(response.choices[0].message.content)
