from openai import OpenAI
import LLM.My_Key as My_Key


class GPT:
    def __init__(self):
        self.client = OpenAI(api_key=My_Key.key)

    def ask(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "there are 20 humans and 1 robot in a 2D map. "
                               "humans move in random trjectories in the map which it can collide with the robot."
                               "you will get state of each human and the robot in this format "
                               "[position x, positon y, velocity x,velocity y,theta] also next 5 predicted trajectory"
                               " in a 5x1 list format of [position x, position y] and you should generate"
                               " likelihood of collision for each human with the robot. output should be json"
                               " with format of id number of each human is the key and the value of that key is the likelihood"
                               " of colision of that human with the robot in percentage.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        print(response.choices[0].message.content)
