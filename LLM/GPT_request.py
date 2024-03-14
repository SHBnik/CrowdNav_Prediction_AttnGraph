from openai import OpenAI
import json
import LLM.My_Key as My_Key
import re


class GPT:
    def __init__(self):
        self.client = OpenAI(api_key=My_Key.key)

    def ask(self, prompt):

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "there are N humans and 1 robot in a 2D map bounded between -6 to 6. "
                    "humans move in trjectories with random goal in the map which it can collide with the robot."
                    "you will get position of each human also next 5 predicted trajectory for the humans."
                    "For the robot you get all the positions it was since the start of the session."
                    "first calculate the distnace between robot and each human and the goal."
                    " based on those distances and the trajectories, "
                    "Generate a json that contains a key named action and the value of the action key"
                    " will be the angle of the robot that you prefer to move toward to avoid collision with humans "
                    "but also reach to the goal position. robot will move toward the direction you generated with a "
                    "constatnt speed of 10. the generated angle should be a int between 0 to 359."
                    "you should keep distance of 0.75 with all humans if it is lower than 0.75 just get away from them."
                    "lower than 0.5 you lose."
                    "also justify your desicion in key justify of json that exactly which human your avoiding and how much your getting closer to goal."
                    "also print who is the closest human to your robot in closest key"
                    # "try go around humans to get to goal"
                    "you can also wait or go further form goal to prevent a collision"
                    # "go around the map to prevent any collision"
                    "double check your distance with humans."
                    # "dont forget human14"
                    "if you want to stay at a position pass -500 as action."
                    "the output should only be the json no extra leters",
                    # "generate an int value between 0 to 359 so the robot gets to the goal as fast as posible. prevent any diviation from goal."
                    # "also on the direct angle key give me what is the direct angle of the robot to the goal",
                },
                {"role": "user", "content": prompt},
            ],
        )
        # response = self.client.chat.completions.create(
        #     model="gpt-3.5-turbo-0125",
        #     response_format={"type": "json_object"},
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "there are N humans and 1 robot in a 2D map bounded between -6 to 6. "
        #             "humans move in trjectories with random goal in the map which it can collide with the robot."
        #             "you will get position of each human also next 5 predicted trajectory for the humans."
        #             "For the robot you get all the positions it was since the start of the session."
        #             "first calculate the distnace between robot and each human and the goal."
        #             " based on those distances and the trajectories, "
        #             "Generate a json that contains a key named action and the value of the action key"
        #             " will be the angle of the robot that you prefer to move toward to avoid collision with humans "
        #             "but also reach to the goal position. robot will move toward the direction you generated with a "
        #             "constatnt speed of 10. the generated angle should be a int between 0 to 359."
        #             "you should keep distance of 0.75 with all humans if it is lower than 0.75 just get away from them."
        #             "lower than 0.5 you lose."
        #             "also justify your desicion in key justify of json that exactly which human your avoiding and how much your getting closer to goal."
        #             "also print who is the closest human to your robot in closest key"
        #             "try go around humans to get to goal"
        #             "you can also wait or go further form goal to prevent a collision"
        #             "go around the map to prevent any collision"
        #             "double check your distance with humans."
        #             # "dont forget human14"
        #             "if you want to stay at a position pass -500 as action.",
        #             # "generate an int value between 0 to 359 so the robot gets to the goal as fast as posible. prevent any diviation from goal."
        #             # "also on the direct angle key give me what is the direct angle of the robot to the goal",
        #         },
        #         {"role": "user", "content": prompt},
        #     ],
        # )
        try:
            print(response.choices[0].message.content)
            # Find the index of the first opening brace and the last closing brace
            start_index = response.choices[0].message.content.find("{")
            end_index = response.choices[0].message.content.rfind("}")

            # Extract the substring that is likely the JSON object
            json_str = response.choices[0].message.content[start_index : end_index + 1]
            # matches = re.findall(r"\{\n(.*?)\n\}", response.choices[0].message.content)
            print(json_str)
            parsed_json = json.loads(json_str)
            print(parsed_json)
            action_value = int(parsed_json["action"])
        except:
            action_value = -500
        return action_value
