from openai import OpenAI
import LLM.My_Key as My_Key
import json
import re


# Function to find and return the collision status from various keys
def get_collision_status(data, keys):
    for key in keys:
        # Check if the key exists in the data, considering case-insensitivity
        for actual_key in data:
            if key.lower() == actual_key.lower():
                return data[actual_key]
    # Return None if none of the keys are found
    return None


class GPT:
    def __init__(self):
        self.client = OpenAI(api_key=My_Key.key)

    def ask(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4",
            # response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "There is a [-6,6] * [-6,6] map where there are N obstacles and 1 robot in the 2D map."
                               "obstacles are blocking robots way and they can collide with the robot."
                               "At each step, obstacles position change a bit."
                               "you will get the x and y position of each obstacles and the positions of robot in this format"
                               "[position x, position y]"
                               "Our control method is trying to navigate the robot throughout the obstacles to a goal without colliding. \
                                Your task is to provide an alarm True or False whether you think robot will collide in any of the next 10 steps \
                                Use a keyword { Collide : True/False } to answer. No added information is needed."
                               # " likelihood of collision for each human with the robot. output should be json"
                               # " with format of id number of each human is the key and the value of that key is the likelihood"
                               # " of colision of that human with the robot in percentage.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        print(response.choices[0].message.content)

        api_response = response.choices[0].message.content

        # Function to safely get the collision status from the match object
        def get_collision_status(match):
            if match:
                # Extract the boolean value and convert it to a proper boolean type
                return match.group(1).lower() == 'true'
            else:
                # Handle cases where no match is found
                return None

        # Regular expression to match 'True' or 'False' in the text, accounting for different cases
        pattern = r'\b(True|False|true|false)\b'

        # Search for the pattern in the response
        match = re.search(pattern, api_response, re.IGNORECASE)

        # Check if we found a match
        # Extract the boolean value and convert it to proper boolean type
        collision_status = match.group().lower() == 'true'
        print(f"Collision status: {collision_status}")

        return collision_status

