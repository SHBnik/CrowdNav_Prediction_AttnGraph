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
        text = """
        "Serve as an advisor for a robot's collision avoidance system within a 2D environment. The environment contains multiple obstacles, each with a radius of 0.5 units. The robot and the obstacles can move within this space, and their positions are updated at each time step.

You are provided with the following information for each decision-making moment:
1. The robot's current position (X_current) and its planned next position (X_next).
2. For each obstacle, the current position (X_obstacle) and the next five predicted positions (X_next_obstacle[1-5]), representing the obstacle's trajectory.

Robot is following an action policy, Your task is to determine if next position generated by policy is safe action for the robot to take in order to avoid collisions with any obstacle. A collision is anticipated if the robot's position is within 0.5 units of an obstacle's position.
Remember that the current policy is trying to generate safe actions but is not always the case! Reason about collision according to guildines below:

Decision Guidelines:
- Advise 'Stay: False' (encouraging the robot to proceed to X_next) by default to promote continuous movement, except in the following situations:
    - If obstacle is closing to robot or staying will result in collusion, prioritize Advising 'Stay: False' for robot to move away from object.
    - If the distance between robot and closest object is higher than 2, prioritize Advising 'Stay: False' for robot to move with the policy.
    - If obstacle next positions will collide with robot but robot can run away, prioritize Advising 'Stay: False' for robot to move away from object.
    - If moving to X_next directly results in a collision with an obstacle or its immediate vicinity (within 0.5 units), advise 'Stay: True', signaling the robot to remain at X_current.
    - If the robot's surroundings, including X_next and all feasible alternative positions within a reasonable deviation from X_next, are blocked by obstacles or their safety radii, advise 'Stay: True'. This suggests the robot should stay put until a clearer path becomes available.
    - If proceeding to X_next almost certainly leads to a collision, due to the trajectories of moving obstacles converging on the robot's path, advise 'Stay: True'.

Respond with 'Stay: True' or 'Stay: False' based on these criteria, ensuring the robot's safety while minimizing the risk of collision."
Explain for your self but do not generate me any more information except {Stay: Ture} or {Stay: False}
be a little more sensetive.
        """
        response = self.client.chat.completions.create(
            # model="gpt-4",
            model="gpt-3.5-turbo",
            # response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": text},
                {"role": "user", "content": prompt},
            ],
        )
        print(response.choices[0].message.content)

        api_response = response.choices[0].message.content

        # Function to safely get the collision status from the match object
        def get_collision_status(match):
            if match:
                # Extract the boolean value and convert it to a proper boolean type
                return match.group(1).lower() == "true"
            else:
                # Handle cases where no match is found
                return None

        # Regular expression to match 'True' or 'False' in the text, accounting for different cases
        pattern = r"\b(True|False|true|false|Ture|ture)\b"

        # Search for the pattern in the response
        match = re.search(pattern, api_response, re.IGNORECASE)

        # Check if we found a match
        # Extract the boolean value and convert it to proper boolean type
        collision_status = match.group().lower() == "true"
        print(f"Collision status: {collision_status}")

        return collision_status
