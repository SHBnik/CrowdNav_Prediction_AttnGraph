import math
import re


class PromptGen:
    @staticmethod
    def make_prompt(
        hvisible, hstate, prev_hstate, rstate, prev_rstates, hpred_traj, rgoal
    ):
        prompt = f"In this scene there are {len(hvisible)} humans and a robot.\n"
        for index, i in enumerate(hvisible):
            prompt += f"Human {i}, last position( px={prev_hstate[i][0]} and py={prev_hstate[i][1]}), current position( px={hstate[i][0]} and py={hstate[i][1]})."
            prompt += f"the next 5 position is predicted as:\n"
            for j in range(len(hpred_traj[index])):
                prompt += f"px{j}={hpred_traj[index][j][0]} and py{j}={hpred_traj[index][j][1]} \n"
        # prompt += f"All robot positions, since the start of the session until now is listed below:\n"
        # for i in range(len(prev_rstates)):
        #     prompt += (
        #         f"px_t{i}={prev_rstates[i][0][0]}, py_t{i}={prev_rstates[i][0][1]}\n"
        #     )
        prompt += f"current robot posisiton is px_now{i}={prev_rstates[-1][0][0]}, py_now{i}={prev_rstates[-1][0][1]}\n"

        x1, y1 = prev_rstates[-1][0][0:2]
        x2, y2 = rgoal[0][0:2]

        # Calculate the slope of the line
        slope = (y2 - y1) / (x2 - x1)

        # Calculate the angle in radians
        angle_radians = math.atan(slope)

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)

        prompt += f"Robot should reach the goal position of px_goal={rgoal[0][0]},py_goal={rgoal[0][1]}."
        prompt += f"direct angle to the goal is {angle_degrees}"
        # prompt += (
        #     f"\n go around the human near you to be away of the predicted trajectory ."
        # )
        return prompt
