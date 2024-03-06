class PromptGen:
    @staticmethod
    def make_prompt(hvisible, hstate, prev_hstate, rstate, prev_rstate, hpred_traj):
        prompt = f"In this scene there are {len(hstate)} humans and 1 robot\n"
        for index, i in enumerate(hvisible):
            prompt += f"Human {i}, last state: {prev_hstate[i]}, current state: {hstate[i]} and predicted trajectory: {hpred_traj[index]} \n"
        prompt += f"Robot, last state: {prev_rstate[0]}, current state: {rstate[0]}"
        return prompt
