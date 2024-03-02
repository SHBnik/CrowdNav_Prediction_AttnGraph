class PromptGen:
    @staticmethod
    def make_prompt(hstate, prev_hstate, rstate, prev_rstate, hpred_traj):
        prompt = f"In this scene there are {len(hstate)} humans and 1 robot\n"
        for i in range(len(hstate)):
            prompt += f"Human {i}, last state: {prev_hstate[i]}, current state: {hstate[i]} and predicted trajectory: {hpred_traj[i]} \n"
        prompt += f"Robot, last state: {prev_rstate[0]}, current state: {rstate[0]}"
        return prompt
