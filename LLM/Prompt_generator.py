class PromptGen:
    @staticmethod
    def make_prompt(hvisible, hstate, prev_hstate, rstate, prev_rstate, hpred_traj, next_pos, rgoal):
        prompt = f"In this scene there is a robot and some obstacales.\n"
        prompt += f"obsticles position are: \n"
        for index, i in enumerate(hvisible):
            prompt += f"X_obstacle{index}={hstate[i][0]} and Y_obstacle{index}={hstate[i][1]}\n"
            for j in range(len(hpred_traj[index])):
                prompt += f"X_next_obstacle{(index)}_{j + 1}={hpred_traj[index][j][0]} and Y_next_obstacle{(index)}_{j + 1}={hpred_traj[index][j][1]} \n"
        prompt += f"current robot posisiton is X_current={rstate[0][0]}, Y_current={rstate[0][1]}\n"
        prompt += f"Robot next position is X_next={next_pos[0]},Y_next={next_pos[1]}."
        # prompt += f"Robots goal is to reach px_goal={rgoal[0]},py_goal={rgoal[1]}."

        # prompt = f"In this scene there are {len(hstate)} humans and 1 robot\n"
        # for index, i in enumerate(hvisible):
        #     prompt += f"Human {i}, last state: {prev_hstate[i]}, current state: {hstate[i]} and predicted trajectory: {hpred_traj[index]} \n"
        # prompt += f"Robot, last state: {prev_rstate[0]}, current state: {rstate[0]} \n"
        # prompt += f"Robot's generated next position is  {next_pos}\n"
        return prompt
