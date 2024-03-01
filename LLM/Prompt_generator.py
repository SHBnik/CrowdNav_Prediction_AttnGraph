class PromptGen:
    @staticmethod
    def make_prompt(hstate, prev_hstate, rstate, prev_rstate):
        # for human in humans_instance:
        #     print(human.get_full_state())
        prompt = f"In this scene there are {len(hstate)} humans and 1 robot\n"

        # Create a dictionary for previous states keyed by ID for easy lookup
        prev_states_dict = {human[-1]: human[:-1] for human in prev_hstate}

        # Describe changes for each human by matching IDs
        for current in hstate:
            current_id = current[-1]
            if current_id in prev_states_dict:
                previous = prev_states_dict[current_id]
                prompt += f"Human {current_id}, last state: {previous}, current state: {current[:-1]}\n"
        prompt += f"Robot, last state: {prev_rstate[0]}, current state: {rstate[0]}"
        return prompt
