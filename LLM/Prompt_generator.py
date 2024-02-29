class PromptGen:
    @staticmethod
    def make_prompt(humans_instance):
        humans_count = len(humans_instance)
        for human in humans_instance:
            print(human.get_full_state())
