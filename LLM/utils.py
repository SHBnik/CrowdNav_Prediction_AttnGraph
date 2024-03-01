def extract_positions_and_velocities(humans):
    data_array = []
    try:
        for human in humans:
            full_state = human.get_full_state_list()
            # position = (full_state[0], full_state[1])  # px, py
            # velocity = (full_state[2], full_state[3])  # vx, vy
            # theta = full_state[8]  # theta

            data_array.append(
                [
                    full_state[0],
                    full_state[1],
                    full_state[2],
                    full_state[3],
                    full_state[8],
                    human.id,
                ]
            )
    except:
        full_state = humans.get_full_state_list()
        data_array.append(
            [
                full_state[0],
                full_state[1],
                full_state[2],
                full_state[3],
                full_state[8],
            ]
        )

    return data_array
