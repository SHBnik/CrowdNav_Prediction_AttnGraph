import numpy as np


def extract_positions_and_velocities(list):
    data_array = []
    goal = []
    try:
        for full_state in list:
            # position = (full_state[0], full_state[1])  # px, py
            # velocity = (full_state[2], full_state[3])  # vx, vy
            # theta = full_state[8]  # theta

            data_array.append(
                [
                    round(full_state[0], 2),
                    round(full_state[1], 2),
                    round(full_state[2], 2),
                    round(full_state[3], 2),
                    round(full_state[8], 2),
                ]
            )
            goal.append(
                [
                    round(full_state[5], 2),
                    round(full_state[6], 2)
                ]
            )
    except:
        full_state = list.copy()
        data_array.append(
            [
                round(full_state[0], 2),
                round(full_state[1], 2),
                round(full_state[2], 2),
                round(full_state[3], 2),
                round(full_state[8], 2),
            ]
        )
        goal.append(
            [
                round(full_state[5], 2),
                round(full_state[6], 2)
            ]
        )


    return data_array, goal


def get_pred_traj_pose(gst_out_traj, robot_pose, human_num=20, predict_steps=5):
    humans_pred_traj = []
    # print(gst_out_traj)
    for i in range(human_num):
        # add future predicted positions of each human
        this_human_pred_traj = []
        if gst_out_traj[0] is not None:
            for j in range(predict_steps):
                try:
                    this_human_pred_traj.append(
                        (
                            gst_out_traj[0][i, (2 * j) : (2 * j + 2)] + robot_pose
                        ).tolist()
                    )
                except:
                    pass
        humans_pred_traj.append(this_human_pred_traj)
    return humans_pred_traj
