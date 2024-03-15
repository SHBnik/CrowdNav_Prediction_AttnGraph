import numpy as np
import torch

from crowd_sim.envs.utils.info import *


from LLM.Prompt_generator import PromptGen
import LLM.utils as llmutil
from LLM.GPT_request import GPT
import keyboard
import math


def evaluate(
    actor_critic,
    eval_envs,
    num_processes,
    device,
    test_size,
    logging,
    config,
    args,
    visualize=False,
):
    """function to run all testing episodes and log the testing metrics"""
    # initializations
    eval_episode_rewards = []

    if config.robot.policy not in ["orca", "social_force"]:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states["human_node_rnn"] = torch.zeros(
            num_processes,
            node_num,
            actor_critic.base.human_node_rnn_size,
            device=device,
        )

        eval_recurrent_hidden_states["human_human_edge_rnn"] = torch.zeros(
            num_processes,
            edge_num,
            actor_critic.base.human_human_edge_rnn_size,
            device=device,
        )

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []

    all_steps = []

    gpt = GPT()

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, "envs"):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        global_time = 0.0
        path_len = 0.0
        too_close = 0.0
        last_pos = obs["robot_node"][0, 0, :2].cpu().numpy()
        stat = None
        safe_direction = 0
        human_state = None
        robot_state = None
        prev_human_state = None
        prev_robot_state = None

        while not done:
            stepCounter = stepCounter + 1
            if config.robot.policy not in ["orca", "social_force"]:
                # run inference on the NN policy
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True,
                    )
            else:
                action = torch.zeros([1, 2], device=device)
            if not done:
                global_time = baseEnv.global_time

                ##### Get Info
                _robot_full_state = None
                _humans_full_state = None
                # stat = None

                if args.env_name == "CrowdSimPredRealGST-v0" and config.env.use_wrapper:
                    out_pred = obs["spatial_edges"][:, :, 2:].to("cpu").numpy()
                    # send manager action to all processes
                    (ack, _robot_full_state, _humans_full_state) = eval_envs.envs[
                        0
                    ].talk2Env(out_pred[0])

                human_state, human_goal = llmutil.extract_positions_and_velocities(
                    _humans_full_state
                )
                robot_state, robot_goal = llmutil.extract_positions_and_velocities(
                    _robot_full_state
                )

                if stepCounter > 1:
                    # 20 X 5
                    humans_pred_traj_pose = llmutil.get_pred_traj_pose(
                        out_pred, obs["robot_node"][0, 0, :2].cpu().numpy()
                    )

                    # Get indices of True values in the mask
                    mask_indices = (
                        obs["visible_masks"].nonzero(as_tuple=True)[1].tolist()
                    )
                    # Use the indices to select the corresponding elements from 'Human'
                    masked_humans = [human_state[i] for i in mask_indices]
                    # indice is based on distance to robot
                    fuckedup_indices = llmutil.generate_traj_mask(
                        masked_humans, robot_state
                    )

                    masked_trajectories = [
                        humans_pred_traj_pose[i] for i in range(len(mask_indices))
                    ]

                    reordered_masked_trajectories = [None] * len(masked_trajectories)

                    for original_index, new_index in enumerate(fuckedup_indices):
                        reordered_masked_trajectories[new_index] = masked_trajectories[
                            original_index
                        ]

            #####

            # render
            if visualize:
                eval_envs.envs[0].render([safe_direction, stat])

            # keyboard.wait("space")
            # Obser reward and next obs
            if stat != None:
                if stat == False:
                    obs, rew, done, infos = eval_envs.step(action)
                else:
                    if (
                        math.dist(
                            robot_state[0][0:2],
                            [
                                round(float(obs["robot_node"][0][0][3]), 2),
                                round(float(obs["robot_node"][0][0][4]), 2),
                            ],
                        )
                        > 2
                    ):
                        action = llmutil.translate_action(safe_direction)
                    # action = torch.zeros([1, 2], device=device)
                    obs, rew, done, infos = eval_envs.step(action)
            else:
                obs, rew, done, infos = eval_envs.step(action)

            ################################# Do every thing here so we can use the render later
            # TODO: Ours
            # TODO: Generate action by LLM and compare
            # TODO: Generate alarm based on action and change planning

            # print("Visible Humans: ")
            # print(mask_indices)
            # print("#######################################")
            # print("Robot: ")
            # print(robot_state)
            # print("#######################################")
            # print("Human: ")
            # print(masked_humans)
            # print("#######################################")
            # print("Trajectory: ")
            # print(reordered_masked_trajectories)
            # print("#######################################")
            # print("action: ")
            # print(action[0]*0.25)
            # print(robot_state)
            # print(robot_state[0][0] + action[0][0]*0.25, robot_state[0][1] + action[0][1]*0.25)
            # print("#######################################")
            if stepCounter > 1:
                obstacles = []
                for index, i in enumerate(mask_indices):
                    obstacles.append(human_state[i][0:2])
                    for j in range(len(reordered_masked_trajectories[index])):
                        obstacles.append(reordered_masked_trajectories[index][j][0:2])
                safe_direction = llmutil.calculate_safe_direction(
                    robot_state, obstacles
                )

                prompt = PromptGen.make_prompt(
                    mask_indices,
                    human_state,
                    prev_human_state,
                    robot_state,
                    prev_robot_state,
                    reordered_masked_trajectories,
                    [
                        round(float(obs["robot_node"][0][0][0]), 2),
                        round(float(obs["robot_node"][0][0][1]), 2),
                    ],
                    [
                        round(float(obs["robot_node"][0][0][3]), 2),
                        round(float(obs["robot_node"][0][0][4]), 2),
                    ],
                )

                # print(prompt)
                stat = gpt.ask(prompt)

                prev_human_state = human_state.copy()
                prev_robot_state = robot_state.copy()

            ###################################

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(
                obs["robot_node"][0, 0, :2].cpu().numpy() - last_pos
            )
            last_pos = obs["robot_node"][0, 0, :2].cpu().numpy()

            if isinstance(infos[0]["info"], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]["info"].min_dist)

            episode_rew += rew[0]

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device,
            )

            for info in infos:
                if "episode" in info.keys():
                    eval_episode_rewards.append(info["episode"]["r"])

        # an episode ends!
        print("")
        print("Reward={}".format(episode_rew))
        print("Episode", k, "ends in", stepCounter)
        all_path_len.append(path_len)
        all_steps.append(stepCounter)
        too_close_ratios.append(too_close / stepCounter * 100)

        if isinstance(infos[0]["info"], ReachGoal):
            success += 1
            success_times.append(global_time)
            print("Success")
        elif isinstance(infos[0]["info"], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print("Collision")
        elif isinstance(infos[0]["info"], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            print("Time out")
        elif isinstance(infos[0]["info"] is None):
            pass
        else:
            raise ValueError("Invalid end signal from environment")

        print(success_times)
        print(collision_times)
        print(collision_cases)
        print(all_steps)
        print(all_path_len)
        print(timeout_cases)
        if len(success_times) != 0:
            print(sum(success_times) / len(success_times))
        print(success / len(all_steps))
        print(len(collision_times) / len(all_steps))
        print(sum(all_steps) / len(all_steps))
        print(sum(all_path_len) / len(all_path_len))

        if k >= 35:
            break
    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    # assert success + collision + timeout == test_size
    avg_nav_time = (
        sum(success_times) / len(success_times) if success_times else time_limit
    )  # baseEnv.env.time_limit

    # logging
    logging.info(
        "Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, "
        "nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, "
        "average minimal distance during intrusions: {:.2f}".format(
            success_rate,
            collision_rate,
            timeout_rate,
            avg_nav_time,
            np.mean(all_path_len),
            np.mean(too_close_ratios),
            np.mean(min_dist),
        )
    )

    logging.info("Collision cases: " + " ".join([str(x) for x in collision_cases]))
    logging.info("Timeout cases: " + " ".join([str(x) for x in timeout_cases]))
    print(
        " Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)
        )
    )

    eval_envs.close()
