import os
import pickle
import pandas as pd
import numpy as np

def create_filename(bonus, distance, effort_model, run_nr, task, variation):
    if bonus == 50:
        filename = f"mobl_arms_index_{task}_hit_bonus_50"
    else:
        filename = f"mobl_arms_index_{task}_{bonus}"

    if distance is not None:
        filename += f"_{distance}"

    filename += f"_{effort_model}"

    filename += f"/evaluate"
    if variation != None:
        filename += "_" + variation
    if run_nr != None:
        filename += "_" + str(run_nr)

    return filename

def create_df_remote_control(rewards):
    data_list = []
    for reward in rewards:
        folder = os.path.abspath(f"../../simulators/mobl_arms_index_remote_driving_{reward}/evaluate_1/")

        with open(os.path.join(folder, "state_log.pickle"), "rb") as f:
            data = pickle.load(f)

        successes = []
        distances_to_joystick = []
        distances_to_target_box = []
        joystick_times = []
        target_times = []
        number_of_episodes = len(data)
        for episode, episode_data in data.items():

            if episode_data["car_moving"].count(True) > 0:
                deviation_count = sum(
                                    prev and not curr
                                    for prev, curr in zip(episode_data["car_moving"][episode_data["car_moving"].index(True):][:-1], episode_data["car_moving"][episode_data["car_moving"].index(True):][1:])
                )
            else:
                deviation_count = len(episode_data["car_moving"])

            if int(episode.split('_')[1])>5:
                break

            dist_to_joystick = episode_data['dist_ee_to_joystick']
            dist_to_target_box = episode_data['dist_car_to_target']
            success = np.any((episode_data['inside_target']))#, np.abs(np.array(x_vel)) <= 0.001))

            if episode_data['car_moving'].count(True) == 0:
                joystick_time = 10
            else:
                moving_idx = episode_data['car_moving'].index(True)
                joystick_time = episode_data['timestep'][moving_idx]
            
            if episode_data['inside_target'].count(True) == 0:
                target_time = 10
            else:
                target_idx = episode_data['inside_target'].index(True)
                target_time = episode_data['timestep'][target_idx]

            joystick_times.append(joystick_time)
            target_times.append(target_time)
            successes.append(success)
            distances_to_joystick.append(sum(dist_to_joystick))
            distances_to_target_box.append(sum(dist_to_target_box))

        cum_dist_to_joystick = np.sum(distances_to_joystick)/5
        cum_dist_target = np.sum(distances_to_target_box)/5
        data_list.append([reward, np.sum(np.asarray(successes))/len(successes), cum_dist_to_joystick, cum_dist_target, np.mean(joystick_times), np.mean(target_times), deviation_count])

    df = pd.DataFrame(data_list,columns=["reward", "success", "cum_dist_to_joystick", "cum_dist_to_target","joystick_time", "target_time", "deviation_count"])
    df["reward"] = df["reward"].replace({
        "distance_and_completion_bonus": "Distance and task completion bonus",
        "distance_and_completion_bonus_and_joystick_reaching_bonus": "Distance and both task bonuses",
        "only_distance": "Distance",
        "only_bonus": "Both task bonuses",
    })
    return df

def create_df(boni, effort_models, distances, DIRNAME_SIMULATION, run_nr, task, variation=None):
    columns = ["bonus", "effort_model", "distance", "episode", "success_rate", "task_completion_time", "current_button", "initial_position"]
    results_df = pd.DataFrame(columns=columns)
    
    for bonus in boni:
        for effort_model in effort_models:
            for distance in distances:

                filename = create_filename(bonus, distance, effort_model, run_nr, task, variation)         
                filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}")
                full_path = os.path.join(filepath, "state_log.pickle")

                if not os.path.exists(full_path):
                    print(filename, "existiert nicht")
                    continue
                elif os.path.getsize(os.path.join(filepath, "state_log.pickle")) <= 0.1:
                    print(f"file {filename} has an empty state_log.pickle")
                    continue
                with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
                    data = pickle.load(f)

                number_of_episodes = len(data)
                for episode, episode_data in data.items():

                    if task == "choice_reaction":
                        new_row = create_row_choice_reaction(bonus, effort_model, distance, episode, episode_data)        
                    elif task == "tracking":
                        new_row = create_row_tracking(bonus, effort_model, distance, episode, episode_data)
                    elif task == "pointing":
                        new_row = create_row_pointing(bonus, effort_model, distance, episode, episode_data)
    
                    if results_df.empty:
                        results_df = pd.DataFrame([new_row])
                    else:
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    
    return results_df, number_of_episodes

def create_sparse_df_tracking(effort_models):
    columns = ["bonus", "effort_model", "distance", "episode", "success_rate", "task_completion_time", "current_button", "initial_position"]
    results_df = pd.DataFrame(columns=columns)

    for bonus in [8]:
        for effort in effort_models:
            for i in range(1,11):
                folder = os.path.abspath(f"../../simulators/mobl_arms_index_tracking_hit_bonus_{bonus}_{effort}/evaluate_{i}/")

                with open(os.path.join(folder, "state_log.pickle"), "rb") as f:
                    data = pickle.load(f)

                for episode, episode_data in data.items():
                    new_row = create_row_tracking(bonus, effort, "zero", episode, episode_data)
                    if results_df.empty:
                        results_df = pd.DataFrame([new_row])
                    else:
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    return results_df

def create_sparse_df_pointing(effort_models, DIRNAME_SIMULATION, run_nr, variation=None):
    columns = ["bonus", "effort_model", "distance", "episode", "success_rate", "task_completion_time", "current_button", "initial_position"]
    results_df = pd.DataFrame(columns=columns)

    for bonus in [8]:
        for effort in ["zero_effort"]:
            folder = os.path.join(DIRNAME_SIMULATION, f"mobl_arms_index_pointing_hit_bonus_{bonus}_{effort}/evaluate")
            
            if variation != None:
                folder += "_" + variation
            if run_nr != None:
                folder += "_" + str(run_nr)

            with open(os.path.join(folder, "state_log.pickle"), "rb") as f:
                data = pickle.load(f)

            for episode, episode_data in data.items():
                new_row = create_row_pointing(bonus, effort, "zero", episode, episode_data)
                if results_df.empty:
                    results_df = pd.DataFrame([new_row])
                else:
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    return results_df

def create_sparse_df_choice_reaction(effort_models):
    columns = ["bonus", "effort_model", "distance", "episode", "success_rate", "task_completion_time", "current_button", "initial_position"]
    results_df = pd.DataFrame(columns=columns)

    for bonus in [8]:
        for effort in ["zero_effort"]:
            folder = os.path.abspath(f"../../simulators/mobl_arms_index_choice_reaction_hit_bonus_{bonus}_{effort}/evaluate/")

            with open(os.path.join(folder, "state_log.pickle"), "rb") as f:
                data = pickle.load(f)

            for episode, episode_data in data.items():
                new_row = create_row_choice_reaction(bonus, effort, "zero", episode, episode_data)
                if results_df.empty:
                    results_df = pd.DataFrame([new_row])
                else:
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    return results_df

def moved_index_choice_reaction(distance, effort_idx):

    if distance == "dist":
        x_jitter = effort_idx - 0.1
    elif distance == "exp_dist":
        x_jitter = effort_idx + 0.1
    else:
        x_jitter = effort_idx

    return x_jitter

def create_row_pointing(bonus, effort_model, distance, episode, episode_data):

    success = sum(episode_data['target_hit'])
    hit_indices = [i for i, hit in enumerate(episode_data['target_hit']) if hit]
    if not hit_indices:
        deviation_count = None
    else:
        deviation_count = 0

    for idx in range(len(hit_indices)):
        if idx == 0:
            inside_target_array = episode_data['inside_target'][0:hit_indices[idx]]
        else:
            inside_target_array = episode_data['inside_target'][hit_indices[idx-1]+1:hit_indices[idx]]

        first_time_inside_target_idx = inside_target_array.index(True)
        if first_time_inside_target_idx == 0:
            break
        inside_target_array = inside_target_array[first_time_inside_target_idx:]

        deviation_count += sum(
            prev and not curr
            for prev, curr in zip(inside_target_array[:-1], inside_target_array[1:])
        )

    completion_times_split = []
    current_segment = []
    indices_target_spawned = [i for i, spawned in enumerate(episode_data['target_spawned']) if spawned]
    for i, new_target in enumerate(episode_data['target_spawned']):
        if new_target:
            completion_times_split.append(np.array(current_segment))
            current_segment = []
        current_segment.append(episode_data['timestep'][i])      
    
    final_times = []
    segment_indices = []
    idx = 0
    k = -1
    for segment in completion_times_split:
        if k > -1:
            idx = indices_target_spawned[k]
        for i, value in enumerate(segment):
            if i == len(segment)-1:
                time = 4
                final_times.append(time)
                segment_indices.append(idx)
                k += 1
            elif (episode_data['inside_target'][idx] and (idx not in indices_target_spawned)):
                if k > -1:
                    time = value - episode_data['timestep'][indices_target_spawned[k]]
                else: 
                    time = value
                final_times.append(time)
                segment_indices.append(idx)
                k += 1
                break
            idx +=1
    
    start_distances = [np.linalg.norm(np.array(episode_data["hand_2distph_xpos"][0]) - np.array(episode_data["target_position"][0])) - episode_data["target_radius"][0]]
    target_radiuses = [episode_data["target_radius"][0]]

    for idx in indices_target_spawned[:-1]:
        fingertip_pos = episode_data["hand_2distph_xpos"][idx]
        target_pos = episode_data["target_position"][idx]
        start_distance = np.linalg.norm(np.array(fingertip_pos) - np.array(target_pos)) - episode_data["target_radius"][idx]
        target_radiuses.append(episode_data["target_radius"][idx])
        start_distances.append(start_distance)

    if sum(episode_data['target_hit']) == 0:
        durations = 4*np.ones(10)
    else:
        true_indices = np.where(np.array(episode_data["target_spawned"]) == True)[0]
        durations = np.array(episode_data["timestep"]) [true_indices]
        durations = np.insert(np.diff(durations), 0, durations[0])


    cum_dist = 0
    for i in range(len(episode_data["hand_2distph_xpos"])):
        dist = np.linalg.norm(np.array(episode_data["hand_2distph_xpos"][i]) - np.array(episode_data["target_position"][i]))
        if dist < episode_data["target_radius"][i]:
            dist = 0
        cum_dist += dist

    new_row = {
        "bonus": bonus,
        "effort_model": effort_model,
        "distance": distance,
        "episode": episode,
        "success_rate": success,
        "hand_2distph_xpos": episode_data["hand_2distph_xpos"],
        "task_completion_time": sum(durations)/10,   
        "durations_to_inside_target": final_times,
        "target_radiuses": target_radiuses,
        "start_distances": start_distances,
        "deviation_count": deviation_count,
        "cum_distance": cum_dist
    }

    return new_row

def create_row_tracking(bonus, effort_model, distance, episode, episode_data):
    if episode_data['inside_target'].count(True) < 1:
        deviation_count = len(episode_data['inside_target'])
    else:
        inside_target_array = episode_data['inside_target'][episode_data['inside_target'].index(True):]
        deviation_count = sum(
            prev and not curr
            for prev, curr in zip(inside_target_array[:-1], inside_target_array[1:])
        )


    avg_timesteps_inside_target = sum(episode_data['inside_target'])/len(episode_data['inside_target'])
    new_row = {
        "bonus": bonus,
        "effort_model": effort_model,
        "distance": distance,
        "episode": episode,
        "time_inside_target": avg_timesteps_inside_target,
        "hand_2distph_xpos": episode_data["hand_2distph_xpos"],
        "distance_to_target": sum(episode_data["distance_to_target"][1:])/len(episode_data["distance_to_target"][1:]),
        "deviation_count": deviation_count
    }
    return new_row

def create_row_choice_reaction(bonus, effort_model, distance, episode, episode_data):
    shoulder_pos = np.array([0.02, -0.31, 0.938])
    target_positions = {"button-0":  np.array([0.41, -0.07, -0.15]) + shoulder_pos, "button-1": np.array([0.41, 0.07, -0.15]) + shoulder_pos, "button-2": np.array([0.5, -0.07, -0.05]) + shoulder_pos, "button-3": np.array([0.5, 0.07, -0.05]) + shoulder_pos}
    if sum(episode_data['target_hit']) == 0:
        durations = 4*np.ones(10)
    else:
        durations = [episode_data["task_completion_time"][i] for i, x in enumerate(episode_data['target_hit']) if x]
        durations = np.insert(np.diff(durations), 0, durations[0])

    success = sum(episode_data['target_hit'])/10
    new_row = {
        "bonus": bonus,
        "effort_model": effort_model,
        "distance": distance,
        "episode": episode,
        "success_rate": success,
        "success_rate_button-0": episode_data["success_rate_button-0"][-1],
        "success_rate_button-1": episode_data["success_rate_button-1"][-1],
        "success_rate_button-2": episode_data["success_rate_button-2"][-1],
        "success_rate_button-3": episode_data["success_rate_button-3"][-1],
        "hand_2distph_xpos": episode_data["hand_2distph_xpos"],
        "task_completion_time":sum(durations)/10, 
        "target_positions": [target_positions[btn] for btn in episode_data["current_button"]],
        "cum_distance": np.sum(np.linalg.norm(np.array([target_positions[btn] for btn in episode_data["current_button"]]) - np.array(episode_data["hand_2distph_xpos"]), axis=1))
    }     
    return new_row