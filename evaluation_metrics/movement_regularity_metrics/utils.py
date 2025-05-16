import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

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

                    new_row = create_row_pointing(bonus, effort_model, distance, episode, episode_data)
    
                    if results_df.empty:
                        results_df = pd.DataFrame([new_row])
                    else:
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    
    return results_df, number_of_episodes

def create_row_pointing(bonus, effort_model, distance, episode, episode_data):

    success = sum(episode_data['target_hit'])
    hit_indices = [i for i, hit in enumerate(episode_data['target_hit']) if hit]
    indices_target_spawned = [i for i, spawned in enumerate(episode_data['target_spawned']) if spawned]

    end_point_distances = []
    for idx in indices_target_spawned:
        if not episode_data['inside_target'][idx]:
            end_point_distances.append(np.linalg.norm(np.array(episode_data["hand_2distph_xpos"][idx]) - np.array(episode_data["target_position"][idx])))

    if len(end_point_distances) < 1:
        end_point_distance_mean = 0
    else:
        end_point_distance_mean = np.mean(end_point_distances)

    if not hit_indices:
        deviation_count = len(episode_data['inside_target'])
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
        "cum_distance": cum_dist,
        "end_point_distances": end_point_distance_mean
    }

    return new_row

def plot_fittslaw(data_df):

    all_IDs = []
    all_durations = []

    for _, row in data_df.iterrows():
        target_radiuses = np.array(row["target_radiuses"])
        start_distances = np.array(row["start_distances"])
        durations = np.array(row["durations_to_inside_target"])

        if len(target_radiuses) == len(start_distances) == len(durations):
            ID = np.log2(start_distances / target_radiuses + 1)
            all_IDs.extend(ID)
            all_durations.extend(durations)

    all_IDs = np.array(all_IDs)
    all_durations = np.array(all_durations)

    # Fit mit linearer Regression
    slope, intercept, r_value, p_value, std_err = linregress(all_IDs, all_durations)

    plt.figure(figsize=(8, 5))
    plt.scatter(all_IDs, all_durations, alpha=0.5, label="Messdaten")
    plt.plot(np.sort(all_IDs), intercept + slope * np.sort(all_IDs), color="red", label=f"Fit: T = {intercept:.2f} + {slope:.2f}·ID")
    plt.xlabel("ID (bits)")
    plt.ylabel("MT (s)")
    plt.title(f"Fitts' Law Fit:\n  Intercept (a): {intercept:.3f}\n  Slope (b): {slope:.3f}\n  R²: {r_value**2:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()