import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import pandas as pd
import matplotlib.lines as mlines

fontsize = 16
figsize = (7,5)
legend_fontsize=12
markersize=12
effort_map = {
    "zero_effort": "No",
    "dc_effort_w1": "DC",
    "jac_effort_w1": "JAC",
    "ctc_effort_w1": "CTC",
    "armmovementpaper_effort": "EJK"
}

shoulder_pos = np.array([0.02, -0.31, 0.938])
target_positions = {"button-0":  np.array([0.41, -0.07, -0.15]) + shoulder_pos, "button-1": np.array([0.41, 0.07, -0.15]) + shoulder_pos, "button-2": np.array([0.5, -0.07, -0.05]) + shoulder_pos, "button-3": np.array([0.5, 0.07, -0.05]) + shoulder_pos,
                        "button-4": np.array([0.482445 ,-0.38 ,0.943]),"button-5": np.array([0.392445 ,-0.38 ,0.843])}

def calculate_speed_choice_reaction(data):
    all_velocities = []
    all_times = []

    for episode, episode_data in data.items():
        times = episode_data['timestep']
        timestep = times[1] - times[0]
        button_pos = [target_positions[button] for button in episode_data["current_button"]]
        end_effector_positions = episode_data["hand_2distph_xpos"]
        new_buttons = episode_data['new_button_generated']
        velocities = [0]
        normalizedCentroidVector = np.abs(end_effector_positions[0] - button_pos[0])/np.linalg.norm(end_effector_positions[0] - button_pos[0]).transpose()
        idx_next_button = new_buttons.index(True)
        for current_timestep in range(1,idx_next_button):
        
            end_effector_vel = np.abs(end_effector_positions[current_timestep] - end_effector_positions[current_timestep-1])/timestep
            centroidVelProjection = (end_effector_vel * normalizedCentroidVector).sum()
            velocities.append(centroidVelProjection)   

        smoothed_velocities = savgol_filter(velocities, 8, 4)
        all_velocities.append(smoothed_velocities)
        all_times.append(episode_data['timestep'][0:len(velocities)])

    return all_velocities, all_times

def calculate_speed_pointing(data):
         
    all_velocities = []
    all_times = []
    
    for episode, episode_data in data.items():
        times = episode_data['timestep']
        timestep = times[1] - times[0]
        target_positions = episode_data["target_position"]
        end_effector_positions = episode_data["hand_2distph_xpos"]
        new_target_indices = [0] + [i for i, spawned in enumerate(episode_data['target_spawned']) if spawned]

        for i, idx in enumerate(new_target_indices[:-1]):
            velocities = []
            normalizedCentroidVector = np.abs(end_effector_positions[idx] - target_positions[idx])/np.linalg.norm(end_effector_positions[idx] - target_positions[idx]).transpose()

            for current_timestep in range(idx+1,new_target_indices[i+1]+1):                       
                end_effector_vel = np.abs(end_effector_positions[current_timestep] - end_effector_positions[current_timestep-1])/timestep
                centroidVelProjection = (end_effector_vel * normalizedCentroidVector).sum()
                velocities.append(centroidVelProjection)   
            
            smoothed_velocities = savgol_filter(velocities, 8, 4)
            all_velocities.append(smoothed_velocities)
            all_times.append(np.array(times[idx:new_target_indices[i+1]]) - times[idx])

    return all_velocities, all_times

def calculate_speed_tracking(data):
    all_velocities = []
    all_times = []
    
    for episode, episode_data in data.items():
        times = episode_data['timestep']
        timestep = times[1] - times[0]
        target_positions = episode_data["target_position"]
        end_effector_positions = episode_data["hand_2distph_xpos"]

        if episode_data['inside_target'].count(True)>0:
            end_idx = episode_data['inside_target'].index(True)
        else:
            end_idx = len(times)
        velocities = [0]
        normalizedCentroidVector = np.abs(end_effector_positions[0] - target_positions[1])/np.linalg.norm(end_effector_positions[0] - target_positions[1]).transpose()

        for current_timestep in range(1,end_idx):                       
            end_effector_vel = np.abs(end_effector_positions[current_timestep] - end_effector_positions[current_timestep-1])/timestep
            centroidVelProjection = (end_effector_vel * normalizedCentroidVector).sum()
            velocities.append(centroidVelProjection)   
            
        smoothed_velocities = savgol_filter(velocities, 8, 4)
        all_velocities.append(smoothed_velocities)
        all_times.append(episode_data['timestep'][0:len(velocities)])

    return all_velocities, all_times
    
def create_df_remote_control(rewards, DIRNAME_SIMULATION, variation=None):
    columns = ["bonus", "effort_model", "distance", "episode"]
    results_df = pd.DataFrame(columns=columns)

    for reward in rewards:
        if variation is None:
            filename = f"mobl_arms_index_remote_driving_{reward}/evaluate_1/"
        else:
            filename = f"mobl_arms_index_remote_driving_{reward}/evaluate_{variation}_1/"
        filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}")
        with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
            data = pickle.load(f)

        number_of_episodes = len(data['episode_0'])
        new_row = create_row_remote_control(reward, data)        

        if results_df.empty:
            results_df = pd.DataFrame([new_row])
        else:
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            
    results_df["reward"] = results_df["reward"].replace({
        "distance_inside_target": "Distance and task completion bonus",
        "distance_inside_target_joystick": "Distance and both task bonuses",
        "only_distance": "Distance",
        "only_bonus": "Both task bonuses",
    })
    return results_df, number_of_episodes

def create_row_remote_control(reward, data):
    all_times, all_velocities = calculate_speed_remote_control(data)

    RTPs = []
    sub_movement_counts = []

    for i, vel in enumerate(all_velocities):
        vel = np.array(vel)
        all_times[i] = np.array(all_times[i])
        peaks, _ = find_peaks(vel, prominence=0.01)
        valleys, _ = find_peaks(-vel, prominence=0.01)

        if len(valleys) > 0:
            if valleys[0] > 5 or len(valleys) < 2:
                start_refinement_phase = valleys[0]
            else:
                start_refinement_phase = valleys[1]
            RTP = (all_times[i][-1] - all_times[i][start_refinement_phase])/all_times[i][-1]
            count = count_submovements(valleys, peaks)
            RTPs.append(RTP)
            sub_movement_counts.append(count)
        else:
            sub_movement_counts.append(1)
    
    if len(RTPs) == 0:
        RTPs = [1]

    new_row = {
        "reward": reward,
        "RTP": np.mean(RTPs),
        "submovements_count": np.mean(sub_movement_counts)
    }     
    return new_row

def calculate_speed_remote_control(data):
        
    all_velocities = []
    all_times = []
    
    for episode, episode_data in data.items():
        if episode_data['car_moving'].count(True) > 0:
            end_idx = episode_data['car_moving'].index(True)
        else:
            end_idx = len(episode_data['timestep'])
            if end_idx < 8:
                continue
        times = episode_data['timestep']
        timestep = times[1] - times[0]
        target_positions = episode_data["target_position"]
        end_effector_positions = episode_data["hand_2distph_xpos"]

        velocities = []
        for idx in range(1,end_idx):
            normalizedCentroidVector = np.abs(end_effector_positions[idx] - target_positions[idx])/np.linalg.norm(end_effector_positions[idx] - target_positions[idx]).transpose()
            end_effector_vel = np.abs(end_effector_positions[idx] - end_effector_positions[idx-1])/timestep
            centroidVelProjection = (end_effector_vel * normalizedCentroidVector).sum()
            velocities.append(centroidVelProjection)

        smoothed_velocities = savgol_filter(velocities, 11, 3)
        all_velocities.append(smoothed_velocities)
        all_times.append(np.array(times[:end_idx-1]))

    return all_times, all_velocities

def count_submovements(valleys, peaks):
    if 0 not in valleys:
        valleys = np.insert(valleys, 0, 0)
    extrema = np.sort(np.concatenate([peaks, valleys]))
    types = ['max' if i in peaks else 'min' for i in extrema]
    count = sum(types[i:i+3] == ['min', 'max', 'min'] for i in range(len(types)-2))
    if types[-1] == 'max' and len(peaks) == 1:
        count += 1
    return count

def create_df(boni, effort_models, distances, DIRNAME_SIMULATION, run_nr, task, variation=None):
    columns = ["bonus", "effort_model", "distance", "episode"]
    results_df = pd.DataFrame(columns=columns)
    
    for bonus in boni:
        for effort_model in effort_models:
            for distance in distances:

                filename = create_filename(bonus, distance, effort_model, run_nr, task, variation)      
                filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}")
                full_path = os.path.join(filepath, "state_log.pickle")

                if not os.path.exists(full_path):
                    print(full_path, "existiert nicht")
                    continue
                elif os.path.getsize(os.path.join(filepath, "state_log.pickle")) <= 0.1:
                    print(f"file {filename} has an empty state_log.pickle")
                    continue

                with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
                    data = pickle.load(f)

                number_of_episodes = len(data)
                new_row = create_row(task, bonus, effort_model, distance, data)        

                if results_df.empty:
                    results_df = pd.DataFrame([new_row])
                else:
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    
    return results_df, number_of_episodes

def create_sparse_df(task):
    columns = ["bonus", "effort_model", "distance", "episode", "success_rate", "task_completion_time", "current_button", "initial_position"]
    results_df = pd.DataFrame(columns=columns)

    for bonus in [8]:
        for effort in ["zero_effort"]:
            folder = os.path.abspath(f"../../simulators/mobl_arms_index_{task}_hit_bonus_{bonus}_{effort}/evaluate_1/")

            with open(os.path.join(folder, "state_log.pickle"), "rb") as f:
                data = pickle.load(f)

            new_row = create_row(task, bonus, effort, "zero", data)
            if results_df.empty:
                results_df = pd.DataFrame([new_row])
            else:
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    return results_df

def create_row(task, bonus, effort_model, distance, data):
    if task == 'choice_reaction':
        all_velocities, all_times = calculate_speed_choice_reaction(data)
    elif task == 'pointing':
        all_velocities, all_times = calculate_speed_pointing(data)
    elif task=="tracking":
        all_velocities, all_times = calculate_speed_tracking(data)

    RTPs = []
    sub_movement_counts = []

    for i, vel in enumerate(all_velocities):
        vel = np.array(vel)
        all_times[i] = np.array(all_times[i])
        peaks, _ = find_peaks(vel, prominence=0.01)
        valleys, _ = find_peaks(-vel, prominence=0.01)

        if len(valleys) > 0:
            if valleys[0] > 5 or len(valleys) < 2:
                start_refinement_phase = valleys[0]
            else:
                start_refinement_phase = valleys[1]
            RTP = (all_times[i][-1] - all_times[i][start_refinement_phase])/all_times[i][-1]
            count = count_submovements(valleys, peaks)
            RTPs.append(RTP)
            sub_movement_counts.append(count)
        else:
            sub_movement_counts.append(1)

    new_row = {
        "bonus": bonus,
        "effort_model": effort_model,
        "distance": distance,
        "RTP": np.mean(RTPs),
        "submovements_count": np.mean(sub_movement_counts)
    }     
    return new_row

def create_filename(bonus, distance, effort_model, run_nr, task, variation):
    if bonus == 50:
        filename = f"mobl_arms_index_{task}_hit_bonus_50"
    else:
        filename = f"mobl_arms_index_{task}_{bonus}"

    if distance != 'no':
        filename += f"_{distance}"

    filename += f"_{effort_model}"

    filename += f"/evaluate"
    if variation != None:
        filename += "_" + variation
    if run_nr != None:
        filename += "_" + str(run_nr)

    return filename

def plot_submovement_count(titlename, figname, data_df, colors):

    legend_labels = []
    for key in colors.keys():
        legend_labels.append(mlines.Line2D([0], [0], color=colors[key], lw=4, label=key))
    fig, ax = plt.subplots(figsize=figsize)

    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['submovements_count'].mean().reset_index()
    result.iloc[[4, 5]] = result.iloc[[5, 4]].values
    effort_models = []

    for idx, row in result.iterrows():
        if row['bonus'] == "no_bonus":
            color = colors['Distance']
        elif (row['bonus'] == "hit_bonus_8" and row["distance"] == "no" and row["effort_model"] == "zero_effort") or row["bonus"] == 8:
            color = colors['Bonus']
        else:
            if row['effort_model'] == 'zero_effort':
                color = colors['Distance + Bonus']
            else:
                color = colors['Distance + Bonus + Effort Model']

        ax.bar(idx, row['submovements_count'], color=color)
        effort_models.append(row["effort_model"])

    plt.rcParams.update({'font.size': fontsize})
    plt.title(titlename)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Number of submovements", fontsize=fontsize)
    plt.legend(handles=legend_labels, title="Reward components", prop={'size': 12}, loc = 'upper left')
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig(f"Nr_of_submovements/number_of_submovements_{figname}.png", bbox_inches="tight")
    plt.show()

def plot_refinement_time(titlename, figname, data_df, colors):

    fig, ax = plt.subplots(figsize=figsize)

    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['RTP'].mean()
    result.iloc[[4, 5]] = result.iloc[[5, 4]].values
    effort_models = []

    for idx, row in result.iterrows():
        if row['bonus'] == "no_bonus":
            color = colors['Distance']
        elif row['bonus'] == 8 or row["distance"] == "no":
            color = colors['Bonus']
        else:
            if row['effort_model'] == 'zero_effort':
                color = colors['Distance + Bonus']
            else:
                color = colors['Distance + Bonus + Effort Model']

        ax.bar(idx, row['RTP']*100, color=color)
        effort_models.append(row["effort_model"])

    plt.rcParams.update({'font.size': fontsize})
    plt.title(titlename)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Refinement time proportion (%)", fontsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig(f"RTP/RTP_{figname}.png", bbox_inches="tight")
    plt.show()

def plot_submovement_count_remote_control(df, colors):
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    bars = plt.bar(range(len(df)), df["submovements_count"], color=colors)
    plt.legend(bars, df["reward"], prop={'size': 12},title = 'Reward function')#, loc="upper left"
    plt.title("Remote Control Task")
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Reward", fontsize=fontsize)
    plt.ylabel("Number of submovements", fontsize=fontsize)
    plt.xticks([]) 
    plt.legend(bars, df["reward"], prop={'size': 12},title = 'Reward function')
    plt.savefig(f"Nr_of_submovements/number_of_submovements_remote_control.png", bbox_inches="tight")
    plt.show()

def plot_refinement_time_remote_control(df, colors):
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    bars = plt.bar(range(len(df)), df["RTP"]*100, color=colors)
    plt.title("Remote Control Task")
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Reward", fontsize=fontsize)
    plt.ylabel("Refinement Time Proportion (%)", fontsize=fontsize)
    plt.xticks([]) 
    plt.savefig(f"RTP/RTP_remote_control.png", bbox_inches="tight")
    plt.show()

def calculate_endpoint_variance(boni, effort_models, distances, DIRNAME_SIMULATION, task, variation=None):
    results = []

    for bonus in boni:
        for effort_model in effort_models:
            for distance in distances:
                end_positions = {f"episode_{i}": [] for i in range(5)}

                for i in range(1,11):
                    filename = create_filename(bonus, distance, effort_model, i, task, variation)      
                    filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}")

                    with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
                        data = pickle.load(f)

                    for episode, episode_data in data.items():
                        if task == "pointing":
                            keyword = "target_spawned"
                            end_idx = episode_data[keyword].index(True)
                        elif task == "choice_reaction":
                            keyword = "new_button_generated"
                            end_idx = episode_data[keyword].index(True)
                        elif task == "tracking":
                            end_idx = len(episode_data['timestep'])-1
                        end_positions[episode].append(episode_data["hand_2distph_xpos"][end_idx])

                endpoint_variance = {}
                for key, value in end_positions.items():
                    endpoint_variance[key] = np.var(value)

                for episode, variance in endpoint_variance.items():
                    results.append({
                        "bonus": bonus,
                        "effort_model": effort_model,
                        "distance": distance,
                        "episode": episode,
                        "endpoint_variance": variance
                    })

    df = pd.DataFrame(results)
    return df

def calculate_endpoint_variance_remote_control(rewards, DIRNAME_SIMULATION, variation=None):
    results = []

    for reward in rewards:
        end_positions = {f"episode_{i}": [] for i in range(5)}

        for i in range(1,11):
            if variation is None:
                filename = f"mobl_arms_index_remote_driving_{reward}/evaluate_{i}/"  
            else:
                filename = f"mobl_arms_index_remote_driving_{reward}/evaluate_{variation}_{i}/" 
            filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}")

            with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
                data = pickle.load(f)

            keyword = "car_moving"
            for episode, episode_data in data.items():
                if episode_data[keyword].count(True)> 0:
                    end_idx = episode_data[keyword].index(True)
                    end_positions[episode].append(episode_data["hand_2distph_xpos"][end_idx])
                else:
                    end_positions[episode].append(np.array([0,0,0]))

        endpoint_variance = {}
        for key, value in end_positions.items():
            endpoint_variance[key] = np.var(value)

        for episode, variance in endpoint_variance.items():
            results.append({
                "reward": reward,
                "episode": episode,
                "endpoint_variance": variance
            })

    df = pd.DataFrame(results)
    return df

def plot_endpoint_variance(titlename, figname, data_df, colors):

    fig, ax = plt.subplots(figsize=figsize)

    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['endpoint_variance'].mean()
    result.iloc[[4, 5]] = result.iloc[[5, 4]].values
    effort_models = []

    for idx, row in result.iterrows():
        if row['bonus'] == "no_bonus":
            color = colors['Distance']
        elif (row['bonus'] == "hit_bonus_8" and row["distance"] == "no" and row["effort_model"] == "zero_effort") or row["bonus"] == 8:
            color = colors['Bonus']
        else:
            if row['effort_model'] == 'zero_effort':
                color = colors['Distance + Bonus']
            else:
                color = colors['Distance + Bonus + Effort Model']

        ax.bar(idx, row['endpoint_variance'], color=color)
        effort_models.append(row["effort_model"])

    plt.rcParams.update({'font.size': fontsize})
    plt.title(titlename)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Endpoint variance", fontsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig(f"endpoint_variance/endpoint_variance_{figname}.png", bbox_inches="tight")
    plt.show()

def plot_endpoint_variance_remote_control(titlename, figname, data_df, colors):

    fig, ax = plt.subplots(figsize=figsize)

    result = data_df.groupby(['reward'], as_index=False)['endpoint_variance'].mean()
    result['reward'] = pd.Categorical(result['reward'], categories=['distance_inside_target', 'distance_inside_target_joystick', 'only_distance', 'only_bonus'], ordered=True)
    result = result.sort_values('reward').reset_index(drop=True)

    for idx, row in result.iterrows():
        ax.bar(idx, row['endpoint_variance'], color=colors[idx])

    plt.rcParams.update({'font.size': fontsize})
    plt.title(titlename)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Reward", fontsize=fontsize)
    plt.ylabel("Endpoint variance", fontsize=fontsize)
    plt.xticks([])
    plt.savefig(f"endpoint_variance/endpoint_variance_{figname}.png", bbox_inches="tight")
    plt.show()