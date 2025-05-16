import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

def plot_velocities_combinations(boni, effort_models, distances, DIRNAME_SIMULATION, variant=None):

    for bonus in boni:
        for effort_model in effort_models:
            for distance in distances:
                filename = f"mobl_arms_index_pointing_{bonus}_bonus_{distance}_{effort_model}/evaluate"
                if variant != None:
                    filename += "_" + variant
                filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}_1")
                with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
                    data = pickle.load(f)
                
                all_velocities = []
                all_times = []
                movement_distances = []
                
                for episode, episode_data in data.items():
                    times = episode_data['timestep']
                    timestep = times[1] - times[0]
                    target_positions = episode_data["target_position"]
                    end_effector_positions = episode_data["hand_2distph_xpos"]
                    new_target_indices = [0] + [i for i, spawned in enumerate(episode_data['target_spawned']) if spawned]
                    RSs = []

                    for i, idx in enumerate(new_target_indices[:-1]):
                        velocities = []
                        normalizedCentroidVector = np.abs(end_effector_positions[idx] - target_positions[idx])/np.linalg.norm(end_effector_positions[idx] - target_positions[idx]).transpose()

                        for current_timestep in range(idx+1,new_target_indices[i+1]+1):                       
                            end_effector_vel = np.abs(end_effector_positions[current_timestep] - end_effector_positions[current_timestep-1])/timestep
                            centroidVelProjection = (end_effector_vel * normalizedCentroidVector).sum()
                            velocities.append(centroidVelProjection)   
                        
                        smoothed_velocities = savgol_filter(velocities, 11, 3)
                        all_velocities.append(smoothed_velocities)
                        all_times.append(np.array(times[idx:new_target_indices[i+1]]) - times[idx])

                        if i == 0:
                            valleys, _ = find_peaks(-smoothed_velocities, prominence=0.05)
                            if len(valleys)> 1:
                                total_movement_distance = np.sum(np.linalg.norm(np.diff(np.array(end_effector_positions)[:idx,:], axis=0)  , axis=1))
                                ballistic_distance = np.sum(np.linalg.norm(np.diff(np.array(end_effector_positions)[:valleys[0],:], axis=0)  , axis=1))
                                RS = ballistic_distance
                                RSs.append(RS)


                plt.title(f"{effort_model}_{distance}_first_target_all_episodes_speed")
                plt.xlabel("Time (s)")
                plt.ylabel("Speed (m/s)")
                return all_times, all_velocities, RSs

def return_peaks(all_times, all_velocities, variant=None):
    i = 0
    submovement_count = []
    RTPs = []
    for vel in all_velocities:
        peaks, _ = find_peaks(vel, prominence=0.05)
        valleys, _ = find_peaks(-vel, prominence=0.05)

        if all_times[i][-1] > 3.95:
            time_end = 4
        else:
            time_end = all_times[i][-1] - 0.1

        if len(valleys) > 0:
            RTP = (time_end - all_times[i][valleys[0]])/time_end
            plt.plot(all_times[i][peaks], vel[peaks], "x", label="Maxima")
            plt.plot(all_times[i][valleys], vel[valleys], "o", label="Minima")
            RTPs.append(RTP)
        if 0 not in valleys:
            valleys = np.insert(valleys, 0, 0)
        extrema = np.sort(np.concatenate([peaks, valleys]))
        types = ['max' if i in peaks else 'min' for i in extrema]
        count = sum(types[i:i+3] == ['min', 'max', 'min'] for i in range(len(types)-2))
        submovement_count.append(count)
        plt.plot(all_times[i], vel)
        i+= 1

    figname = "speed_plots/pointing_speed"
    if variant != None:
        figname += "_" + variant
    plt.savefig(figname)
    plt.clf()

    return RTPs, submovement_count

def plot_RTP(all_RTPs, variant_names, colors):
    fig, ax = plt.subplots(figsize=(6.8, 4))
    i = 0
    for key, value in all_RTPs.items():
        ax.bar(i, value, label=variant_names[i], color=colors[key])
        i += 1
    plt.title("Pointing variations")
    plt.yticks(fontsize=15)
    plt.xticks([])
    plt.xlabel("Task Variant", fontsize=15)
    plt.ylabel("Refinement time proportion (%)", fontsize=15)
    plt.savefig("RTP/RTP_pointing_variants.png")

def plot_RSP(all_RTPs, variant_names, colors):
    fig, ax = plt.subplots(figsize=(6.8, 4))
    i = 0
    for key, value in all_RTPs.items():
        ax.bar(i, value, label=variant_names[i], color=colors[key])
        i += 1
    plt.title("Pointing variations")
    plt.yticks(fontsize=15)
    plt.xticks([])
    plt.xlabel("Task Variant", fontsize=15)
    plt.ylabel("Refinement space proportion (%)", fontsize=15)
    plt.savefig("RSP/RSP_pointing_variants.png")

def plot_submovement_count(all_submovement_counts, variant_names, colors):
    fig, ax = plt.subplots(figsize=(6.8, 4))
    i = 0
    for key, value in all_submovement_counts.items():
        ax.bar(i, value, label=variant_names[i], color=colors[key])
        i += 1
    plt.legend(loc = "lower right")
    plt.title("Pointing variations")
    plt.yticks(fontsize=15)
    plt.xticks([])
    plt.xlabel("Task Variant", fontsize=15)
    plt.ylabel("Number of submovements", fontsize=15)
    plt.savefig("Nr_of_submovements/Nr_of_submovements_pointing_variants.png")

if __name__ == '__main__':
    all_RTPs = {}
    all_RSs = {}
    all_submovement_counts = {}
    variations = ["horizontal_target_area", "smaller_targets", "reach_envelope", "distractor"]
    all_variants = ["default"] + variations
    variant_names = ["Default", "Transverse", "Small targets", "Limits", "Mapping"]
    values = np.linspace(0, len(all_variants), len(all_variants)) 
    cmap = plt.get_cmap('plasma')
    colors = {key: cmap(v / 5) for key, v in zip(all_variants, values)}
    markers = {"no_bonus": "+", "hit_bonus": "o"}
    effort_models = ["zero_effort", "dc_effort_w1", "jac_effort_w1", "ctc_effort_w1", "armmovementpaper_effort"]
    plt.rcParams.update({'font.size': 15})

    all_times, all_velocities, RSs = plot_velocities_combinations(["hit"], ["zero_effort"], ["dist"], "../../simulators/")
    RTPs, submovement_count = return_peaks(all_times, all_velocities)
    all_RTPs["default"] = np.mean(RTPs)
    all_RSs["default"] = np.mean(RSs)
    all_submovement_counts["default"] = np.mean(submovement_count)
    
    for v in variations:
        all_times, all_velocities, RSs = plot_velocities_combinations(["hit"], ["zero_effort"], ["dist"], "../../simulators/", variant=v)
        RTPs, submovement_count = return_peaks(all_times, all_velocities, variant=v)
        all_RTPs[v] = np.mean(RTPs)
        all_RSs[v] = np.mean(RSs)
        all_submovement_counts[v] = np.mean(submovement_count)
    
    plot_RSP(all_RSs, variant_names, colors)
    plot_RTP(all_RTPs, variant_names, colors)
    plot_submovement_count(all_submovement_counts, variant_names, colors)
