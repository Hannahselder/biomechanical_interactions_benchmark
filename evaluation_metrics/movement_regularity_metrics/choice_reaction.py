import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

shoulder_pos = np.array([0.02, -0.31, 0.938])
target_positions = {"button-0":  np.array([0.41, -0.07, -0.15]) + shoulder_pos, "button-1": np.array([0.41, 0.07, -0.15]) + shoulder_pos, "button-2": np.array([0.5, -0.07, -0.05]) + shoulder_pos, "button-3": np.array([0.5, 0.07, -0.05]) + shoulder_pos}

def plot_velocities_combinations(boni, effort_models, distances, DIRNAME_SIMULATION):

    for bonus in boni:
        for effort_model in effort_models:
            for distance in distances:
                if distance == "no":
                    filename = f"mobl_arms_index_choice_reaction_{bonus}_{effort_model}/evaluate_1"
                else:
                    filename = f"mobl_arms_index_choice_reaction_{bonus}_{distance}_{effort_model}/evaluate_1"
                filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}")
                with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
                    data = pickle.load(f)

                all_velocities, all_times, all_NPEs = plot_first_button(data, effort_model, distance)
                #times, velocities, NPEs = plot_all_button_combis(data, effort_model, distance)

                RTPs = []
                sub_movement_counts = []

                for i, vel in enumerate(all_velocities):
                    vel = np.array(vel)
                    all_times[i] = np.array(all_times[i])
                    peaks, _ = find_peaks(vel, prominence=0.01)
                    valleys, _ = find_peaks(-vel, prominence=0.01)

                    if len(valleys) > 0:
                        RTP = (all_times[i][-1] - all_times[i][valleys[0]])/all_times[i][-1]
                        plt.plot(all_times[i][peaks], vel[peaks], "x", label="Maxima")
                        plt.plot(all_times[i][valleys], vel[valleys], "o", label="Minima")
                        count = count_submovements(valleys, peaks)
                        RTPs.append(RTP)
                        sub_movement_counts.append(count)

                    plt.plot(all_times[i], vel)

                return RTPs, sub_movement_counts
                
def plot_all_button_combis(data, effort_model, distance):
    
    transitions = {}  
    
    for episode, episode_data in data.items():
        indices_buttons_generated = [i for i, generated in enumerate(episode_data['new_button_generated']) if generated]
        #print(episode_data.keys())
        all_velocities = []
        all_times = []
        times = episode_data['timestep']
        timestep = times[1] - times[0]
        button_sequence = episode_data["current_button"]  # Liste der gedrückten Buttons
        button_pos = [target_positions[button] for button in button_sequence]
        end_effector_positions = episode_data["hand_2distph_xpos"]
        new_buttons = episode_data['new_button_generated']

        velocities = []
        NPEs = []
        idx_second_button = new_buttons.index(True)
        prev_button = button_sequence[0]  # Start-Button
        normalizedCentroidVector = np.abs(end_effector_positions[idx_second_button] - button_pos[idx_second_button]) / np.linalg.norm(end_effector_positions[idx_second_button] - button_pos[idx_second_button]).transpose()
        k = 0
        for current_timestep in range(idx_second_button+1, len(times)):
            
            if new_buttons[current_timestep]: #zweiter button wird getroffen -> angezeigter button ist
                pressed_button = button_sequence[current_timestep-1] #button, der getroffen wurde
                transition_key = f"{prev_button}→{pressed_button}"
                if episode_data["target_hit"][current_timestep] == True:
                    if transition_key not in transitions:
                        transitions[transition_key] = []
                    
                    transitions[transition_key].append((episode_data['timestep'][0:len(velocities)], velocities))
                    all_velocities.append(velocities)
                    all_times.append(episode_data['timestep'][0:len(velocities)])

                    total_movement_distance = np.sum(np.linalg.norm(np.diff(end_effector_positions[indices_buttons_generated[k]: indices_buttons_generated[k+1]], axis=0)  , axis=1))
                    #print(end_effector_positions[indices_buttons_generated[k]], end_effector_positions[indices_buttons_generated[k]-1], target_positions[pressed_button])
                    shortest_distance = np.linalg.norm(target_positions[pressed_button] - end_effector_positions[indices_buttons_generated[k]]) - 0.15
                    NPE = (total_movement_distance - shortest_distance)/total_movement_distance
                    k += 1
                    NPEs.append(NPE)

                velocities = []
                normalizedCentroidVector = np.abs(end_effector_positions[current_timestep] - button_pos[current_timestep]) / np.linalg.norm(end_effector_positions[current_timestep] - button_pos[current_timestep]).transpose()
                prev_button = pressed_button

            end_effector_vel = np.abs(end_effector_positions[current_timestep] - end_effector_positions[current_timestep - 1]) / timestep
            centroidVelProjection = (end_effector_vel * normalizedCentroidVector).sum()
            velocities.append(centroidVelProjection)

    for transition, velocity_data in transitions.items():
        plot_button_combis(transition, velocity_data, effort_model, distance)
    
    return all_times, all_velocities, NPEs

def plot_button_combis(transition, velocity_data, effort_model, distance):
    """Creates plot for special button-combis with all episodes."""

    for i, (time_steps, velocities) in enumerate(velocity_data):
        window_length = 15
        if len(velocities) < 15:
          continue
        plt.plot(time_steps, savgol_filter(velocities, window_length, 3), label=f"Episode {i}")

    plt.title(f"Transition: {transition} |{effort_model},{distance}")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.savefig(f"./{effort_model}_{distance}_{transition}_speed.png")
    plt.clf()

def plot_first_button(data, effort_model, distance):
    all_velocities = []
    all_times = []
    NPEs = []

    for episode, episode_data in data.items():
        indices_buttons_generated = [i for i, generated in enumerate(episode_data['new_button_generated']) if generated]
        pressed_button = episode_data["current_button"][0] 
        times = episode_data['timestep']
        timestep = times[1] - times[0]
        button_pos = [target_positions[button] for button in episode_data["current_button"]]
        end_effector_positions = episode_data["hand_2distph_xpos"]
        new_buttons = episode_data['new_button_generated']
        first_button = episode_data["current_button"][0]
        velocities = [0]
        normalizedCentroidVector = np.abs(end_effector_positions[0] - button_pos[0])/np.linalg.norm(end_effector_positions[0] - button_pos[0]).transpose()
        idx_next_button = new_buttons.index(True)
        for current_timestep in range(1,idx_next_button):
        
            end_effector_vel = np.abs(end_effector_positions[current_timestep] - end_effector_positions[current_timestep-1])/timestep
            centroidVelProjection = (end_effector_vel * normalizedCentroidVector).sum()
            velocities.append(centroidVelProjection)   
        
        if episode_data["target_hit"][idx_next_button] == True:
            smoothed_velocities = savgol_filter(velocities, 8, 4)
            plt.plot(times[:idx_next_button], smoothed_velocities, label=f"Episode {episode}")  
            all_velocities.append(smoothed_velocities)
            all_times.append(episode_data['timestep'][0:len(velocities)])

            total_movement_distance = np.sum(np.linalg.norm(np.diff(end_effector_positions[indices_buttons_generated[0]: indices_buttons_generated[1]], axis=0)  , axis=1))
            #print(end_effector_positions[indices_buttons_generated[k]], end_effector_positions[indices_buttons_generated[k]-1], target_positions[pressed_button])
            shortest_distance = np.linalg.norm(target_positions[pressed_button] - end_effector_positions[indices_buttons_generated[0]]) - 0.15
            NPE = (total_movement_distance - shortest_distance)/total_movement_distance
            NPEs.append(NPE)


    plt.title(f"{effort_model}_{distance}_first_button_all_episodes_speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    #plt.legend()
    plt.savefig(f"speed_plots/{effort_model}_{distance}_first_button_all_episodes_speed.png")  
    plt.clf()
    return all_velocities, all_times, NPEs
    
def count_submovements(valleys, peaks):
    if 0 not in valleys:
        valleys = np.insert(valleys, 0, 0)
    extrema = np.sort(np.concatenate([peaks, valleys]))
    types = ['max' if i in peaks else 'min' for i in extrema]
    count = sum(types[i:i+3] == ['min', 'max', 'min'] for i in range(len(types)-2))
    return count

if __name__ == '__main__':

    effort_models = ["zero_effort", "dc_effort_w1", "jac_effort_w1", "ctc_effort_w1", "armmovementpaper_effort"]
    RTPs, sub_movement_counts = plot_velocities_combinations(["no_bonus"], ["zero_effort"], ["dist"], "../../simulators/")
    RTPs2, sub_movement_counts2 = plot_velocities_combinations(["hit_bonus"], effort_models, ["dist"], "../../simulators/")
    RTPs3, sub_movement_counts3 = plot_velocities_combinations(["hit_bonus_8"], ["zero_effort"], ["no"], "../../simulators/")

    plt.plot()


