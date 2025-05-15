import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

def plot_velocities_combinations(boni, effort_models, distances, DIRNAME_SIMULATION):

    for bonus in boni:
        for effort_model in effort_models:
            for distance in distances:
                filename = f"mobl_arms_index_tracking_{bonus}_bonus_{distance}_{effort_model}/evaluate"
                filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}")
                with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
                    data = pickle.load(f)
                
                all_velocities = []
                
                for episode, episode_data in data.items():
                    print(len(episode_data['timestep']))

                    if episode != "episode_0":
                        break

                    times = episode_data['timestep']
                    timestep = times[1] - times[0]
                    target_positions = episode_data["target_position"]
                    end_effector_positions = episode_data["hand_2distph_xpos"]

                    for idx in range(1,len(times)):

                        normalizedCentroidVector = np.abs(end_effector_positions[idx] - target_positions[idx])/np.linalg.norm(end_effector_positions[idx] - target_positions[idx]).transpose()
                   
                        end_effector_vel = np.abs(end_effector_positions[idx] - end_effector_positions[idx-1])/timestep
                        centroidVelProjection = (end_effector_vel * normalizedCentroidVector).sum()
                        all_velocities.append(centroidVelProjection)
                
                    movement_distances = np.linalg.norm(np.diff(end_effector_positions, axis=0)  , axis=1)  
                    total_movement_distance = np.sum(movement_distances)
                    shortest_distances = np.linalg.norm(np.diff(target_positions[1:], axis=0)  , axis=1)  
                    total_shortest_distance = np.sum(shortest_distances) + np.linalg.norm(end_effector_positions[0] - target_positions[1])

                    NPE = (total_movement_distance - total_shortest_distance)/total_movement_distance

                plt.title(f"{effort_model}_{distance}_first_target_all_episodes_speed")
                plt.xlabel("Time (s)")
                plt.ylabel("Speed (m/s)")
                #plt.legend()
                #plt.show()
                #plt.clf()
                smoothed_velocities = savgol_filter(all_velocities, 11, 3)
                return smoothed_velocities, NPE

def return_peaks(velocities, NPE):

    peaks, _ = find_peaks(velocities, prominence=0.05)
    valleys, _ = find_peaks(-velocities, prominence=0.05)

    print('Number of Peaks', len(peaks),'Number of Valleys',  len(valleys), "NPE", NPE)
    plt.plot(np.arange(0, 10, 0.05), velocities)
    plt.plot(np.arange(0, 10, 0.05)[peaks], velocities[peaks], "x", label="Maxima")
    plt.plot(np.arange(0, 10, 0.05)[valleys], velocities[valleys], "o", label="Minima")
    plt.show()

if __name__ == '__main__':
    all_velocities, NPE = plot_velocities_combinations(["hit"], ["zero_effort"], ["dist"], "../simulators/")
    return_peaks(all_velocities, NPE)