import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

def plot_velocities_combinations(rewards, DIRNAME_SIMULATION):

    for reward in rewards:
        filename = f"mobl_arms_index_remote_driving_{reward}_evaluate/"
        filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}")
        with open(os.path.join(filepath, "state_log.pickle"), "rb") as f:
            data = pickle.load(f)
        
        all_velocities = []
        all_times = []
        NPEs = []
        
        for episode, episode_data in data.items():
            end_idx = episode_data['car_moving'].index(True)

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

            total_movement_distance = np.sum(np.linalg.norm(np.diff(end_effector_positions[0:end_idx], axis=0)  , axis=1))
            shortest_distance = np.linalg.norm(episode_data['joystick_xpos'][0] - end_effector_positions[0]) 
            NPE = (total_movement_distance - shortest_distance)/total_movement_distance
            NPEs.append(NPE)

        return all_times, all_velocities, NPEs

def return_peaks(all_times, velocities, NPEs):

    i = 0
    for vel in velocities:

        peaks, _ = find_peaks(vel, prominence=0.05)
        valleys, _ = find_peaks(-vel, prominence=0.05)

        if len(valleys) > 0:
            RTP = (all_times[i][-1] - all_times[i][valleys[0]])/all_times[i][-1]
            plt.plot(all_times[i][peaks], vel[peaks], "x", label="Maxima")
            plt.plot(all_times[i][valleys], vel[valleys], "o", label="Minima")
            print('Number of Peaks:', len(peaks),'Number of Valleys:',  len(valleys), "RTP: ", RTP, "NPE: ", NPEs[i])
        else:
            print('Number of Peaks', len(peaks),'Number of Valleys',  len(valleys), "NPE: ", NPEs[i])

        plt.plot(all_times[i], vel)
        i += 1

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.show()

if __name__ == '__main__':
    all_times, all_velocities, NPEs = plot_velocities_combinations(['distance_and_completion_bonus', 'distance_and_completion_bonus_and_joystick_reaching_bonus', 'only_distance', 'only_bonus'], "../simulators/remote_driving/")
    return_peaks(all_times, all_velocities, NPEs)