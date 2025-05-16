import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress
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

def plot_distance_to_target(plot_type, legend_title, figname, legend_labels, colors, markers, data_df):

    fig, ax = plt.subplots(figsize=figsize)
    effort_models = []
    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['distance_to_target'].mean()

    for idx, row in result.iterrows():
        if row['bonus'] == 8:
            row['bonus'] = 'hit_bonus'
        ax.plot(idx, row['distance_to_target'], color=colors[row['distance']], marker=markers[row['bonus']], markersize=12)
        effort_models.append(row["effort_model"])

    plt.rcParams.update({'font.size': fontsize})
    plt.legend(handles=legend_labels, title=legend_title, prop={'size': legend_fontsize})#, bbox_to_anchor=(0.46,0.55))
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)

    ax.set_yscale('log')
    ax.set_yticks([0.1,0.25,0.5])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Average distance to target (m)", fontsize=fontsize)
    plt.title('Tracking Task')
    plt.savefig(f'distance_to_target_plots/{figname}')

def plot_time_inside_target(plot_type, figname, colors, markers, data_df):
    
    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['time_inside_target'].mean()
    effort_models = []
    plt.figure(figsize=figsize)
    for idx, row in result.iterrows():
        if row['bonus'] == 8:
            row['bonus'] = 'hit_bonus'
        print()
        plt.plot(idx, row['time_inside_target'], color=colors[row['distance']], marker=markers[row['bonus']], markersize=12)
        effort_models.append(row["effort_model"])

    plt.rcParams.update({'font.size': fontsize})
    plt.yticks(fontsize=fontsize)
    plt.title('Tracking Task')
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Time inside target (%)", fontsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig(f'time_inside_target_plots/{figname}')

def plot_success_rate(plot_type, figname, legend_labels, colors, markers, data_df, sparse_df=None):
    plt.figure(figsize=figsize)

    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['success_rate'].mean()
    effort_models = []

    for idx, row in result.iterrows():
        if row['bonus'] == 8:
            row['bonus'] = 'hit_bonus'
        plt.plot(idx, row['success_rate'] / 10 * 100, color=colors[row['distance']], marker=markers[row['bonus']], markersize=12)
        effort_models.append(row["effort_model"])

    plt.rcParams.update({'font.size': fontsize})
    plt.title('Pointing Task')
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Success rate (%)", fontsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.legend(handles=legend_labels, title="Reward components", prop={'size': legend_fontsize})#, bbox_to_anchor=(0.46,0.55))
    plt.savefig(f'success_rate_plots/{figname}')

def plot_success_rate_remote_control(df):
    plt.figure(figsize=(9, 4))
    plt.rcParams.update({'font.size': 16})
    colors = ['b','g', 'r', 'tab:orange']
    bars = plt.bar(range(len(df)), df["success"]*100, color=colors)
    plt.ylabel("Success Rate (%)")
    plt.xticks([]) 
    plt.title("Remote Control Task")

    for i, (bar, value) in enumerate(zip(bars, df["success"])):
        text_color = colors[i] if abs(value) < 0.01 else 'black' 
        if abs(value) < 0.01:
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value*100:.1f}%", ha='center', fontsize=20, color=text_color)

    plt.legend(bars, df["reward"], prop={'size': 12},title = 'Reward function')#, loc="upper left"
    plt.savefig('success_rate_plots/success_rate_remote_control.png')
    plt.show()

def plot_success_rate_choice_reaction(data_df, colors):

    legend_labels = []
    for key in colors.keys():
        legend_labels.append(mlines.Line2D([0], [0], color=colors[key], lw=4, label=key))
    fig, ax = plt.subplots(figsize=figsize)

    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['success_rate'].mean()
    effort_models = []

    for idx, row in result.iterrows():
        if row['bonus'] == "no_bonus":
            color = colors['Distance']
        elif row['bonus'] == 8:
            color = colors['Bonus']
        else:
            if row['effort_model'] == 'zero_effort':
                color = colors['Distance + Bonus']
            else:
                color = colors['Distance + Bonus + Effort Model']

        ax.bar(idx, row['success_rate'], color=color)
        effort_models.append(row["effort_model"])

    plt.rcParams.update({'font.size': fontsize})
    plt.title('Choice Reaction Task')
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Success rate(%)", fontsize=fontsize)
    plt.legend(handles=legend_labels, title="Reward components", prop={'size': 12}, loc = 'lower left')
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig("success_rate_plots/success_rates_choice_reaction.png", bbox_inches="tight")
    plt.show()

def plot_task_completion_time_choice_reaction(data_df, colors):
    fig, ax = plt.subplots(figsize=(7, 5))
    result = data_df.groupby(['effort_model', 'distance', 'bonus'], as_index=False)['task_completion_time'].mean()
    effort_models = []

    for idx, row in result.iterrows():
        if row['bonus'] == "no_bonus":
            color = colors['Distance']
        elif row['bonus'] == 8:
            color = colors['Bonus']
        else:
            if row['effort_model'] == 'zero_effort':
                color = colors['Distance + Bonus']
            else:
                color = colors['Distance + Bonus + Effort Model']
        ax.bar(idx, row['task_completion_time'], color=color)
        effort_models.append(row["effort_model"])


    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    ax.set_xticks(indices) 
    ax.set_xticklabels(effort_names)
    ax.set_xlabel('Effort Model', fontsize=fontsize)
    ax.set_ylabel('Task Completion Time (s)', fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_yticks([0.4, 0.6, 1, 1.8])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title('Choice Reaction Task')
    plt.savefig("task_completion_time_plots/task_completion_time_choice_reaction.png")
    plt.show()

def plot_subtask_completion_times(df):
    rewards = df['reward']
    colors = ['b','g', 'r', 'tab:orange']
    x = np.arange(len(df)) 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    bars1 = ax1.bar(x, df['joystick_time'], color=colors)
    ax1.set_title('Joystick Time')
    ax1.set_ylabel('Subtask Completion Time (s)')
    ax1.set_xticklabels([])

    bars2 = ax2.bar(x, df['target_time'], color=colors)
    ax2.set_title('Target Time')
    ax2.set_xticklabels([])

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[i], label=rewards[i]) for i in range(len(rewards))]
    fig.legend(handles=legend_handles, title="Reward", loc='lower center', ncol=len(rewards), bbox_to_anchor=(0.5,-0.15))

    plt.show()

def plot_subtask_cum_distance(df):
    rewards = df['reward']
    colors = ['b','g', 'r', 'tab:orange']
    x = np.arange(len(df)) 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    bars1 = ax1.bar(x, df['cum_dist_to_joystick'], color=colors)
    ax1.set_title('Joystick Distance')
    ax1.set_ylabel('Cumulative Distance (m)')
    ax1.set_xticklabels([])

    bars2 = ax2.bar(x, df['cum_dist_to_target'], color=colors)
    ax2.set_title('Target Distance')
    ax2.set_xticklabels([])

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[i], label=rewards[i]) for i in range(len(rewards))]
    fig.legend(handles=legend_handles, title="Reward", loc='lower center', ncol=len(rewards), bbox_to_anchor=(0.5,-0.15))

    plt.show()

def plot_deviation_count(plot_type, figname, colors, markers, data_df):
    plt.figure(figsize=figsize)
    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['deviation_count'].mean()
    effort_models = []

    for idx, row in result.iterrows():
        if row['bonus'] == 8:
            row['bonus'] = 'hit_bonus'
        plt.plot(idx, row['deviation_count'] / 10 * 100, color=colors[row['distance']], marker=markers[row['bonus']], markersize=12)
        effort_models.append(row["effort_model"])

    plt.rcParams.update({'font.size': fontsize})
    if plot_type == "tracking":
        plt.yscale("log") 
    plt.yticks(fontsize=fontsize)
    plt.title(f'{plot_type} Task')
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Number of re-entries", fontsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig(f'deviation_count_plots/{figname}')

def plot_deviation_count_remote_control(figname, data_df):
    
    colors = ['b','g', 'r', 'tab:orange']
    plt.figure(figsize=figsize)
    for i in range(len(data_df)):
        df = data_df.iloc[i]
        plt.bar(i, df['deviation_count'], label=df["reward"], color=colors[i])

    plt.rcParams.update({'font.size': fontsize})
    plt.yticks(fontsize=fontsize)
    plt.title(f'Remote Control Task')
    plt.xticks([])
    plt.xlabel("Reward", fontsize=fontsize)
    plt.ylabel("Number of re-entries", fontsize=fontsize)
    plt.legend(title="Reward")
    plt.savefig(f'deviation_count_plots/{figname}')

def plot_cum_dist(plot_type, figname, colors, markers, data_df):
    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['cum_distance'].mean()
    effort_models = []
    plt.figure(figsize=figsize)
    for idx, row in result.iterrows():
        if row['bonus'] == 8:
            row['bonus'] = 'hit_bonus'
        plt.plot(idx, row['cum_distance'], color=colors[row['distance']], marker=markers[row['bonus']], markersize=markersize)
        effort_models.append(row["effort_model"])


    plt.rcParams.update({'font.size': fontsize})
    if plot_type == "tracking":
        plt.yscale("log") 
    plt.yticks(fontsize=fontsize)
    plt.title(f'{plot_type} Task')
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Cumulative distance", fontsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig(f'cumulative_distance_plots/{figname}')

def plot_end_point_dist(plot_type, figname, colors, markers, data_df):
    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['end_point_distances'].mean()
    effort_models = []
    plt.figure(figsize=figsize)
    for idx, row in result.iterrows():
        if row['bonus'] == 8:
            row['bonus'] = 'hit_bonus'
        plt.plot(idx, row['end_point_distances'], color=colors[row['distance']], marker=markers[row['bonus']], markersize=markersize)
        effort_models.append(row["effort_model"])


    plt.rcParams.update({'font.size': fontsize})
    if plot_type == "tracking":
        plt.yscale("log") 
    plt.yticks(fontsize=fontsize)
    plt.title(f'{plot_type} Task')
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Endpoint distance (m)", fontsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig(f'end_point_distance_plots/{figname}')

def plot_end_point_dist_choice_reaction(plot_type, figname, colors, markers, data_df):
    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['end_point_distances'].mean()
    effort_models = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, row in result.iterrows():
        if row['bonus'] == "no_bonus":
            color = colors['Distance']
        elif row['bonus'] == 8:
            color = colors['Bonus']
        else:
            if row['effort_model'] == 'zero_effort':
                color = colors['Distance + Bonus']
            else:
                color = colors['Distance + Bonus + Effort Model']
        if row['bonus'] == 8:
            row['bonus'] = 'hit_bonus'
        ax.bar(idx, row['end_point_distances'], color=color)
        effort_models.append(row["effort_model"])


    plt.rcParams.update({'font.size': fontsize})
    if plot_type == "tracking":
        plt.yscale("log") 
    plt.yticks(fontsize=fontsize)
    plt.title(f'{plot_type} Task')
    plt.xlabel("Effort model", fontsize=fontsize)
    plt.ylabel("Endpoint distance (m)", fontsize=fontsize)
    effort_names = [effort_map[e] for e in effort_models]
    indices = np.arange(0, len(effort_names), 1)
    plt.xticks(ticks=indices, labels=effort_names, fontsize=fontsize)
    plt.savefig(f'end_point_distance_plots/{figname}', bbox_inches="tight")

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

def moved_indices(plot_type, bonus, distance, effort_model):
    if plot_type == "tracking" or plot_type == "pointing" or plot_type == "choice_reaction":
        return moved_distance_indices(bonus, distance, effort_model)
    elif plot_type == "tracking_dist_sens":
        return moved_distance_sensitivity_index(bonus, distance, effort_model)
    elif plot_type == "tracking_bonus_sens":
        return moved_bonus_sensitivity_index(bonus, distance, effort_model)
    elif plot_type == "tracking_effort_sens":
        return moved_effort_sensitivity_index(bonus, distance, effort_model)
    else:
        return None

def moved_distance_indices(bonus, distance, effort_model):

    effort_models = ["zero_effort", "dc_effort_w1", "jac_effort_w1", "ctc_effort_w1", "armmovementpaper_effort"]
    effort_idx = effort_models.index(effort_model)

    if distance == "dist":
        x_jitter = effort_idx - 0.1
    elif distance == "exp_dist":
        x_jitter = effort_idx + 0.1
    else:
        x_jitter = effort_idx

    return x_jitter

def moved_distance_sensitivity_index(bonus, distance, effort_model):

    effort_models = ["zero_effort", "dc_effort_w1", "jac_effort_w1", "ctc_effort_w1", "armmovementpaper_effort"]
    distances = ['exp_dist_w_d_10', 'exp_dist_w_d_5', 'exp_dist_w_t_5', 'exp_dist_w_t_10', 'exp_dist']

    effort_idx = effort_models.index(effort_model)
    distance_idx = distances.index(distance)

    if distance_idx == 0:
        idx = effort_idx - 0.2
    elif distance_idx == 1:
        idx = effort_idx - 0.1
    elif distance_idx == 2:
        idx = effort_idx + 0.1
    elif distance_idx == 3:
        idx = effort_idx + 0.2
    else:
        idx = effort_idx

    return idx

def moved_distance_indices_pointing(bonus, distance, effort_model):

    effort_models = ["zero_effort", "dc_effort_w1", "jac_effort_w1", "ctc_effort_w1", "armmovementpaper_effort"]
    effort_idx = effort_models.index(effort_model)

    if distance == "dist":
        x_jitter = effort_idx - 0.1
    elif distance == "exp_dist":
        x_jitter = effort_idx + 0.1
    else:
        x_jitter = effort_idx

    return x_jitter

def moved_bonus_sensitivity_index(bonus, distance, effort_model):

    effort_models = ["zero_effort", "dc_effort_w1", "jac_effort_w1", "ctc_effort_w1", "armmovementpaper_effort"]
    effort_idx = effort_models.index(effort_model)

    if bonus =="hit_bonus":
        idx = effort_idx +0.25
    elif bonus == "no_bonus":
        idx = effort_idx - 0.25
    elif bonus =="hit_bonus_0_125":
        idx = effort_idx - 0.15
    elif bonus =="hit_bonus_0_25":
        idx = effort_idx - 0.05
    elif bonus =="hit_bonus_0_5":
        idx = effort_idx + 0.05
    else:
        idx = effort_idx + 0.15

    return idx

def moved_effort_sensitivity_index(bonus, distance, effort_model):

    effort_components = effort_model.split('_')
    if len(effort_components) < 5:
        prefactor = None
        factor = None
        effort = effort_components[0]
    else:
        effort = effort_components[0]
        prefactor = effort_components[3]
        factor=int(effort_components[4])

    effort_models = ["dc", "jac", "ctc", "armmovementpaper"]
    effort_idx = effort_models.index(effort)
    
    if prefactor == "d":
        t = -1
    elif prefactor == "t":
        t = 1
    else:
        t = 0

    if factor == 10:
        eps = 0.1
    elif factor == 20:
        eps = 0.2
    else:
        eps = 0
    
    
    idx = effort_idx + t*eps

    return idx

def calculate_task_completion_times_df(data_df, number_of_episodes):
    data = []
    for bonus in ['hit_bonus']:
        for effort_model in data_df['effort_model'].unique():
            for distance in data_df['distance'].unique():
                df = data_df[
                    (data_df["effort_model"] == effort_model) & 
                    (data_df["distance"] == distance) &
                    (data_df['bonus']== 'hit_bonus')
                ]
                time = 0
                for index, row in df.iterrows():
                    time += row['task_completion_time'].sum()

                task_completion_time = time/(10* number_of_episodes)

                df.loc[
                    (df["effort_model"] == effort_model) & 
                    (df["distance"] == distance) &
                    (df['bonus']== 'hit_bonus')
                ]['task_completion_time_cal'] = task_completion_time
                data.append([effort_model,distance, task_completion_time])
    r_df = pd.DataFrame(data, columns=["effort_model", "distance", "task_completion_time"])
    return r_df

def plot_task_completion_time_pointing(data_df, colors):
    fig, ax = plt.subplots(figsize=(7, 5))
    result = data_df.groupby(['effort_model', 'bonus', 'distance'], as_index=False)['task_completion_time'].mean()
    effort_models = []

    for idx, row in result.iterrows():
        if row['bonus'] == 8:
            row['bonus'] = 'hit_bonus'
        ax.bar(idx, row['task_completion_time'], color=colors[row['distance']],  width=0.2)
        effort_models.append(row["effort_model"])

    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    indices = np.arange(0, len(effort_models), 1)
    effort_names = [effort_map[e] for e in effort_models]
    ax.set_xticks(ticks=indices, labels=effort_names, fontsize=fontsize)  # Setzt die Positionen für die Gruppennamen
    ax.set_xticklabels(effort_names)

    ax.set_xlabel('Effort Model', fontsize=fontsize)
    ax.set_ylabel('Task Completion Time (s)', fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_yticks([0.3, 0.4,0.6, 1, 2])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title('Pointing Task', fontsize=16)
    plt.savefig("task_completion_time_plots/task_completion_time_pointing.png")#, bbox_inches="tight", pad_inches=0.1)
