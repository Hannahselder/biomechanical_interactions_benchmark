import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress

def index_at_threshold(rewards, threshold):
    for i, r in enumerate(rewards):
        if r >= threshold:
            return i

def path_to_event_file(event_folder):
    latest_ppo_folder = max(
    (os.path.join(event_folder, f) for f in os.listdir(event_folder) if f.startswith('PPO') and os.path.isdir(os.path.join(event_folder, f))),
    key=os.path.getmtime, default=None
    )
    if latest_ppo_folder:
        event_file = max(
            (os.path.join(latest_ppo_folder, f) for f in os.listdir(latest_ppo_folder)),
            key=os.path.getmtime, default=None
        )
        if event_file:
            print(f"Event file: : {event_file}")
        else:
            print("No event file found.")
    else:
        print("No PPO folder found.")

    return event_file

def read_event_file(event_folder):

    event_file = path_to_event_file(event_folder)
    rewards = []
    steps = []
    
    for e in tf.compat.v1.train.summary_iterator(event_file):
        for v in e.summary.value:
            if v.tag == 'rollout/ep_rew_mean' and e.step <= 35000000: 
                #print(f"rollout/ep_rew_mean at step {e.step}: {v.simple_value}")
                rewards.append(v.simple_value)
                steps.append(e.step) 
    return steps, rewards

def print_reward_threshholds(steps, rewards):
    max_reward = max(rewards)
    threshold2_index = index_at_threshold(rewards, 0.2 * max_reward)
    threshold5_index = index_at_threshold(rewards, 0.5 * max_reward)
    threshold9_index = index_at_threshold(rewards, 0.9 * max_reward)
    max_index = index_at_threshold(rewards, max_reward)
    print(f'Threshold Index 20%: {steps[threshold2_index]} steps, 50%: {steps[threshold5_index]} steps, 90%: {steps[threshold9_index]} steps')
    #print(f'Maximum Index: {steps[max_index]} steps')

    normalized_rewards = (np.array(rewards) - min(rewards)) / (max(rewards) - min(rewards))
    auc_normalized = np.trapezoid(normalized_rewards)
    print(f'Area under the curve: {auc_normalized}')

    sub_rewards = rewards[max_index:]
    mean_reward = np.mean(sub_rewards)
    max_deviation = max(abs(sub_rewards - mean_reward))
    print(f'Mean value starting from convergence point: {mean_reward} and max deviation: {max_deviation}')

def variance_last_steps(steps, rewards):
    steps_idx = steps.index(30000000)
    sub_rewards = rewards[steps_idx:]

    variance = np.var(sub_rewards)
    print(f"Variance from 30M steps on: {variance}")

    x = np.arange(len(sub_rewards))
    slope, intercept, r_value, p_value, std_err = linregress(x, sub_rewards)

    if slope > 0:
        print(f"Reward increases in trend (slope = {slope:.4f})")
    else:
        print(f"Reward does not increase in trend (slope = {slope:.4f})")

def plot_reward(steps, rewards, task, figname):
    steps, rewards = zip(*sorted(zip(steps, rewards)))
    plt.plot(steps, (np.array(rewards) - min(rewards)) / (max(rewards) - min(rewards)))
    plt.xlabel('Global Step')
    plt.ylabel('Reward')
    plt.title(task)
    plt.savefig(f"reward_plots/{figname}.png")
    plt.show()