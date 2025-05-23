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

    return event_file

def read_event_file(event_folder, steps_limit = 35000000):

    event_file = path_to_event_file(event_folder)
    rewards = []
    steps = []
    print(event_file)
    
    for e in tf.compat.v1.train.summary_iterator(event_file):
        for v in e.summary.value:
            if v.tag == 'rollout/ep_rew_mean' and e.step <= steps_limit: 
                #print(f"rollout/ep_rew_mean at step {e.step}: {v.simple_value}")
                rewards.append(v.simple_value)
                steps.append(e.step) 
    return steps, rewards

def calculate_reward_thresholds(steps, rewards, thresholds):

    max_reward = max(rewards)
    min_reward = min(rewards)
    reward_span = abs(max_reward - min_reward)

    threshold_steps = {}
    for t in thresholds:
        t_idx = index_at_threshold(rewards, min_reward + (t * reward_span))
        threshold_steps[t] = steps[t_idx]
        print(f"Threshold {int(t*100)}% after {steps[t_idx]} steps")

    normalized_rewards = (np.array(rewards) - min(rewards)) / (max(rewards) - min(rewards))
    auc_normalized = np.trapezoid(normalized_rewards)
    print(f'Area under the curve: {auc_normalized}')

    return threshold_steps, auc_normalized

def variance_last_steps(steps, rewards):
    steps_idx = len(steps)-126
    sub_rewards = (np.array(rewards[steps_idx:])  - min(rewards)) / (max(rewards) - min(rewards))
    variance = np.var(sub_rewards)
    print(len(sub_rewards), np.max(sub_rewards), np.min(sub_rewards))
    print(f"Variance from 30M steps on: {variance}")
    return variance

def plot_reward(steps, rewards, task, figname):
    steps, rewards = zip(*sorted(zip(steps, rewards)))
    plt.plot(steps, (np.array(rewards) - min(rewards)) / (max(rewards) - min(rewards)))
    plt.xlabel('Global Step')
    plt.ylabel('Reward')
    plt.title(task)
    plt.savefig(f"reward_plots/{figname}.png")
    plt.show()