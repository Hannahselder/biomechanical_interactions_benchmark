# Biomechanical interaction benchmark
This repository provides a benchmark suite for training and evaluating biomechanical reinforcement learning agents on interactive, motor control-centric tasks relevant to human-computer interaction. The suite includes four core tasks—pointing, tracking, choice reaction, and remote control—along with systematic task variations to assess agent generalization, and a set of evaluation metrics to quantify agent performance.

---

## Repository Structure

### `tasks/`
This folder contains implementations of the four benchmark tasks:

- **Pointing**
- **Tracking**
- **Choice Reaction**
- **Remote Control**

Each task includes a set of **systematic variations** designed to challenge the generalization ability of RL agents:


- **Pointing**:  
  - [Transverse](tasks/pointing_horizontal_target_area): target area placed in the horizontal/transverse plane (rather than frontal plane)   
  - [Small](tasks/pointing_smaller_targets): Smaller target area, smaller target sizes  
  - [Reach Limits](tasks/pointing_reach_envelope): Targets are placed at 90% of the model's reach limit  
  - [Distractor](tasks/pointing_distractor): added a second "distractor" sphere  

- **Tracking**:  
  - [3D](tasks/tracking_3D_target_area): 3D random target movements (rather than 2D within frontal plane)  

- **Choice Reaction**:  
  - [Randmap](tasks/choice_reaction_distance_sensor_mixed_button_positions): Button colors randomly assigned at each trial 
  - [6Buttons](tasks/choice_reaction_distance_sensor_6_buttons): added two buttons (6 instead of 4)  
  - [Force](tasks/choice_reaction_distance_sensor_force_constraint): increased contact force that is required to select a button  

- **Remote Control**:  
  - [Depth Car](tasks/remote_driving_depth_direction): Car moves in depth direction (rather than horizontal direction)  
  - [LR Joystick](tasks/remote_driving_lr_joystick): Left-right joystick movements required to accelerate/decelerate the car (rather than up/down) 

The default task implementations are based on the [User-in-the-Box repository](https://github.com/User-in-the-Box/user-in-the-box).

---
### `simulators/`

Contains pre-trained policies for the default versions of each task, trained using different reward configurations. Each simulator subfolder provides:

- Trained policy checkpoints  
- Evaluation results on both default and variations tasks (evaluate folder)
- Compatibility with the [User-in-the-Box framework](https://github.com/User-in-the-Box/user-in-the-box), allowing to continue training or integration into new experiments

---

### `evaluation_metrics/`

This repository includes tools for assessing policies using metrics from three key categories:

- **Task Success**  
- **Movement Regularity**  
- **Training Efficacy & Efficiency**

These metrics allow for a comparison of different policies on one task or one policy on the task variations. 

Example notebooks:

- [`task_success_metrics/choice_reaction.ipynb`](evaluation_metrics/task_success_metrics/choice_reaction.ipynb): evaluates policy performance on the default Choice Reaction task  
- [`task_success_metrics/choice_reaction_variants.ipynb`](evaluation_metrics/task_success_metrics/choice_reaction_variations.ipynb): evaluates the same policy on all task variations
