import numpy as np
import mujoco
import xml.etree.ElementTree as ET

from ..base import BaseTask
import sys
import importlib
from .reward_functions import RewardFactory

class ChoiceReaction_distance_sensor(BaseTask):

  def __init__(self, model, data, end_effector, shoulder, reward_function,**kwargs):
    super().__init__(model, data, **kwargs)
      
    bonus = kwargs.get('kwargs', {}).get('bonus', 8)
    dist_weight = kwargs.get('kwargs', {}).get('dist_weight', 1)

    # This task requires an end-effector and shoulder to be defined
    assert end_effector[0] == "geom", "end-effector must be a geom because contacts are between geoms"
    self._end_effector = end_effector[1]
    self._shoulder = shoulder
    
    # For LLC policy  #TODO: remove?
    self._target_qpos = [0,0,0,0,0]
    

    # Get buttons
    self._buttons = [f"button-{idx}" for idx in range(4)]
    self._current_button = self._buttons[0]
    self._current_button_force = 0

    # Use early termination if target is not hit in time
    self._steps_since_last_hit = 0
    self._max_steps_without_hit = self._action_sample_freq*4

    # Define a maximum number of button presses
    self._trial_idx = 0
    self._trial_idx_buttons = {b: 0 for b in self._buttons}
    self._max_trials = kwargs.get('max_trials', 10)
    self._targets_hit = 0
    self._targets_hit_buttons = {b: 0 for b in self._buttons}

    # Used for logging states
    self._info = {"target_hit": False, "new_button_generated": False,
                  "terminated": False, "truncated": False, "termination": False, "task_completion_time": 0}
    self._info.update({f"success_rate_{b}": 0 for b in self._buttons})

    # Define a default reward function
    self._reward_function = RewardFactory.create_reward_function(reward_function, k=dist_weight, bonus=bonus)

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Get shoulder position
    shoulder_pos = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos.copy()

    # Update button positions
    model.body("button-0").pos = shoulder_pos + [0.41, -0.07, -0.15]
    model.body("button-1").pos = shoulder_pos + [0.41, 0.07, -0.15]
    model.body("button-2").pos = shoulder_pos + [0.5, -0.07, -0.05]
    model.body("button-3").pos = shoulder_pos + [0.5, 0.07, -0.05]

    # Set camera angle
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([1.1, -0.9, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])
    #model.cam_pos[model.camera_name2id('for_testing')] = np.array([-0.8, -0.6, 1.5])
    #model.cam_quat[model.camera_name2id('for_testing')] = np.array(
    #  [0.718027, 0.4371043, -0.31987, -0.4371043])

  @classmethod
  def initialise(cls, task_kwargs):

    assert "end_effector" in task_kwargs, "End-effector must be defined for this environment"
    end_effector = task_kwargs["end_effector"][1]

    # Parse xml file
    tree = ET.parse(cls.get_xml_file())
    root = tree.getroot()

    # Add contact between end-effector and buttons
    if root.find('contact') is None:
      root.append(ET.Element('contact'))
    for idx in range(4):
      root.find('contact').append(ET.Element('pair', name=f"ee-button-{idx}", geom1=end_effector,
                                             geom2=f"button-{idx}"))
    for idx in range(4):
      root.find('contact').append(ET.Element('pair', name=f"ee2-button-{idx}", geom1="hand_2midph",
                                             geom2=f"button-{idx}"))
    for idx in range(4):
      root.find('contact').append(ET.Element('pair', name=f"ee3-button-{idx}", geom1="hand_2proxph",
                                             geom2=f"button-{idx}"))

    return tree

  def _update(self, model, data):

    # Set defaults
    terminated = False
    truncated = False
    self._info["new_button_generated"] = False
    # Check if the correct button has been pressed with suitable force
    #force = data.sensor(self._current_button).data
    self._current_button_force = data.sensor(self._current_button).data
    self._info["button_force"] = self._current_button_force

    # Get end-effector position and target position
    ee_position = data.geom(self._end_effector).xpos
    target_position = data.geom(self._current_button).xpos

    # Distance to target
    #dist = np.linalg.norm(target_position - ee_position)
    dist = abs(data.sensor(f"{self._current_button}_distance").data[0])

    if 50 > self._current_button_force > 25:
      self._info["task_completion_time"] = data.time
      self._info["target_hit"] = True
      self._trial_idx += 1
      self._trial_idx_buttons[self._current_button] += 1
      self._targets_hit += 1
      self._targets_hit_buttons[self._current_button] += 1
      self._steps_since_last_hit = 0
      self._info["acc_dist"] += dist
      self._info[f"success_rate_{self._current_button}"] = self._targets_hit_buttons[self._current_button] / self._trial_idx_buttons[self._current_button]
      self._choose_button(model, data)
      self._info["new_button_generated"] = True

    else:
      self._info["target_hit"] = False

      # Check if time limit has been reached
      self._steps_since_last_hit += 1
      if self._steps_since_last_hit >= self._max_steps_without_hit:
        # Choose a new button
        self._steps_since_last_hit = 0
        self._trial_idx += 1
        self._trial_idx_buttons[self._current_button] += 1
        self._info["acc_dist"] += dist
        self._info[f"success_rate_{self._current_button}"] = self._targets_hit_buttons[self._current_button] / self._trial_idx_buttons[self._current_button]
        self._choose_button(model, data)
        self._info["new_button_generated"] = True

    # Check if max number trials reached
    if self._trial_idx >= self._max_trials:
      self._info["dist_from_target"] = self._info["acc_dist"]/self._trial_idx
      truncated = True
      self._info["termination"] = "max_trials_reached"

    # Calculate reward
    reward = self._reward_function.get(self, dist, self._info.copy())

    return reward, terminated, truncated, self._info.copy()

  def _choose_button(self, model, data):

    # Choose a new button randomly, but don't choose the same button as previous one
    while True:
      new_button = self._rng.choice(self._buttons)
      if new_button != self._current_button:
        self._current_button = new_button
        break

    # Set color of screen
    model.geom("screen").rgba = model.geom(self._current_button).rgba

    mujoco.mj_forward(model, data)

  def _get_state(self, model, data):
    state = dict()
    state["current_button"] = self._current_button
    state["trial_idx"] = self._trial_idx
    state["targets_hit"] = self._targets_hit
    state.update(self._info)
    return state

  def _reset(self, model, data):

    # Reset counters
    self._steps_since_last_hit = 0
    self._trial_idx = 0
    self._trial_idx_buttons = {b: 0 for b in self._buttons}
    self._targets_hit = 0
    self._targets_hit_buttons = {b: 0 for b in self._buttons}

    self._info = {"target_hit": False, "new_button_generated": False, "terminated": False, "truncated": False,
                  "termination": False, "dist_from_target": 0, "acc_dist": 0, "task_completion_time": 0}
    self._info.update({f"success_rate_{b}": 0 for b in self._buttons})

    # Choose a new button
    self._choose_button(model, data)

    return self._info

  def get_stateful_information(self, model, data):
    # Time features
    targets_hit = -1.0 + 2*(self._trial_idx/self._max_trials)
    return np.array([targets_hit])
