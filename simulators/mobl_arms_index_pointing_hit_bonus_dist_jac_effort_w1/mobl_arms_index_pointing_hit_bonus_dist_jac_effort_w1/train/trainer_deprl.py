import sys
import os, shutil
import logging
from datetime import datetime

import random

import wandb
from wandb.integration.sb3 import WandbCallback

from uitb.simulator import Simulator
from uitb.utils.functions import output_path, timeout_input

import os
import traceback

import torch

import deprl

from deprl import custom_distributed
from deprl.utils import load_checkpoint, prepare_params
from deprl.vendor.tonic import logger

import sys
import gymnasium
from gymnasium.envs.registration import register

import argparse

def train(
    config, simulator, checkpoint_path
):
    """pip ins
    Trains an agent on an environment.
    """
    #tonic_conf = config["tonic"]

    # Run the header first, e.g. to load an ML framework.
    #if "header" in tonic_conf:
    #    exec(tonic_conf["header"])

    # In case no env_args are passed via the config
    #if "env_args" not in config or config["env_args"] is None:
    config["env_args"] = {}

    # Build the training environment.
    _environment = f"deprl.environments.Gym('{id_env}')"
    environment = custom_distributed.distribute(
        environment=_environment, 
        config=config,
        env_args=config["env_args"],
    )
    seed_config = 0
    environment.initialize(seed=seed_config)

    # Build the testing environment.
    _test_environment = _environment

    test_env_args = config["env_args"]

    test_environment = custom_distributed.distribute(
        environment=_test_environment,
        config=config,
        env_args=test_env_args,
        parallel=1,
        sequential=1,
    )
    test_environment.initialize(seed = seed_config + 1000000)

    # Build the agent.
    agent1 = "deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())      (replay=deprl.replays.buffers.Buffer(return_steps=3,batch_size=256, steps_between_batches=1000, batch_iterations=30, steps_before_batches=2e5))"
    agent = eval(agent1)

    # Set custom mpo parameters
    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed_config,
    )

    # Set DEP parameters
    if hasattr(agent, "expl") and "DEP" in config:
        agent.expl.set_params(config["DEP"])

    # Initialize the logger to get paths
    logger.initialize(
        script_path=__file__,
        config=config,
        test_env=test_environment,
        resume=1,# TODO: resume??
    )
    path = logger.get_path()

    # Process the checkpoint path same way as in tonic_conf.play
    #checkpoint_path = os.path.join(path, "checkpoints")

    time_dict = {"steps": 0, "epochs": 0, "episodes": 0}
    if checkpoint_path is not None:
        (
            _,
            checkpoint_path,
            loaded_time_dict,
        ) = load_checkpoint(checkpoint_path, checkpoint="last")
    else: loaded_time_dict = None
    time_dict = time_dict if loaded_time_dict is None else loaded_time_dict

    if checkpoint_path:
        # Load the logger from a checkpoint.
        logger.load(checkpoint_path, time_dict)
        # Load the weights of the agent form a checkpoint.
        agent.load(checkpoint_path)

    # Build the trainer.
    trainer = "deprl.custom_trainer.Trainer(steps=int(1e8), epoch_steps=int(2e5), save_steps=int(1e6))"
    trainer = eval(trainer)
    trainer.initialize(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
        full_save=1,
    )

    # Run some code before training.
    #if tonic_conf["before_training"]:
        #exec(tonic_conf["before_training"])

    # Train.
    try:
        trainer.run(config, **time_dict)
    except Exception as e:
        logger.log(f"trainer failed. Exception: {e}")
        traceback.print_tb(e.__traceback__)

    # Run some code after training.
    #if tonic_conf["after_training"]:
        #exec(["after_training"])

if __name__=="__main__":
  #gymnasium.pprint_registry()

  parser = argparse.ArgumentParser(description='Train an agent.')
  parser.add_argument('config_file_path', type=str,
                      help='path to the config file')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='filename of a specific checkpoint to resume training at '
                           '(default: None, start training from the scratch)')
  parser.add_argument('--resume', action='store_true', help='resume at latest checkpoint')
  parser.add_argument('--eval', type=int, default=None, const=400000, nargs='?', help='run and store evaluations at a specific frequency (every ``eval`` timestep)')
  parser.add_argument('--eval_info_keywords', type=str, nargs='*', default=[], help='additional keys of ``info``  dict that should be logged at the end of each evaluation episode')
  args = parser.parse_args()

  # Get config file path
  config_file_path = args.config_file_path

  # Build the simulator
  simulator_folder = Simulator.build(config_file_path)
  sys.path.insert(0, simulator_folder)

  # Initialise
  simulator = Simulator.get(simulator_folder)

  # Get the config
  config = simulator.config

  # Get simulator name
  name = config.get("simulator_name", None)

  # Get checkpoint dir
  checkpoint_dir = os.path.join(simulator._simulator_folder, 'checkpoints')

  # Restore wandb run_id from checkpoint if available
  checkpoint = args.checkpoint
  resume_training = args.resume or (checkpoint is not None)
  if resume_training:
    if os.path.isdir(checkpoint_dir):
      existing_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if
              os.path.isfile(os.path.join(checkpoint_dir, f))]
    else:
      raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}\nTry to run without --checkpoint or --resume.")

    if checkpoint is not None:
      checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
      if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    else:
      assert len(existing_checkpoints) > 0, f"There are no checkpoints found in checkpoint directory: {checkpoint_dir}\n" \
                                            f"Maybe existing checkpoints were moved to a backup directory, " \
                                            f"which can be renamed to 'checkpoints/' to resume training."
      # # find largest checkpoint
      # checkpoint_path = sorted(existing_checkpoints, key=lambda f: int(f.split("_steps")[0].split("_")[-1]))[-1]
      # find latest checkpoint
      checkpoint_path = sorted(existing_checkpoints, key=os.path.getctime)[-1]
    try:
      _data, _, _ = load_from_zip_file(checkpoint_path)
      wandb_id = _data["policy_kwargs"]["wandb_id"]
      print(f"Resume wandb run {wandb_id} starting at checkpoint {checkpoint_path}.")
    except Exception:
      logging.warning("Cannot reliably identify wandb run id. Will resume training, but with new wandb instance and with step counter reset to zero.")
      wandb_id = None
  else:
    checkpoint_path = None
    wandb_id = None
    # Backup existing checkpoint directory
    if os.path.isdir(checkpoint_dir):
      existing_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if
              os.path.isfile(os.path.join(checkpoint_dir, f))]
      if len(existing_checkpoints) > 0:
        last_checkpoint_time = max([os.path.getctime(f) for f in existing_checkpoints])
        last_checkpoint_time = datetime.fromtimestamp(last_checkpoint_time).strftime('%Y%m%d_%H%M%S')
        checkpoint_dir_backup = os.path.join(simulator._simulator_folder, f'checkpoints_{last_checkpoint_time}')
        shutil.move(checkpoint_dir, checkpoint_dir_backup)

  if name is None:
    # Ask user to name this run
    name = timeout_input("Give a name for this run. Input empty string or wait for 30 seconds for a random name.",
                         timeout=30, default="")
    config["simulator_name"] = name.replace("-", "_")

  # Get project name
  project = config.get("project", "uitb")

  # Prepare evaluation by storing custom log tags
  with_evaluation = args.eval is not None
  eval_freq = args.eval
  eval_info_keywords = tuple(args.eval_info_keywords)

  # Initialise wandb
  #if wandb_id is None:
    #wandb_id = wandb.util.generate_id()
  #run = wandb.init(id=wandb_id, resume="allow", project=project, name=name, config=config, sync_tensorboard=True, save_code=True, dir=output_path())
  #wandb.save(os.path.join(model_folder, run.name, 'checkpoints', "model_*_steps.zip"),
             #base_path=os.path.join(model_folder, run.name, 'checkpoints'))

  #from gymnasium.envs.registration import register

  id_env = 'uitb:' + name + '-v0'

  #register(
  #  id = id_env,                 # Name der Umgebung
  #  entry_point='uitb:Simulator',   # Pfad zur Klasse
  #)

  train(config, id_env ,checkpoint_path)# missing information, wandb_id, eval_specifics





    

  # Initialise RL model
  # rl_cls = simulator.get_class("rl", config["rl"]["algorithm"])
   #rl_model = rl_cls(simulator, checkpoint_path=checkpoint_path, wandb_id=wandb_id)


  # Start the training
  # rl_model.learn(WandbCallback(verbose=2))
   #rl_model.learn(WandbCallback(verbose=2),
   #               with_evaluation=with_evaluation, eval_freq=eval_freq, eval_info_keywords=eval_info_keywords)

    #run.finish()
