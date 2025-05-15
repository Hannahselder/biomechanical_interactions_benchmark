import gymnasium

from gymnasium.envs.registration import registry


# Add simulator_folder to Python path
import sys
simulator_folder =  "/home/sc.uni-leipzig.de/hm68arex/user-in-the-box/simulators/mobl_arms_index_pointing"
sys.path.insert(0, simulator_folder)

# Import the module so that the gym env is registered
import importlib
importlib.import_module("mobl_arms_index_pointing")

# Initialise a simulator with gym(nasium)
import gymnasium as gym
simulator = gym.make("uitb:mobl_arms_index_pointing-v1")

# Zeige alle registrierten Environments an
gymnasium.pprint_registry()