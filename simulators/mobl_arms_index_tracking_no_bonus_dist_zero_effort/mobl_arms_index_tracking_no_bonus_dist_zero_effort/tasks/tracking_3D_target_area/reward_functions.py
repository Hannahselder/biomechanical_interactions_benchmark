import numpy as np
from abc import ABC, abstractmethod


class BaseFunction(ABC):
  @abstractmethod
  def get(self, env, dist, info):
    pass
  @abstractmethod
  def __repr__(self):
    pass

class RewardFactory:
    @staticmethod
    def create_reward_function(class_name: str, k=None):
        if class_name == "Distance":
            return Distance(k)
        elif class_name == "DistanceWithHitBonus":
            return DistanceWithHitBonus(k)
        elif class_name == "QuadraticDistance":
            return QuadraticDistance(k)
        elif class_name == "QuadraticDistanceWithHitBonus":
            return QuadraticDistanceWithHitBonus(k)
        elif class_name == "ExpDistanceWithHitBonus":
            return ExpDistanceWithHitBonus(k)
        elif class_name == "ExpDistance":
            return ExpDistance(k)
        elif class_name == "PositiveBinary":
            return PositiveBinary(k)
        elif class_name == "PositiveBinaryWithPenalty":
            return PositiveBinaryWithPenalty(k)
        elif class_name == "PositiveBinaryWithPenaltyWithoutBonus":
            return PositiveBinaryWithPenaltyWithoutBonus(k)
        else:
            raise ValueError(f"Unbekannte Klasse: {class_name}")

class Distance(BaseFunction):

  def __init__(self, k=1):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 0
    return (-self.k*dist)

  def __repr__(self):
    return "Distance"

class DistanceWithHitBonus(BaseFunction):

  def __init__(self, k=1):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 8
    else:
      return (-self.k*dist)

  def __repr__(self):
    return "DistanceWithHitBonus"

class QuadraticDistance(BaseFunction):

  def __init__(self, k=1):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 0
    return (-self.k*dist**2)

  def __repr__(self):
    return "Distance"

class QuadraticDistanceWithHitBonus(BaseFunction):

  def __init__(self, k=1):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 8
    else:
      return (-self.k*dist**2)

  def __repr__(self):
    return "Distance"

class ExpDistance(BaseFunction):

  def __init__(self, k=1):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 0
    return (np.exp(-dist*10*self.k) - 1)/10

  def __repr__(self):
    return "ExpDistance"

class ExpDistanceWithHitBonus(BaseFunction):

  def __init__(self, k=1):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 8
    else:
      return (np.exp(-dist*10*self.k) - 1)/10
  def __repr__(self):
    return "ExpDistanceWithHitBonus"

#sparse rewards
class PositiveBinary(BaseFunction):

  def __init__(self, k=1):  #default: 8 at goal achievement
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return self.k
    else:
      return 0

  def __repr__(self):
    return "PositiveBinary"

class PositiveBinaryWithPenalty(BaseFunction):

  def __init__(self, k=1):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return self.k
    else:
      return -1

  def __repr__(self):
    return "PositiveBinaryWithPenalty"

class PositiveBinaryWithPenaltyWithoutBonus(BaseFunction):

  def __init__(self, k=1):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 0
    else:
      return -self.k

  def __repr__(self):
    return "PositiveBinaryWithPenaltyWithoutBonus"