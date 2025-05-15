import numpy as np
import xml.etree.ElementTree as ET
import pathlib
import os
import shutil
import inspect
import mujoco
from abc import ABC, abstractmethod
import importlib
from typing import final
import random

from ..utils.functions import parent_path
from ..utils import element_tree as ETutils


class BaseBMModel(ABC):

  def __init__(self, model, data, **kwargs):
    """Initializes a new `BaseBMModel`.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      **kwargs: Many keywords that should be documented somewhere
    """

    # Initialise mujoco model of the biomechanical model, easier to manipulate things
    bm_model = mujoco.MjModel.from_xml_path(self.get_xml_file())

    # Get an rng
    self._rng = np.random.default_rng(kwargs.get("random_seed", None))

    # Reset type
    self._reset_type = kwargs.get("reset_type", "epsilon_uniform")
    ## valid reset types: 
    _valid_reset_types = ("zero", "epsilon_uniform", "range_uniform")
    assert self._reset_type in _valid_reset_types, f"Invalid reset type '{self._reset_type} (valid types are {_valid_reset_types})."

    # Total number of actuators
    self._nu = bm_model.nu

    # Number of muscle actuators
    self._na = bm_model.na

    # Number of motor actuators
    self._nm = self._nu - self._na
    self._motor_act = np.zeros((self._nm,))
    self._motor_alpha = 0.9

    # Get actuator names (muscle and motor)
    self._actuator_names = [mujoco.mj_id2name(bm_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(bm_model.nu)]
    self._muscle_actuator_names = set(np.array(self._actuator_names)[bm_model.actuator_trntype==3])
    self._motor_actuator_names = set(self._actuator_names) - self._muscle_actuator_names

    # Sort the names to preserve original ordering (not really necessary but looks nicer)
    self._muscle_actuator_names = sorted(self._muscle_actuator_names, key=self._actuator_names.index)
    self._motor_actuator_names = sorted(self._motor_actuator_names, key=self._actuator_names.index)

    # Find actuator indices in the simulation
    self._muscle_actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                             for actuator_name in self._muscle_actuator_names]
    self._motor_actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                            for actuator_name in self._motor_actuator_names]

    # Get joint names (dependent and independent)
    self._joint_names = [mujoco.mj_id2name(bm_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(bm_model.njnt)]
    self._dependent_joint_names = {self._joint_names[idx] for idx in
                                  np.unique(bm_model.eq_obj1id[bm_model.eq_active0.astype(bool)])} \
      if bm_model.eq_obj1id is not None else set()
    self._independent_joint_names = set(self._joint_names) - self._dependent_joint_names

    # Sort the names to preserve original ordering (not really necessary but looks nicer)
    self._dependent_joint_names = sorted(self._dependent_joint_names, key=self._joint_names.index)
    self._independent_joint_names = sorted(self._independent_joint_names, key=self._joint_names.index)

    # Find dependent and independent joint indices in the simulation
    self._dependent_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                             for joint_name in self._dependent_joint_names]
    self._independent_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                               for joint_name in self._independent_joint_names]

    # If there are 'free' type of joints, we'll need to be more careful with which dof corresponds to
    # which joint, for both qpos and qvel/qacc. There should be exactly one dof per independent/dependent joint.
    def get_dofs(joint_indices):
      qpos = []
      dofs = []
      for joint_idx in joint_indices:
        if model.jnt_type[joint_idx] not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
          raise NotImplementedError(f"Only 'hinge' and 'slide' joints are supported, joint "
                                    f"{self._joint_names[joint_idx]} is of type {mujoco.mjtJoint(model.jnt_type[joint_idx]).name}")
        qpos.append(model.jnt_qposadr[joint_idx])
        dofs.append(model.jnt_dofadr[joint_idx])
      return qpos, dofs
    self._dependent_qpos, self._dependent_dofs = get_dofs(self._dependent_joints)
    self._independent_qpos, self._independent_dofs = get_dofs(self._independent_joints)

    # Get the effort model; some models might need to know dt
    self._effort_model = self.get_effort_model(kwargs.get("effort_model", {"cls": "Zero"}), dt=kwargs["dt"])

    # Define signal-dependent noise
    self._sigdepnoise_type = kwargs.get("sigdepnoise_type", None)  #"white")
    self._sigdepnoise_level = kwargs.get("sigdepnoise_level", 0.103)
    self._sigdepnoise_rng = np.random.default_rng(kwargs.get("random_seed", None))
    self._sigdepnoise_acc = 0  #only used for red/Brownian noise

    # Define constant (i.e., signal-independent) noise
    self._constantnoise_type = kwargs.get("constantnoise_type", None)  #"white")
    self._constantnoise_level = kwargs.get("constantnoise_level", 0.185)
    self._constantnoise_rng = np.random.default_rng(kwargs.get("random_seed", None))
    self._constantnoise_acc = 0  #only used for red/Brownian noise

  def _reset_zero(self, model, data):
    """ Resets the biomechanical model. """

    # Set joint angles and velocities to zero
    nq = len(self._independent_qpos)
    qpos = np.zeros((nq,))
    qvel =  np.zeros((nq,))

    # Randomly sample act within unit interval
    act = self._rng.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))

    # Set qpos and qvel
    data.qpos[self._dependent_qpos] = 0
    data.qpos[self._independent_qpos] = qpos
    data.qvel[self._dependent_dofs] = 0
    data.qvel[self._independent_dofs] = qvel
    data.act[self._muscle_actuators] = act

    # Sample random initial values for motor activation
    self._motor_act = self._rng.uniform(low=np.zeros((self._nm,)), high=np.ones((self._nm,)))
    # Reset smoothed average of motor actuator activation
    self._motor_smooth_avg = np.zeros((self._nm,))

    # Reset accumulative noise
    self._sigdepnoise_acc = 0
    self._constantnoise_acc = 0
  
  def _reset_epsilon_uniform(self, model, data):
    """ Resets the biomechanical model. """

    # Randomly sample qpos and qvel around zero values, and act within unit interval
    nq = len(self._independent_qpos)
    qpos = self._rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    qvel = self._rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    act = self._rng.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))

    # Set qpos and qvel
    ## TODO: ensure that constraints are initially satisfied
    data.qpos[self._dependent_qpos] = 0
    data.qpos[self._independent_qpos] = qpos
    data.qvel[self._dependent_dofs] = 0
    data.qvel[self._independent_dofs] = qvel
    data.act[self._muscle_actuators] = act

    # Sample random initial values for motor activation
    self._motor_act = self._rng.uniform(low=np.zeros((self._nm,)), high=np.ones((self._nm,)))
    # Reset smoothed average of motor actuator activation
    self._motor_smooth_avg = np.zeros((self._nm,))

    # Reset accumulative noise
    self._sigdepnoise_acc = 0
    self._constantnoise_acc = 0

  def _reset_range_uniform(self, model, data):
    """ Resets the biomechanical model. """

    # Randomly sample qpos within joint range, qvel around zero values, and act within unit interval
    random_index = self._rng.integers(0, len(random_poses))
    values = random_poses[random_index]
    
    #nq = len(self._independent_qpos)
    #jnt_range = model.jnt_range[self._independent_joints]
    #qpos = self._rng.uniform(low=jnt_range[:, 0], high=jnt_range[:, 1])
    #qvel = self._rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    #act = self._rng.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))

    # Set qpos and qvel
    data.qpos[self._independent_qpos] = values['qpos']
    # data.qpos[self._dependent_qpos] = 0
    data.qvel[self._independent_dofs] = values['qvel']
    # data.qvel[self._dependent_dofs] = 0
    self.ensure_dependent_joint_angles(model, data)
    data.act[self._muscle_actuators] = values['act']

    # Sample random initial values for motor activation
    self._motor_act = self._rng.uniform(low=np.zeros((self._nm,)), high=np.ones((self._nm,)))
    # Reset smoothed average of motor actuator activation
    self._motor_smooth_avg = np.zeros((self._nm,))

    # Reset accumulative noise
    self._sigdepnoise_acc = 0
    self._constantnoise_acc = 0

  def ensure_dependent_joint_angles(self, model, data):
    """ Adjusts virtual joints according to active joint constraints. """

    for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
            model.eq_obj1id[
                (model.eq_type == 2) & (data.eq_active == 1)],
            model.eq_obj2id[
                (model.eq_type == 2) & (data.eq_active == 1)],
            model.eq_data[(model.eq_type == 2) &
                                        (data.eq_active == 1), 4::-1]):
        if physical_joint_id >= 0:
            data.joint(virtual_joint_id).qpos = np.polyval(poly_coefs, data.joint(physical_joint_id).qpos)

  ############ The methods below you should definitely overwrite as they are important ############

  @classmethod
  @abstractmethod
  def _get_floor(cls):
    """ If there's a floor in the bm_model.xml file it should be defined here.

    Returns:
      * None if there is no floor in the file
      * A dict like {"tag": "geom", "name": "name-of-the-geom"}, where "tag" indicates what kind of element the floor
      is, and "name" is the name of the element.
    """
    pass


  ############ The methods below are overwritable but often don't need to be overwritten ############

  def _reset(self, model, data):
    """ Resets the biomechanical model. """
    if self._reset_type == "zero":
      return self._reset_zero(model, data)
    elif self._reset_type == "epsilon_uniform":
      return self._reset_epsilon_uniform(model, data)
    elif self._reset_type == "range_uniform":
      return self._reset_range_uniform(model, data)
    else:
      raise NotImplementedError

  def _update(self, model, data):
    """ Update the biomechanical model after a step has been taken in the simulator. """
    pass

  def _get_state(self, model, data):
    """ Return the state of the biomechanical model. These states are used only for logging/evaluation, not for RL
    training

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.

    Returns:
      A dict where each key should have a float or a numpy vector as their value
    """
    return dict()

  def set_ctrl(self, model, data, action):
    """ Set control values for the biomechanical model.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      action: Action values between [-1, 1]

    """

    _selected_motor_control = np.clip(self._motor_act + action[:self._nm], 0, 1)
    _selected_muscle_control = np.clip(data.act[self._muscle_actuators] + action[self._nm:], 0, 1)

    if self._sigdepnoise_type is not None:
        if self._sigdepnoise_type == "white":
            _added_noise = self._sigdepnoise_level*self._sigdepnoise_rng.normal(scale=_selected_muscle_control)
            _selected_muscle_control += _added_noise
        elif self._sigdepnoise_type == "whiteonly":  #only for debugging purposes
            _selected_muscle_control = self._sigdepnoise_level*self._sigdepnoise_rng.normal(scale=_selected_muscle_control)
        elif self._sigdepnoise_type == "red":
            # self._sigdepnoise_acc *= 1 - 0.1
            self._sigdepnoise_acc += self._sigdepnoise_level*self._sigdepnoise_rng.normal(scale=_selected_muscle_control)
            _selected_muscle_control += self._sigdepnoise_acc
        else:
            raise NotImplementedError(f"{self._sigdepnoise_type}")
    if self._constantnoise_type is not None:
        if self._constantnoise_type == "white":
            _selected_muscle_control += self._constantnoise_level*self._constantnoise_rng.normal(scale=1)
        elif self._constantnoise_type == "whiteonly":  #only for debugging purposes
            _selected_muscle_control = self._constantnoise_level*self._constantnoise_rng.normal(scale=1)
        elif self._constantnoise_type == "red":
            self._constantnoise_acc += self._constantnoise_level*self._constantnoise_rng.normal(scale=1)
            _selected_muscle_control += self._constantnoise_acc
        else:
            raise NotImplementedError(f"{self._constantnoise_type}")

    # Update smoothed online estimate of motor actuation
    self._motor_act = (1 - self._motor_alpha) * self._motor_act \
                             + self._motor_alpha * np.clip(_selected_motor_control, 0, 1)

    data.ctrl[self._motor_actuators] = model.actuator_ctrlrange[self._motor_actuators,0] + self._motor_act*(model.actuator_ctrlrange[self._motor_actuators, 1] - model.actuator_ctrlrange[self._motor_actuators, 0])
    data.ctrl[self._muscle_actuators] = np.clip(_selected_muscle_control, 0, 1)


  @classmethod
  def get_xml_file(cls):
    """ Overwrite this method if you want to call the mujoco xml file something other than 'bm_model.xml'. """
    return os.path.join(parent_path(inspect.getfile(cls)), "bm_model.xml")

  def get_effort_model(self, specs, dt):
    """ Returns an initialised object of the effort model class.

    Overwrite this method if you want to define your effort models somewhere else. But note that in that case you need
    to overwrite the 'clone' method as well since it assumes the effort models are defined in
    uitb.bm_models.effort_models.

    Args:
      specs: Specifications of the effort model, in format of
        {"cls": "name-of-class", "kwargs": {"kw1": value1, "kw2": value2}}}
      dt: Elapsed time between two consecutive simulation steps

    Returns:
       An instance of a class that inherits from the uitb.bm_models.effort_models.BaseEffortModel class
    """
    module = importlib.import_module(".".join(BaseBMModel.__module__.split(".")[:-1]) + ".effort_models")
    return getattr(module, specs["cls"])(self, **{**specs.get("kwargs", {}), **{"dt": dt}})

  @classmethod
  def clone(cls, simulator_folder, package_name):
    """ Clones (i.e. copies) the relevant python files into a new location.

    Args:
       simulator_folder: Location of the simulator.
       package_name: Name of the simulator (which is a python package)
    """

    # Create 'bm_models' folder
    dst = os.path.join(simulator_folder, package_name, "bm_models")
    os.makedirs(dst, exist_ok=True)

    # Copy this file
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))

    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[2:]) + " import " + cls.__name__)

    # Copy bm-model folder
    src = parent_path(inspect.getfile(cls))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

    # Copy assets
    shutil.copytree(os.path.join(src, "assets"), os.path.join(simulator_folder, package_name, "assets"),
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

    # Copy effort models
    shutil.copyfile(os.path.join(base_file.parent, "effort_models.py"), os.path.join(dst, "effort_models.py"))

  @classmethod
  def insert(cls, simulator_tree):
    """ Inserts the biomechanical model into the simulator by integrating the xml files together.

     Args:
       simulator_tree: An `xml.etree.ElementTree` containing the parsed simulator xml file
    """

    # Parse xml file
    bm_tree = ET.parse(cls.get_xml_file())
    bm_root = bm_tree.getroot()

    # Get simulator root
    simulator_root = simulator_tree.getroot()

    # Add defaults
    ETutils.copy_or_append("default", bm_root, simulator_root)

    # Add assets, except skybox
    ETutils.copy_children("asset", bm_root, simulator_root,
                          exclude={"tag": "texture", "attrib": "type", "name": "skybox"})

    # Add bodies, except floor/ground  TODO this might not be currently working
    if cls._get_floor() is not None:
      floor = cls._get_floor()
      ETutils.copy_children("worldbody", bm_root, simulator_root,
                            exclude={"tag": floor["tag"], "attrib": "name", "name": floor["name"]})
    else:
      ETutils.copy_children("worldbody", bm_root, simulator_root)

    # Add tendons
    ETutils.copy_children("tendon", bm_root, simulator_root)

    # Add actuators
    ETutils.copy_children("actuator", bm_root, simulator_root)

    # Add equality constraints
    ETutils.copy_children("equality", bm_root, simulator_root)

  def close(self, **kwargs):
    """ Perform any necessary clean up. """
    pass

  ############ The methods below you should not overwrite ############

  @final
  def update(self, model, data):
    """ Updates the biomechanical model and effort model. """
    self._update(model, data)
    self._effort_model.update(model, data)

  @final
  def reset(self, model, data):
    """ Resets the biomechanical model and effort model. """
    self._reset(model, data)
    self._effort_model.reset(model, data)
    self.update(model, data)
    mujoco.mj_forward(model, data)

  @final
  def get_state(self, model, data):
    """ Returns the state of the biomechanical model (as a dict). """
    state = dict()
    state.update(self._get_state(model, data))
    state.update(self._effort_model._get_state(model, data))
    return state

  @final
  def get_effort_cost(self, model, data):
    """ Returns effort cost from the effort model. """
    return self._effort_model.cost(model, data)

  @property
  @final
  def independent_joints(self):
    """ Returns indices of independent joints. """
    return self._independent_joints.copy()

  @property
  @final
  def independent_qpos(self):
    """ Returns qpos indices of independent joints. """
    return self._independent_qpos.copy()

  @property
  @final
  def independent_dofs(self):
    """ Returns qvel/qacc indices of independent joints. """
    return self._independent_dofs.copy()

  @property
  @final
  def nu(self):
    """ Returns number of actuators (both muscle and motor). """
    return self._nu

  @property
  def motor_act(self):
    """ Returns (smoothed average of) motor actuation. """
    return self._motor_act

random_poses = [{'qpos': [1.6353040160874788, 1.874128165582045, -1.015652374220629, 0.09745410729521709, 1.4880326476656827], 'qvel': [0.009647170406468841, 0.029026316441872133, 0.04103393812954528, 0.018815444759174516, -0.031000852661227686], 'act': [0.9814789806272665, 0.28474005001764857, 0.6292731697490502, 0.5810364787803565, 0.5999122731322729, 0.5352481053616507, 0.9957771042185515, 0.501946084696926, 0.7710225764025075, 0.49416177862830957, 0.9976744931721895, 0.9786219213695594, 0.3935680449666996, 0.321925262792205, 0.8621806511192496, 0.7993248953689494, 0.6914306327149987, 0.4085178516582625, 0.38976513267618007, 0.13164839559325658, 0.6254974877535642, 0.08240137043864959, 0.2745729886995335, 0.6561170271226267, 0.014676030285389996, 0.8345321719828639]}, {'qpos': [0.3883190204116016, 2.1084632396554075, -0.9360709531515771, 2.0384510944122494, 0.5114273797662545], 'qvel': [-0.030820265345176234, 0.007515852516535769, 0.024940288088313, 0.027992532837255113, -0.00622684366672762], 'act': [0.27831268478565063, 0.6160240831955066, 0.7299013392874861, 0.5072351464800304, 0.18460784299851174, 0.6142941478664415, 0.7603520680306871, 0.9150144582006495, 0.994448831659649, 0.368279249567819, 0.8055951101944674, 0.6895361013014285, 0.9729364618467076, 0.5087678736971596, 0.8606415080798312, 0.504046329122597, 0.28825274862633854, 0.9603826135172568, 0.2840741027111261, 0.3106582333274175, 0.05978151883028593, 0.0004562743930304203, 0.34155918207106173, 0.27069433277060573, 0.7265339385540586, 0.014649687692348645]}, {'qpos': [-0.7072249046546505, 3.0261759114671025, -1.410834683400622, 0.5533458581548231, 0.909298488472362], 'qvel': [0.027555689921232557, 0.008951908791561158, 0.0023436196814011004, -0.01531516187521196, -0.03405603816324558], 'act': [0.6451586761498982, 0.4002102958946384, 0.7742495511650035, 0.15949226710966202, 0.40734591001113174, 0.6238458277910438, 0.43871028555313774, 0.02875160165117685, 0.539677499060975, 0.08918609924368426, 0.017091558198468082, 0.21880623426588508, 0.18257886050963645, 0.7311768441037428, 0.6994670843458559, 0.7907505575960128, 0.7948185196200099, 0.9522057056500551, 0.4498522000898083, 0.8689472269762804, 0.6430659760010089, 0.6473123805896477, 0.16902127106731313, 0.8748406369048539, 0.7303995440190266, 0.6110675513157837]}, {'qpos': [-0.35310272277844, 1.5089869876863062, -0.31329469885554007, 2.164906061453433, -1.5085438777460862], 'qvel': [-0.0050786581016417295, -0.04298416897005891, 0.015353001123802154, -0.03388766558879183, -0.020470134274622055], 'act': [0.5458350926641974, 0.7728030031448865, 0.4060682624749635, 0.24034102483768105, 0.2899941660547838, 0.43695044928970084, 0.7262641279397056, 0.8821907190033436, 0.9821417565722973, 0.9164030020970013, 0.959159671660169, 0.4730102827786068, 0.38899072048038874, 0.47853595647483216, 0.38867907087472864, 0.5585147465940534, 0.5918151306936006, 0.07500282493895194, 0.9965950594491901, 0.752359808024983, 0.24483443760986434, 0.7106901635584231, 0.977182285723743, 0.4795109079640394, 0.20545336209706166, 0.559690502040132]}, {'qpos': [1.7145788142127947, 0.9060121059586982, 0.1164837117885762, 0.4225381598112338, -0.2920765827159417], 'qvel': [0.02547435356098708, -0.03507917913138445, -0.04767044181928102, -0.03521071706142928, 0.035160915198559725], 'act': [0.4849379101302892, 0.13313737852257013, 0.9762779664826158, 0.44723754314987396, 0.6736461753659068, 0.9559618021930049, 0.4561255187663258, 0.6166479705324439, 0.22289207882962292, 0.6680792985792692, 0.9066717924721072, 0.6607884695461181, 0.5455926205657291, 0.8738153146359763, 0.5489655808839735, 0.3101093632318018, 0.9370763010746299, 0.5591242338314896, 0.8103934084125187, 0.27362377901910706, 0.4883030021174295, 0.07490796854762738, 0.9934419764647638, 0.6984135840877842, 0.7828550198950991, 0.930822264795625]}, {'qpos': [1.8640725147484989, 1.0041469149520674, -0.191644282189124, 0.7506144724101862, 0.876448361666682], 'qvel': [0.03910600844539373, -0.02133203788671009, 0.025567049529060512, -0.016965494071022984, 0.000616015729407858], 'act': [0.17880695499637167, 0.19456288941683075, 0.7722259240901949, 0.27485643035642005, 0.3546606252654003, 0.20164704235427167, 0.13802826916168176, 0.7077335620041426, 0.9872931012034235, 0.4809066544555973, 0.3297212159939289, 0.6972098788529687, 0.9100847715809521, 0.09709127584763311, 0.5113807528788227, 0.35479194059129215, 0.2451581815621221, 0.02048227152044313, 0.1514370961654109, 0.4814584870951717, 0.9867858246200353, 0.3310908888273918, 0.2928796099930987, 0.5472946383903036, 0.7962666052549644, 0.7716078761791104]}, {'qpos': [-0.17484994646386776, 0.011528605138745384, -0.14738928505283821, 0.33748415949995364, 1.1940322711301155], 'qvel': [0.026391241117418, -0.009756921905884133, 0.03467219828215283, -0.008003745466015143, 0.00457565225563087], 'act': [0.043899773557812605, 0.892425134630615, 0.4524119572211762, 0.7632306304246025, 0.9836868975960302, 0.5086584693602508, 0.42319139980827547, 0.8111275517812705, 0.8396764573078652, 0.22498902786475772, 0.6142127885391802, 0.2890668900618817, 0.18459873568285357, 0.8801397475614293, 0.7639495969323475, 0.6953249633188141, 0.8219364587216282, 0.9668626056783144, 0.5431927558274837, 0.9397804588912669, 0.386606137512828, 0.999073893052066, 0.698940632138193, 0.19942665960673533, 0.14412721621649782, 0.5271597941495683]}, {'qpos': [-0.03648172655144433, 1.2171329103897177, -0.8370139025681422, 0.8673984245199099, 0.24769571793255873], 'qvel': [0.03762868875596674, -0.038348810635545974, -0.026440320301779275, 0.007601918434140709, -0.0014055573759583564], 'act': [0.22455927299308887, 0.9089792889034443, 0.47457785625907656, 0.9576226850915438, 0.8547173054309347, 0.7781375209822977, 0.31512332726040093, 0.14060545633070842, 0.912272374596588, 0.3132562995574748, 0.7105569713020685, 0.09956075262907138, 0.22844152911252136, 0.4575099850772114, 0.9592618693517496, 0.31595737659206025, 0.2292197628695105, 0.21318338531159176, 0.9485636332520403, 0.3141261561396407, 0.19290306136398716, 0.5467321420769914, 0.09351968693989032, 0.1545161021617555, 0.46280734175578075, 0.4343859177619297]}, {'qpos': [-1.0796351274432587, 0.7717800202943715, -0.7014141389701518, 1.2669454500719304, -0.92225908547746], 'qvel': [0.00794034710481445, -0.043284814429850896, -0.001313270347848547, -0.037936219079494626, -0.0382569352973201], 'act': [0.3305421922710501, 0.5470860311120507, 0.8055157655645602, 0.7178009171635651, 0.429118336988926, 0.9258223574164013, 0.9159101826386405, 0.3887195529831948, 0.2930258507261625, 0.9006753037782692, 0.30144308737768344, 0.6976005573636708, 0.4444855451848707, 0.6263011272081995, 0.36395803833790996, 0.8590864783463467, 0.8029174796376101, 0.4101776783455058, 0.9422267652206601, 0.26600897129485834, 0.9048061999374793, 0.6940833295434186, 0.800784554363265, 0.8974617365982053, 0.48515491088237783, 0.512018020208787]}, {'qpos': [1.5704448256813317, 1.5557365469231708, 0.027483085459713674, 1.4853932822329434, 0.9407746886255224], 'qvel': [-0.04132347370591527, -0.004118527364074719, -0.0252399702792128, 0.005090771468975797, 0.028028402800905725], 'act': [0.9072632534109079, 0.6379969727688839, 0.9016132889457336, 0.8946623045415754, 0.08310394050750969, 0.0623128976127445, 0.3441662481726646, 0.5998148587007027, 0.15766919042160843, 0.754228149137513, 0.8205520234583401, 0.0854637670487669, 0.11022706633514312, 0.23579825259771048, 0.964385836215214, 0.3653346637981111, 0.4348858414049924, 0.26659910534185305, 0.6619242653021616, 0.1242141605569328, 0.03131466490533341, 0.01924161676516667, 0.6375000440259232, 0.06811056122150771, 0.287362004168616, 0.8494607149015361]}]