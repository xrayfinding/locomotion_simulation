# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Sequence, Tuple

import numpy as np
# import time

from locomotion.agents.whole_body_controller import gait_generator as gait_generator_lib
from locomotion.agents.whole_body_controller import leg_controller
from locomotion.agents.whole_body_controller import qp_torque_optimizer

_FORCE_DIMENSION = 3
#KP = np.array((0., 0., 100., 100., 100., 0.))
KP = np.array((100., 300., 300., 200., 200., 200.))
KD = np.array((40., 30., 10., 10., 10., 30.))
#KD = np.array((10., 10., 10., 10., 10., 10.))
# VSMC 增益：

# Kx = np.array((9000., 9000., 9000., 9000., 9000., 9000.))*.6
# Ka = np.array((7000., 7000., 700., 7000., 7000., 700.))
# Dx = np.array((12000., 12000., 12000., 12000., 12000., 12000.))
# Da = np.array((14000., 14000., 140000., 14000., 14000., 140000.))

Kx = np.array((9000., 9000., 9000., 9000., 9000., 9000.))*.8
Ka = np.array((7000., 7000., 7000., 7000., 7000., 7000.))
Dx = np.array((12000., 12000., 12000., 12000., 12000., 12000.))*4
Da = np.array((140000., 140000., 140000., 140000., 140000., 140000.))*4

MAX_DDQ = np.array((10., 100., 10., 20., 20., 20.))*10
MIN_DDQ = -MAX_DDQ


class TorqueStanceLegController(leg_controller.LegController):
  """A torque based stance leg controller framework.

  Takes in high level parameters like walking speed and turning speed, and
  generates necessary the torques for stance legs.
  """
  def __init__(
      self,
      robot: Any,
      gait_generator: Any,
      state_estimator: Any,
      desired_speed: Tuple[float, float] = (0, 0),
      desired_twisting_speed: float = 0,
      desired_body_height: float = 0.45,
      num_legs: int = 4,
      friction_coeffs: Sequence[float] = (0.45, 0.45, 0.45, 0.45),
  ):
    """Initializes the class.

    Tracks the desired position/velocity of the robot by computing proper joint
    torques using MPC module.

    Args:
      robot: A robot instance.
      gait_generator: Used to query the locomotion phase and leg states.
      state_estimator: Estimate the robot states (e.g. CoM velocity).
      desired_speed: desired CoM speed in x-y plane.
      desired_twisting_speed: desired CoM rotating speed in z direction.
      desired_body_height: The standing height of the robot.
      body_mass: The total mass of the robot.
      body_inertia: The inertia matrix in the body principle frame. We assume
        the body principle coordinate frame has x-forward and z-up.
      num_legs: The number of legs used for force planning.
      friction_coeffs: The friction coeffs on the contact surfaces.
    """
    self._robot = robot
    self._gait_generator = gait_generator
    self._state_estimator = state_estimator
    self.desired_speed = desired_speed
    self.desired_twisting_speed = desired_twisting_speed

    self._desired_body_height = desired_body_height
    self._num_legs = num_legs
    self._friction_coeffs = np.array(friction_coeffs)
    self._qp_torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
        robot.MPC_BODY_MASS, robot.MPC_BODY_INERTIA)
    # 差分
    self._Xb_error = np.array((0.,0.,0.,0.,0.,0.))
    self._Xb_error_last = np.array((0.,0.,0.,0.,0.,0.))
    self._Ab_error = np.array((0.,0.,0.,0.,0.,0.))
    self._Ab_error_last = np.array((0.,0.,0.,0.,0.,0.))

    self._Ab_dot = np.array((0.,0.,0.,0.,0.,0.))
    self._Init_PA = 0
    self._Init_base = np.array((0.,0.,0.,0.,0.,0.))
    self._Rd = np.array([[1,0,0],[0,1,0],[0,0,1]])
  def reset(self, current_time):
    del current_time

  def update(self, current_time):
    del current_time

  def _estimate_robot_height(self, contacts, foot_positions):
    base_orientation = self._robot.GetBaseOrientation()
    rot_mat = self._robot.pybullet_client.getMatrixFromQuaternion(
        base_orientation)
    rot_mat = np.array(rot_mat).reshape((3, 3))
    foot_positions_world_frame = (rot_mat.dot(foot_positions.T)).T

    # pylint: disable=unsubscriptable-object
    useful_heights = contacts * (-foot_positions_world_frame[:, 2])
    return np.sum(useful_heights) / np.sum(contacts)

  def get_action(self):
    """Computes the torque for stance legs."""
    # Actual q and dq
    contacts = np.array(
        [(leg_state in (gait_generator_lib.LegState.STANCE,
                        gait_generator_lib.LegState.EARLY_CONTACT))
         for leg_state in self._gait_generator.desired_leg_state],
        dtype=np.float64)
    foot_positions = self._robot.GetFootPositionsInBaseFrame()

    robot_com_height = self._estimate_robot_height(contacts, foot_positions)
    robot_com_velocity = self._state_estimator.com_velocity_body_frame
    robot_com_velocity = self._robot.GetBaseVelocity()
    robot_com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw())
    R = self._robot.ComputeR(robot_com_roll_pitch_yaw)

    #print("RPY: ", robot_com_roll_pitch_yaw)
    #robot_com_roll_pitch_yaw[2] = 0.  # To prevent yaw drifting
    robot_com_roll_pitch_yaw_rate = self._robot.GetBaseRollPitchYawRate()
    robot_q = np.hstack(([0., 0., robot_com_height], robot_com_roll_pitch_yaw))
    # TODO :此处的Q不准确，质心的X，Y都是用0来代替的，需要改为实时读取。
    robot_base = self._robot.GetBasePosition()
    list_base = [0,0,0]
    # for i in range(3):
    #   list_base[i] = robot_base[i]
    #   list_base[i]=round(list_base[i], 4)
    # print("robot_base: ",list_base)
    for i in range(3):
      robot_q[i] = robot_base[i]

    robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))
    # Desired q and dq
    desired_com_position = np.array((0., 0., self._desired_body_height),
                                    dtype=np.float64)
    desired_com_velocity = np.array(
        (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)
    desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)
    desired_com_angular_velocity = np.array(
        (0., 0., self.desired_twisting_speed), dtype=np.float64)
    desired_q = np.hstack((desired_com_position, desired_com_roll_pitch_yaw))
    desired_dq = np.hstack(
        (desired_com_velocity, desired_com_angular_velocity))
    # Desired ddq
    # print("contacts: " , contacts) 
    # print("desired_robot_base",desired_q)
    # print("Delta_base" , desired_q - robot_q)
    # print("KP_segment:", KP * (desired_q - robot_q))
    # print("KD_segment:", KD * (desired_dq - robot_dq))
    desired_ddq = KP * (desired_q - robot_q) + KD * (desired_dq - robot_dq)
    desired_ddq = np.clip(desired_ddq, MIN_DDQ, MAX_DDQ)
    contact_forces = self._qp_torque_optimizer.compute_contact_force(
        foot_positions, desired_ddq, contacts=contacts)
    #print("Contact_Force_QP:: ", contact_forces)
    #VSMC 的实现：
    #Golaoxu:
    # Position error 的表示： 1*6
    # if self._Init_PA == 1:
    #   desired_q = self._Init_base
    #print("Base: ", robot_q)
    X_f1 = (desired_q - robot_q)[:3]
    X_f2 = (desired_q - robot_q)[:3]
    X_f = np.hstack(
        (X_f1, X_f2))
    self._Xb_error_last = self._Xb_error
    self._Xb_error = X_f
    _Xb_vel = self._Xb_error - self._Xb_error_last

    Xb_dot = (desired_dq - robot_dq)[:3]
    Xb_dot = np.hstack((Xb_dot, Xb_dot))

    # Attitude error 的表示 ： 1*6
    # 0.026 表示 VH
    hip1_In_base = np.array((0.22, -0.0875, 0.002))
    hip2_In_base = np.array((-0.22, 0.0875, 0.002))
    R_d = np.array([[1,0,0],[0,1,0],[0,0,1]])
    base_orientation = self._robot.GetBaseOrientation()
    # R = self._robot.pybullet_client.getMatrixFromQuaternion(
    #     base_orientation)
    # print("R_1", R)

    R = np.array(R).reshape((3, 3))
    # if self._Init_PA == 1:
    #   R_d = self._Rd
    A_f1 = ((R_d - R).dot(hip1_In_base)).T
    A_f2 = ((R_d - R).dot(hip2_In_base)).T
    A_f = np.hstack((A_f1, A_f2))
    print("A_f", A_f)
    self._Ab_error_last = self._Ab_error
    self._Ab_error = A_f
    _Ab_vel = self._Ab_error - self._Ab_error_last
    self._Ab_dot -= A_f
    Ab_dot = self._Ab_dot
    self._Ab_dot = A_f
  
    #Golaoxu : 控制率
    #666666666666666
    #666666666666666
    #999999999999999
    F_d = Kx*X_f + Ka*A_f + Dx*_Xb_vel + Da*_Ab_vel
    #print("_Ab_vel ", Da*_Ab_vel, "A_f", Ka*A_f)
    gravity_comp = np.array((0.,0.,self._robot.MPC_BODY_MASS*9.8/2.0,0.,0.,self._robot.MPC_BODY_MASS*9.8/2.0))
    F_d += gravity_comp
    # Golaoxu: F perhaps is not in the right frame 
    F_d1 = F_d[:3]
    F_d2 = F_d[3:6]
    F_d1 = (R.T).dot(F_d1)
    F_d2 = (R.T).dot(F_d2)
    R_all = np.hstack((R.T, R.T))

    #Golaoxu：
    # ...
    # guess the Jacob is not right
    # compute the Jacob right here 
    # all_joint_angles = [state[0] for state in robot.joint_states]
    # zero_vec = [0] * len(all_joint_angles)
    # jv, _ = robot.pybullet_client.calculateJacobian(robot.quadruped, link_id,
    #                                                 (0, 0, 0), all_joint_angles,
    #                                                 zero_vec, zero_vec)
    # jacobian = np.array(jv)


    Jaco_1 = self._robot.ComputeJacobian(0)
    Jaco_2 = self._robot.ComputeJacobian(3)

    Jaco = np.vstack((Jaco_1.T, Jaco_2.T))
    Fq = -Jaco.dot((R_all).dot(F_d))
    # Fq1 = Jaco_1.dot(F_d1)
    # Fq2 = Jaco_2.dot(F_d2)
    # Fq = np.hstack((Fq1,Fq2))

    Fq = Fq
    #print("Fq", Fq)
    #print("VSMC_torques: ", Fq)
    action = {}
    #Golaoxu : 双腿支撑时，切换控制器。
    if contacts[1] == 0:
        if self._Init_PA == 0:
          self._Init_base = robot_q
          self._Init_PA = 1
          R_tmp = self._robot.pybullet_client.getMatrixFromQuaternion(
            base_orientation)
          self._Rd = np.array(R).reshape((3, 3))
        #   print("Finish Init")
        # print("IN VSMC CONTROL")
        # print("Xf: ", Kx*X_f)
        # print("Af: ", Ka*A_f)
        # print("Xfd: ", Dx*Xb_dot)
        # print("Afd: ", Da*Ab_dot)
        contact_forces[0,:] = -F_d1
        contact_forces[3,:] = -F_d2
    # print("Contact_Force_QP:: ", contact_forces)
    for leg_id, force in enumerate(contact_forces):
      # While "Lose Contact" is useful in simulation, in real environment it's
      # susceptible to sensor noise. Disabling for now.
      # if self._gait_generator.leg_state[
      #     leg_id] == gait_generator_lib.LegState.LOSE_CONTACT:
      #   force = (0, 0, 0)
      motor_torques = self._robot.MapContactForceToJointTorques(leg_id, force)
      #print("motor_torque: ",motor_torques)
      for joint_id, torque in motor_torques.items():
        action[joint_id] = (0, 0, 0, 0, torque)
        # if contacts[1] == 0:
        #   if leg_id == 0:
        #     action[joint_id] = (0,0,0,0,Fq[joint_id])
        #   if leg_id == 3:
        #     action[joint_id] = (0,0,0,0,Fq[joint_id-6])
    return action, contact_forces
