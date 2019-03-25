# Copyright (c) 2017, Rensselaer Polytechnic Institute, Wason Technology LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Rensselaer Polytechnic Institute, or Wason 
#       Technology LLC, nor the names of its contributors may be used to 
#       endorse or promote products derived from this software without 
#       specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import threading
import sys
import rospy
import numpy as np

import general_robotics_toolbox as rox
import general_robotics_toolbox.urdf as rox_urdf
import general_robotics_toolbox.ros_msg as rox_msg

from ..srv import \
    SetControllerMode, SetControllerModeRequest

from ..msg import \
    ControllerMode, ControllerState

from geometry_msgs.msg import PoseStamped, Pose
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, \
      JointTolerance
import actionlib
from sensor_msgs.msg import JointState

class ControllerCommander(object):
    
    MODE_INVALID_EXTERNAL_SETPOINT = ControllerMode.MODE_INVALID_EXTERNAL_SETPOINT
    MODE_FT_THRESHOLD_VIOLATION = ControllerMode.MODE_FT_THRESHOLD_VIOLATION
    MODE_SENSOR_INVALID_STATE = ControllerMode.MODE_SENSOR_INVALID_STATE
    MODE_SENSOR_COMMUNICATION_ERROR = ControllerMode.MODE_SENSOR_COMMUNICATION_ERROR
    MODE_SENSOR_FAULT = ControllerMode.MODE_SENSOR_FAULT
    MODE_COLLISION_IMMINENT = ControllerMode.MODE_COLLISION_IMMINENT
    MODE_TRAJECTORY_ABORTED = ControllerMode.MODE_TRAJECTORY_ABORTED
    MODE_TRAJECTORY_TRACKING_ERROR = ControllerMode.MODE_TRAJECTORY_TRACKING_ERROR
    MODE_INVALID_TRAJECTORY = ControllerMode.MODE_INVALID_TRAJECTORY
    MODE_SETPOINT_TRACKING_ERROR = ControllerMode.MODE_SETPOINT_TRACKING_ERROR
    MODE_ROBOT_SINGULARITY = ControllerMode.MODE_ROBOT_SINGULARITY
    MODE_ROBOT_INVALID_STATE = ControllerMode.MODE_ROBOT_INVALID_STATE
    MODE_ROBOT_COMMUNICATION_ERROR = ControllerMode.MODE_ROBOT_COMMUNICATION_ERROR
    MODE_ROBOT_FAULT = ControllerMode.MODE_ROBOT_FAULT
    MODE_INVALID_ARGUMENT = ControllerMode.MODE_INVALID_ARGUMENT
    MODE_INVALID_OPERATION = ControllerMode.MODE_INVALID_OPERATION
    MODE_INTERNAL_ERROR = ControllerMode.MODE_INTERNAL_ERROR
    MODE_HALT = ControllerMode.MODE_HALT
    MODE_JOINT_TELEOP = ControllerMode.MODE_JOINT_TELEOP
    MODE_CARTESIAN_TELEOP = ControllerMode.MODE_CARTESIAN_TELEOP
    MODE_CYLINDRICAL_TELEOP = ControllerMode.MODE_CYLINDRICAL_TELEOP
    MODE_SPHERICAL_TELEOP = ControllerMode.MODE_SPHERICAL_TELEOP
    MODE_SHARED_TRAJECTORY = ControllerMode.MODE_SHARED_TRAJECTORY
    MODE_AUTO_TRAJECTORY = ControllerMode.MODE_AUTO_TRAJECTORY
    MODE_EXTERNAL_SETPOINT_0 = ControllerMode.MODE_EXTERNAL_SETPOINT_0
    MODE_EXTERNAL_SETPOINT_1 = ControllerMode.MODE_EXTERNAL_SETPOINT_1
    MODE_EXTERNAL_SETPOINT_2 = ControllerMode.MODE_EXTERNAL_SETPOINT_2
    MODE_EXTERNAL_SETPOINT_3 = ControllerMode.MODE_EXTERNAL_SETPOINT_3
    MODE_EXTERNAL_SETPOINT_4 = ControllerMode.MODE_EXTERNAL_SETPOINT_4
    MODE_EXTERNAL_SETPOINT_5 = ControllerMode.MODE_EXTERNAL_SETPOINT_5
    MODE_EXTERNAL_SETPOINT_6 = ControllerMode.MODE_EXTERNAL_SETPOINT_6
    MODE_EXTERNAL_SETPOINT_7 = ControllerMode.MODE_EXTERNAL_SETPOINT_7
    MODE_CUSTOM_MODE_0 = ControllerMode.MODE_CUSTOM_MODE_0
    MODE_SUCCESS = ControllerMode.MODE_SUCCESS
    
    def __init__(self, arm_controller_ns = "", rox_robot = None ):
                                        
        if isinstance(rox_robot, rox.Robot):
            self.rox_robot = rox_robot
        elif isinstance(rox_robot, basestring):
            self.rox_robot = rox_urdf.robot_from_parameter_server(rox_robot)
        elif rox_robot is None:
            self.rox_robot = rox_urdf.robot_from_parameter_server()
        else:
            raise AssertionError("Invalid parameter type for rox_robot")
        
        set_controller_mode_name = rospy.names.ns_join(arm_controller_ns, "set_controller_mode")        
        self._set_controller_mode=rospy.ServiceProxy(set_controller_mode_name, SetControllerMode)
        self._arm_controller_ns = arm_controller_ns
        joint_trajectory_action_name = rospy.names.ns_join(arm_controller_ns, "joint_trajectory_action")
        self.joint_trajectory_action=actionlib.SimpleActionClient(joint_trajectory_action_name,FollowJointTrajectoryAction)
        self.controller_state_name = rospy.names.ns_join(arm_controller_ns, "controller_state")
        
    def set_controller_mode(self, mode, speed_scalar=1.0, ft_bias=[], ft_threshold=[]):
        req=SetControllerModeRequest()
        req.mode.mode=mode
        req.speed_scalar=speed_scalar
        req.ft_bias=ft_bias
        req.ft_stop_threshold=ft_threshold
        
        res=self._set_controller_mode(req)
        if (res.error_code.mode != ControllerMode.MODE_SUCCESS): raise Exception("Could not set controller mode")
        
    def subscribe_controller_state(self, cb):
        controller_state_name = rospy.names.ns_join(self.arm_controller_ns, "controller_state")
        return rospy.Subscriber(controller_state_name, ControllerState, cb)
                    
    def compute_fk(self, joint = None):
        
        if joint is None:
            joint=self.get_current_joint_values()
        
        return rox.fwdkin(self.rox_robot, joint)
    
    def get_current_joint_values(self, timeout=1):
        joint_state_msg=rospy.wait_for_message("joint_states", JointState, timeout)
        res = np.zeros((len(self.rox_robot.joint_names),),dtype=np.float64)
        for i in xrange(len(res)):
            j = joint_state_msg.name.index(self.rox_robot.joint_names[i])
            res[i] = joint_state_msg.position[j] 
        return res
    
    def get_current_pose_msg(self):
        return self.compute_fk()        
    
    def _fill_joint_trajectory_action_goal(self, joint_trajectory, path_tolerance, goal_tolerance, goal_time_tolerance):
        n_joints = len(self.rox_robot.joint_names)
        if not isinstance(path_tolerance, list):
            path_tolerance1 = []
            for n in self.rox_robot.joint_names:
                path_tolerance1.append(JointTolerance(n, path_tolerance, 0, 0))
            path_tolerance = path_tolerance1
        
        if not isinstance(goal_tolerance, list):
            goal_tolerance1 = []
            for n in self.rox_robot.joint_names:
                goal_tolerance1.append(JointTolerance(n, goal_tolerance, 0, 0))
            goal_tolerance = goal_tolerance1
        
        if not isinstance(goal_time_tolerance, rospy.Duration):
            goal_time_tolerance = rospy.Duration(goal_time_tolerance)
        
        return FollowJointTrajectoryGoal(joint_trajectory, path_tolerance, goal_tolerance, goal_time_tolerance)
            
    def execute_trajectory(self, joint_trajectory, path_tolerance = 0.01, goal_tolerance = 0.04, goal_time_tolerance = 0.5, \
                            execute_timeout = rospy.Duration(30), preempt_timeout = rospy.Duration(5), 
                            ft_stop = False):
        
        goal = self._fill_joint_trajectory_action_goal(joint_trajectory, path_tolerance, goal_tolerance, goal_time_tolerance)
        
        res = self.joint_trajectory_action.send_goal_and_wait(goal, execute_timeout, preempt_timeout)
        
        if res != actionlib.GoalStatus.SUCCEEDED:
           
            if not ft_stop:
                raise Exception("Trajectory execution failed")
            msg = rospy.wait_for_message(self.controller_state_name, ControllerState)
            if msg.mode.mode == self.MODE_FT_THRESHOLD_VIOLATION:
                return
            raise Exception("Trajectory execution failed")
        
    def async_execute_trajectory(self, joint_trajectory, done_cb, path_tolerance = 0.01, goal_tolerance = 0.04, goal_time_tolerance = 0.5, \
                            execute_timeout = rospy.Duration(30), preempt_timeout = rospy.Duration(5), 
                            ft_stop = False):
                        
        def action_done(status, result):
            if status == actionlib.GoalStatus.SUCCEEDED:
                done_cb(None)
                return
           
            if not ft_stop:
                done_cb("Trajectory execution failed")
                return
            msg = rospy.wait_for_message(self.controller_state_name, ControllerState)
            if msg.mode.mode == self.MODE_FT_THRESHOLD_VIOLATION:
                done_cb(None)
                return
            done_cb(Exception("Trajectory execution failed"))
            
        goal = self._fill_joint_trajectory_action_goal(joint_trajectory, path_tolerance, goal_tolerance, goal_time_tolerance)   
        self.joint_trajectory_action.send_goal(goal, done_cb = action_done)
            
    def stop_trajectory(self):
        self.joint_trajectory_action.cancel_all_goals()
