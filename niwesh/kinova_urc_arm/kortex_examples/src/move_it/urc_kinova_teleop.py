#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi

class ExampleMoveItTrajectories(object):
    def __init__(self):
        super(ExampleMoveItTrajectories, self).__init__()
        
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('example_move_it_trajectories')

        try:
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                                moveit_msgs.msg.DisplayTrajectory, queue_size=20)
            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        except Exception as e:
            print(e)
            self.is_init_success = False
        else:
            self.is_init_success = True



    def get_cartesian_pose(self):
        return self.arm_group.get_current_pose().pose

    def reach_relative_position(self, dx, dy, dz, tolerance=0.01):
        arm_group = self.arm_group
        current_pose = arm_group.get_current_pose().pose

        # Update position
        current_pose.position.x += dx
        current_pose.position.y += dy
        current_pose.position.z += dz

        # Set tolerance
        arm_group.set_goal_position_tolerance(tolerance)

        # Set new pose target
        arm_group.set_pose_target(current_pose)
        rospy.loginfo(f"Moving to relative position dx: {dx}, dy: {dy}, dz: {dz}")
        
        # Plan and execute
        return arm_group.go(wait=True)


def main():
    example = ExampleMoveItTrajectories()

    if example.is_init_success:
        rospy.loginfo("Moving to a relative position...")
        success = example.reach_relative_position(dx=0.1, dy=0.1, dz=0.1)
        print("Success:", success)
    else:
        rospy.logerr("Initialization failed.")

if __name__ == '__main__':
    main()
