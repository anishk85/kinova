#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import signal
from geometry_msgs.msg import Pose, PointStamped
from pynput import keyboard

# Global variable for last known coordinates
last_known_target = None
threshold = 0.02  # Minimum change required to update the target


def signal_handler(sig, frame):
    rospy.loginfo("Shutting down teleoperation node...")
    moveit_commander.roscpp_shutdown()
    sys.exit(0)

# Callback function for the YOLO world coordinates topic

# Callback function for the YOLO world coordinates topic
def yolo_callback(msg):
    global last_known_target

    new_target = (msg.point.x, msg.point.y, msg.point.z)

    # Update only if the difference is significant
    if last_known_target is None or any(
        abs(new - old) > threshold for new, old in zip(new_target, last_known_target)
    ):
        last_known_target = new_target
        rospy.loginfo(f"Updated target: x={msg.point.x}, y={msg.point.y}, z={msg.point.z}")
    else:
        # rospy.loginfo("Received target, but change is too small. Ignoring.")
        pass

# Function to log the current position of the robot's end effector
def log_current_position(move_group):
    current_pose = move_group.get_current_pose().pose
    rospy.loginfo(f"Current Position: x={current_pose.position.x:.3f}, y={current_pose.position.y:.3f}, z={current_pose.position.z:.3f}")

# Function to move the robot towards a target position with tolerance
def move_robot_to_target(move_group, target_x, target_y, target_z, tolerance=0.05):
    rospy.loginfo("Starting to move towards the target...")
    while not rospy.is_shutdown():
        current_pose = move_group.get_current_pose().pose
        current_x, current_y, current_z = current_pose.position.x, current_pose.position.y, current_pose.position.z
        
        # Calculate differences
        dx, dy, dz = target_x - current_x, target_y - current_y, target_z - current_z

        log_current_position(move_group)
        rospy.loginfo(f"Target Position: x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}")

        # Check if within tolerance
        if abs(dx) < tolerance and abs(dy) < tolerance and abs(dz) < tolerance:
            rospy.loginfo("Reached target within tolerance.")
            return
        
        # Move in small steps
        step_size = 0.02
        move_robot(
            move_group,
            dx=min(step_size, max(-step_size, dx)),
            dy=min(step_size, max(-step_size, dy)),
            dz=min(step_size, max(-step_size, dz)),
        )

# Function to move the robot by a step
def move_robot(move_group, dx=0.0, dy=0.0, dz=0.0, max_retries=5, tolerance=0.01):
    retries = 0
    while not rospy.is_shutdown():
        try:
            current_pose = move_group.get_current_pose().pose
            target_pose = Pose()
            target_pose.position.x = current_pose.position.x + dx
            target_pose.position.y = current_pose.position.y + dy
            target_pose.position.z = current_pose.position.z + dz
            target_pose.orientation = current_pose.orientation

            if abs(dx) < tolerance and abs(dy) < tolerance and abs(dz) < tolerance:
                rospy.loginfo("Movement within tolerance.")
                return
            
            if abs(dx) < 0.002 and abs(dy) < 0.002 and abs(dz) < 0.002:
                rospy.loginfo("Movement is too small, skipping.")
                return

            rospy.loginfo(f"Moving: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
            move_group.set_pose_target(target_pose)
            success = move_group.go(wait=True)

            if success:
                rospy.loginfo("Step successful.")
                move_group.stop()
                move_group.clear_pose_targets()
                log_current_position(move_group)
                return
            else:
                rospy.logwarn("Retrying...")
                retries += 1
                if retries >= max_retries:
                    rospy.logerr("Max retries reached. Aborting movement.")
                    break

        except Exception as e:
            rospy.logerr(f"Exception: {e}")
            retries += 1
            if retries >= max_retries:
                rospy.logerr("Max retries reached. Aborting movement.")
                break
        
        rospy.sleep(1)

# Main function
def main():
    global last_known_target
    signal.signal(signal.SIGINT, signal_handler)
    
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('teleop_move_arm', anonymous=True)
    
    move_group = moveit_commander.MoveGroupCommander("arm")
    move_group.allow_replanning(True)
    move_group.set_max_acceleration_scaling_factor(0.5)
    # move_group.set_planning_time(10)

    rospy.Subscriber("/yolo/world_coordinates", PointStamped, yolo_callback)
    
    rospy.loginfo("Press 't' to move towards the last known detected object.")
    
    def on_press(key):
        try:
            if key.char == 't':
                rospy.loginfo("Auto-moving towards the last known object...")
                if last_known_target:
                    target_x, target_y, target_z = last_known_target
                    move_robot_to_target(move_group, target_x, target_y, target_z, tolerance=0.05)
                else:
                    rospy.loginfo("No known target yet.")
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        rospy.spin()
        listener.join()

    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
