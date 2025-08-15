#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import tkinter as tk
import signal
from pynput import keyboard

# Signal handler for clean shutdown
def signal_handler(sig, frame):
    rospy.loginfo("Shutting down teleoperation node...")
    moveit_commander.roscpp_shutdown()
    sys.exit(0)

# Function to move the robot arm in specified directions
def move_robot(move_group, dx=0.0, dy=0.0, dz=0.0):
    current_pose = move_group.get_current_pose().pose

    target_pose = Pose()
    target_pose.position.x = current_pose.position.x + dx
    target_pose.position.y = current_pose.position.y + dy
    target_pose.position.z = current_pose.position.z + dz
    target_pose.orientation = current_pose.orientation  # Keep the same orientation

    move_group.set_pose_target(target_pose)
    move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    rospy.loginfo(f"Moved to: {move_group.get_current_pose().pose}")
    


# Class for Virtual Joystick control
class VirtualJoystick:
    def __init__(self, move_group):
        self.move_group = move_group
        self.step_scale = 0.1  # Scale joystick movement to robot motion

        self.root = tk.Tk()
        self.root.title("Virtual Joystick")

        self.canvas = tk.Canvas(self.root, width=300, height=300, bg="lightgray")
        self.canvas.pack()

        self.base = self.canvas.create_oval(50, 50, 250, 250, fill="white", outline="black")
        self.knob = self.canvas.create_oval(125, 125, 175, 175, fill="blue")

        self.canvas.bind("<B1-Motion>", self.drag_knob)
        self.canvas.bind("<ButtonRelease-1>", self.reset_knob)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.update_robot_motion)
        self.root.mainloop()



    def drag_knob(self, event):
        x, y = event.x, event.y
        cx, cy = 150, 150  # Center of the joystick base
        dx, dy = x - cx, y - cy
        distance = (dx**2 + dy**2)**0.5
        max_distance = 100

        if distance > max_distance:
            dx = dx / distance * max_distance
            dy = dy / distance * max_distance

        self.canvas.coords(self.knob, cx + dx - 25, cy + dy - 25, cx + dx + 25, cy + dy + 25)

    def reset_knob(self, event):
        self.canvas.coords(self.knob, 125, 125, 175, 175)

    def update_robot_motion(self):
        knob_coords = self.canvas.coords(self.knob)
        knob_x = (knob_coords[0] + knob_coords[2]) / 2
        knob_y = (knob_coords[1] + knob_coords[3]) / 2

        cx, cy = 150, 150
        dx = -(knob_x - cx) / 100 * self.step_scale
        dy = (knob_y - cy) / 100 * self.step_scale

        if abs(dx) > 0.01 or abs(dy) > 0.01:
            move_robot(self.move_group, dx=dx, dy=dy)

        self.root.after(100, self.update_robot_motion)

    def on_close(self):
        self.root.destroy()
        sys.exit(0)


# Keyboard press handler for Z-axis movement
def on_press(key, move_group):
    try:
        if key.char == 'w':
            rospy.loginfo("Moving up in +Z direction")
            move_robot(move_group, dz=0.01)
        elif key.char == 's':
            rospy.loginfo("Moving down in -Z direction")
            move_robot(move_group, dz=-0.01)
    except AttributeError:
        pass

# Main function
def main():
    signal.signal(signal.SIGINT, signal_handler)

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('kinova_gen3_teleop', anonymous=True)

    move_group = moveit_commander.MoveGroupCommander("arm")

    rospy.loginfo("""
    Teleoperation started. Use the following controls:
        - Virtual Joystick (GUI) for X and Y movements
        - Terminal keys:
            w → Move up (+Z direction)
            s → Move down (-Z direction)
    Press Ctrl+C to exit.
    """)

    with keyboard.Listener(on_press=lambda key: on_press(key, move_group)) as listener:
        VirtualJoystick(move_group)
        listener.join()

    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
