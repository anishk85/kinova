#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, Constraints, PositionConstraint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
import shape_msgs.msg
import threading


class MotionPlanner:
    
    def __init__(self, robot_namespace):
        self.robot_namespace = robot_namespace
        
        # More conservative movement parameters to avoid acceleration limits
        self.press_depth = 0.015  # Reduced from 0.03 to 1.5cm
        self.dwell_time = 0.3
        self.approach_height = 0.02  # Increased to 2cm above key
        self.safe_height = 0.005
        
        self.typing_active = False
        
        self.workspace_bounds = {
            'x_min': 0.2, 'x_max': 0.8,   # 20cm to 80cm forward
            'y_min': -0.4, 'y_max': 0.4,  # ±40cm left/right  
            'z_min': 0.1, 'z_max': 0.5    # 10cm to 50cm height
        }
        
        self.setup_moveit()
        
    def setup_moveit(self):
        """Initialize MoveIt components with better trajectory parameters"""
        try:
            rospy.loginfo("🔧 Setting up MoveIt...")
            
            # Initialize MoveIt commander
            moveit_commander.roscpp_initialize([])
            
            # Robot and scene
            self.robot = moveit_commander.RobotCommander(
                robot_description=f"{self.robot_namespace}/robot_description",
                ns=self.robot_namespace
            )
            self.scene = moveit_commander.PlanningSceneInterface(ns=self.robot_namespace)

            # Get available groups
            available_groups = self.robot.get_group_names()
            rospy.loginfo(f"Available planning groups: {available_groups}")
            
            # Use 'arm' group (common for Kinova)
            self.move_group = moveit_commander.MoveGroupCommander(
                "arm",
                robot_description=f"{self.robot_namespace}/robot_description",
                ns=self.robot_namespace
            )
            
            # Configure MoveIt for precise typing with conservative parameters
            self.configure_moveit_for_typing()

            # Much more conservative velocity and acceleration for typing
            self.move_group.set_max_velocity_scaling_factor(0.05)  # 5% max speed (was 10%)
            self.move_group.set_max_acceleration_scaling_factor(0.05)  # 5% max acceleration (was 10%)
            
            rospy.loginfo("✅ MoveIt setup complete.")
            rospy.loginfo(f"   Planning Group: arm")
            rospy.loginfo(f"   Planning Frame: {self.move_group.get_planning_frame()}")
            rospy.loginfo(f"   End Effector: {self.move_group.get_end_effector_link()}")

        except Exception as e:
            rospy.logfatal(f"❌ MoveIt setup failed: {e}")
            raise

    def configure_moveit_for_typing(self):
        """Configure MoveIt parameters for precise typing with conservative settings"""
        self.move_group.set_planning_time(15.0)  # Increased planning time
        self.move_group.set_num_planning_attempts(15)  # More planning attempts
        self.move_group.allow_replanning(True)
        
        self.move_group.set_goal_position_tolerance(0.005)  # Relaxed tolerance (was 0.002)
        self.move_group.set_goal_orientation_tolerance(0.2)  # Relaxed orientation tolerance
        
        self.move_group.set_planner_id("RRTConnect")

    def execute_typing(self, text, available_coords, publish_status_callback):
        """Execute typing sequence for given text"""
        self.typing_active = True
        publish_status_callback(f"Starting to type: '{text}'")

        try:
            if not available_coords:
                publish_status_callback("❌ No keyboard coordinates available")
                return

            success_count = 0
            total_chars = len(text)
            
            for i, char in enumerate(text):
                if rospy.is_shutdown():
                    break
                    
                publish_status_callback(f"Typing {i+1}/{total_chars}: '{char}'")
                
                key_name = self.char_to_key(char)
                
                if key_name and key_name in available_coords:
                    if self.type_key(key_name, available_coords):
                        success_count += 1
                        rospy.loginfo(f"✅ Typed '{char}'")
                    else:
                        rospy.logwarn(f"❌ Failed to type '{char}'")
                else:
                    rospy.logwarn(f"⚠️ Key '{key_name}' not available for '{char}'")
                    rospy.loginfo(f"📋 Available keys: {list(available_coords.keys())}")
                
                rospy.sleep(0.8)  # Longer delay between keystrokes

            publish_status_callback(f"✅ Typing complete: {success_count}/{total_chars}")

        except Exception as e:
            rospy.logerr(f"Typing execution error: {e}")
            publish_status_callback(f"❌ Typing error: {str(e)}")
        finally:
            self.typing_active = False

    def char_to_key(self, char):
        if char == ' ':
            return 'SPACE'
        elif char == '\n':
            return 'ENTER'
        elif char == '\b':
            return 'BACKSPACE'
        elif char.isalnum():
            return char.upper()
        else:
            return char.upper()

    def type_key(self, key_name, available_coords):
        if key_name not in available_coords:
            return False

        try:
            target_pos = available_coords[key_name]['position']
            rospy.loginfo(f"⌨️ Typing key '{key_name}' at {target_pos}")

            # Try improved Cartesian approach first
            if self.type_key_improved_cartesian(key_name, target_pos):
                return True

            # Fallback to joint space approach
            if self.type_key_joint_space(key_name, target_pos):
                return True

            rospy.logwarn(f"All typing methods failed for key '{key_name}'")
            return False

        except Exception as e:
            rospy.logerr(f"Key typing error for '{key_name}': {e}")
            return False

    def type_key_improved_cartesian(self, key_name, target_pos):
        """
        MODIFIED Cartesian approach with a horizontal press motion.
        """
        try:
            rospy.loginfo(f"🎯 Horizontal Cartesian approach for '{key_name}'")
            current_pose = self.move_group.get_current_pose().pose

            # --- Define Horizontal Motion Parameters ---
            # How far in front of the key to start the motion (in meters)
            horizontal_offset = 0.05  # 5 cm
            # How high above the key surface to move (in meters)
            vertical_clearance = 0.01 # 1 cm
            z_press_offset = 0.05

            waypoints = []

            # 1. Move to a safe point: IN FRONT of and ABOVE the key
            safe_pose = Pose()
            safe_pose.position.x = target_pos[0] - horizontal_offset # Move back along X-axis
            safe_pose.position.y = target_pos[1]
            safe_pose.position.z = target_pos[2] + vertical_clearance # Move up
            safe_pose.orientation = current_pose.orientation
            waypoints.append(safe_pose)

            # 2. Move to pre-press position: IN FRONT of the key, at the correct height
            pre_press_pose = Pose()
            pre_press_pose.position.x = target_pos[0]  # Still back along X
            pre_press_pose.position.y = target_pos[1]
            pre_press_pose.position.z = target_pos[2] # At target height
            pre_press_pose.orientation = current_pose.orientation
            waypoints.append(pre_press_pose)

            # 3. Press the key: Move FORWARD horizontally
            press_pose = Pose()
            press_pose.position.x = target_pos[0]  # Move to target X
            press_pose.position.y = target_pos[1]
            press_pose.position.z = target_pos[2] + z_press_offset
            press_pose.orientation = current_pose.orientation
            waypoints.append(press_pose)

            # 4. Release: Move BACK horizontally to the pre-press position
            waypoints.append(pre_press_pose)
            
            # 5. Retract: Move back to the safe position
            waypoints.append(safe_pose)

            rospy.loginfo("Computing horizontal Cartesian path...")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step=0.002,
                 # Disable jump threshold
            )

            rospy.loginfo(f"Cartesian path: {fraction*100:.1f}% complete")

            if fraction > 0.7:
                if hasattr(plan, 'joint_trajectory'):
                    self.scale_trajectory_timing(plan.joint_trajectory)
                
                rospy.loginfo("Executing horizontal Cartesian trajectory...")
                success = self.move_group.execute(plan, wait=True)
                
                if success:
                    rospy.loginfo(f"✅ Horizontal Cartesian typing successful for '{key_name}'")
                    # Dwell time is effectively handled by the motion itself, but a small sleep can help
                    rospy.sleep(0.1) 
                    self.move_group.stop()
                    return True
                else:
                    rospy.logwarn("Horizontal Cartesian execution failed")
            else:
                rospy.logwarn(f"Horizontal Cartesian path only {fraction*100:.1f}% valid, trying step-by-step")
                if self.execute_waypoints_stepwise(waypoints):
                    return True

        except Exception as e:
            rospy.logerr(f"Horizontal Cartesian typing error: {e}")
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()

        return False

    def scale_trajectory_timing(self, joint_trajectory):
        """Scale trajectory timing to reduce velocities and accelerations"""
        try:
            # Scale time stamps to make movement slower
            time_scale_factor = 3.0  # Make 3x slower
            
            for i, point in enumerate(joint_trajectory.points):
                if i == 0:
                    continue  # Skip first point
                
                # Scale time
                point.time_from_start = rospy.Duration(
                    point.time_from_start.to_sec() * time_scale_factor
                )
                
                # Reduce velocities
                if point.velocities:
                    point.velocities = [v / time_scale_factor for v in point.velocities]
                
                # Reduce accelerations  
                if point.accelerations:
                    point.accelerations = [a / (time_scale_factor * time_scale_factor) for a in point.accelerations]
                    
        except Exception as e:
            rospy.logerr(f"Trajectory scaling error: {e}")

    def execute_waypoints_stepwise(self, waypoints):
        """Execute waypoints one by one instead of as a single trajectory"""
        try:
            rospy.loginfo("Executing waypoints step-by-step...")
            
            for i, waypoint in enumerate(waypoints):
                rospy.loginfo(f"Moving to waypoint {i+1}/{len(waypoints)}")
                
                self.move_group.set_pose_target(waypoint)
                self.move_group.set_planning_time(10.0)
                
                plan = self.move_group.plan()
                
                if plan[0]:  # If planning succeeded
                    success = self.move_group.execute(plan[1], wait=True)
                    if success:
                        rospy.loginfo(f"✅ Reached waypoint {i+1}")
                        
                        # Add dwell time for press waypoint
                        if i == 2:  # Press waypoint (0-indexed)
                            rospy.sleep(self.dwell_time)
                    else:
                        rospy.logwarn(f"❌ Failed to reach waypoint {i+1}")
                        return False
                else:
                    rospy.logwarn(f"❌ Planning failed for waypoint {i+1}")
                    return False
                
                rospy.sleep(0.2)  # Small delay between waypoints
            
            rospy.loginfo("✅ Step-by-step execution completed")
            return True
            
        except Exception as e:
            rospy.logerr(f"Step-by-step execution error: {e}")
            return False
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()

    def type_key_joint_space(self, key_name, target_pos):
        """
        MODIFIED: Joint space typing constrained to the workspace with better logging.
        """
        try:
            # --- 1. Create and apply workspace path constraints ---
            constraints = Constraints()
            constraints.name = "typing_workspace_constraint"

            pc = PositionConstraint()
            pc.header.frame_id = self.move_group.get_planning_frame()
            pc.link_name = self.move_group.get_end_effector_link()
            pc.weight = 1.0

            box_bvolume = moveit_msgs.msg.BoundingVolume()
            box_primitive = shape_msgs.msg.SolidPrimitive()
            box_primitive.type = box_primitive.BOX
            
            box_dims = [
                self.workspace_bounds['x_max'] - self.workspace_bounds['x_min'],
                self.workspace_bounds['y_max'] - self.workspace_bounds['y_min'],
                self.workspace_bounds['z_max'] - self.workspace_bounds['z_min']
            ]
            box_primitive.dimensions = box_dims
            
            box_pose = Pose()
            box_pose.position.x = (self.workspace_bounds['x_max'] + self.workspace_bounds['x_min']) / 2
            box_pose.position.y = (self.workspace_bounds['y_max'] + self.workspace_bounds['y_min']) / 2
            box_pose.position.z = (self.workspace_bounds['z_max'] + self.workspace_bounds['z_min']) / 2
            box_pose.orientation.w = 1.0
            
            box_bvolume.primitives.append(box_primitive)
            box_bvolume.primitive_poses.append(box_pose)
            pc.constraint_region = box_bvolume
            
            constraints.position_constraints.append(pc)
            self.move_group.set_path_constraints(constraints)

            # --- 2. Proceed with the typing motion plan ---
            rospy.loginfo(f"🔧 Horizontal Joint Space approach for '{key_name}' with path constraints")
            
            current_pose = self.move_group.get_current_pose().pose
            self.move_group.set_max_velocity_scaling_factor(0.15)
            self.move_group.set_max_acceleration_scaling_factor(0.15)
            
            horizontal_offset = 0.05
            x_press_offset = 0.015

            approach_pose = Pose()
            approach_pose.position.x = target_pos[0] - horizontal_offset
            approach_pose.position.y = target_pos[1]
            approach_pose.position.z = target_pos[2]
            approach_pose.orientation = current_pose.orientation
            
            self.move_group.set_pose_target(approach_pose)
            self.move_group.set_planning_time(10.0)
            
            rospy.loginfo("Planning approach movement (to front of key)...")
            approach_plan = self.move_group.plan()
            
            # --- 3. Check for planning success and provide clear logs ---
            if approach_plan[0]: 
                rospy.loginfo("Executing approach...")
                if self.move_group.execute(approach_plan[1], wait=True):
                    rospy.loginfo("✅ Reached approach position")
                    
                    press_pose = Pose()
                    press_pose.position.x = target_pos[0] + x_press_offset
                    press_pose.position.y = target_pos[1]
                    press_pose.position.z = target_pos[2]
                    press_pose.orientation = current_pose.orientation
                    
                    self.move_group.set_pose_target(press_pose)
                    press_plan = self.move_group.plan()
                    
                    if press_plan[0]:
                        rospy.loginfo("Executing key press (moving forward)...")
                        if self.move_group.execute(press_plan[1], wait=True):
                            rospy.loginfo(f"✅ Pressed key '{key_name}'")
                            rospy.sleep(self.dwell_time)
                            
                            self.move_group.set_pose_target(approach_pose)
                            retract_plan = self.move_group.plan()
                            if retract_plan[0]:
                                self.move_group.execute(retract_plan[1], wait=True)
                                rospy.loginfo("✅ Retracted from key")
                            
                            return True # Typing was successful
                    else:
                        rospy.logwarn("!!! Press movement planning FAILED.")
                else:
                    rospy.logwarn("!!! Approach movement EXECUTION failed.")
            else:
                rospy.logwarn("!!! Approach planning FAILED. The robot could not find a valid path within the constraints. Try widening the workspace.")
                
        except Exception as e:
            rospy.logerr(f"Joint space typing error: {e}")
            
        finally:
            # --- 4. CRITICAL: Clear constraints and restore state ---
            self.move_group.clear_path_constraints()
            rospy.loginfo("Restoring default conservative typing speed (5%).")
            self.move_group.set_max_velocity_scaling_factor(0.05)
            self.move_group.set_max_acceleration_scaling_factor(0.05)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False # Return False if any part of the process fails

    def get_robot_status(self, current_joint_state):
        try:
            status = {
                'timestamp': rospy.Time.now().to_sec(),
                'typing_active': self.typing_active,
                'current_pose': None,
                'joint_states': None
            }
            
            try:
                current_pose = self.move_group.get_current_pose().pose
                status['current_pose'] = {
                    'position': [current_pose.position.x, current_pose.position.y, current_pose.position.z],
                    'orientation': [current_pose.orientation.x, current_pose.orientation.y, 
                                  current_pose.orientation.z, current_pose.orientation.w]
                }
            except:
                pass
                
            if current_joint_state:
                status['joint_states'] = {
                    'names': list(current_joint_state.name),
                    'positions': list(current_joint_state.position)
                }
                
            return status
            
        except Exception as e:
            rospy.logerr(f"Robot status error: {e}")
            return {}

    def emergency_stop(self, publish_status_callback):
        rospy.logwarn("🛑 Emergency stop!")
        try:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.typing_active = False
            publish_status_callback("🛑 Emergency stop activated")
            rospy.logwarn("All robot motion stopped")
        except Exception as e:
            rospy.logerr(f"Emergency stop error: {e}")

    def shutdown(self):
        rospy.loginfo("🔄 Shutting down Motion Planner...")
        
        try:
            self.typing_active = False
            
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            moveit_commander.roscpp_shutdown()
            
            rospy.loginfo("✅ Motion Planner shutdown complete")
            
        except Exception as e:
            rospy.logerr(f"Motion Planner shutdown error: {e}")