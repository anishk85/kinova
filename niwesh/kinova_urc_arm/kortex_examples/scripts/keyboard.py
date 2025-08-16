#!/usr/bin/env python3

import rospy
import logging
import os
import json
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge
from tf2_geometry_msgs import do_transform_point
import tf2_ros
import cv2
from ultralytics import YOLO
import threading
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class RoboticTypingController:
    
    def __init__(self):
        rospy.init_node("robotic_typing_controller", log_level=rospy.INFO)
        logging.basicConfig(level=logging.INFO)

        rospy.loginfo("🤖 Initializing Robotic Typing Controller...")

        self.robot_namespace = "my_gen3"  
        self.base_frame = f'{self.robot_namespace}/base_link'
        self.camera_frame = 'camera_link'

        
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.color_topic = '/camera/color/image_raw'
        self.camera_info_topic = '/camera/color/camera_info'

        self.press_depth = 0.003
        self.dwell_time = 0.2
        self.approach_height = 0.02
        self.safe_height = 0.05

        self.setup_vision()
        
        self.load_keyboard_layout()

        self.setup_moveit()

        self.bridge = CvBridge()
        self.setup_subscribers_publishers()

        self.keypoints_3d = {}
        self.typing_active = False
        self.keyboard_detected = False
        self.depth_image = None
        self.color_image = None
        self.camera_info = None
        self.current_joint_state = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("✅ Robotic Typing Controller initialized successfully!")


    def setup_vision(self):
        try:
            rospy.loginfo("🔧 Setting up vision components...")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '../src/trained_yolov8n.pt')
            
            if not os.path.exists(model_path):
                rospy.logwarn("YOLO model not found, using default detection")
                self.model = None
            else:
                self.model = YOLO(model_path)
                logging.getLogger("ultralytics").setLevel(logging.ERROR)
            
            rospy.loginfo("✅ Vision setup complete.")
            
        except Exception as e:
            rospy.logwarn(f"Vision setup warning: {e}")
            self.model = None
    def load_keyboard_layout(self):
        """Load keyboard layout from JSON file"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, 'keyboard_layout.json')
            
            if not os.path.exists(json_path):
                rospy.logwarn(f"Keyboard layout file not found at {json_path}")
                # Create a basic layout
                return
                
            with open(json_path, 'r') as f:
                self.keyboard_layout = json.load(f)
                rospy.loginfo(f"✅ Loaded keyboard layout with {len(self.keyboard_layout)} keys")
                
        except Exception as e:
            rospy.logerr(f"Error loading keyboard layout: {e}")

    def setup_moveit(self):
            """Initialize MoveIt components"""
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
                
                # Configure MoveIt for precise typing
                self.configure_moveit_for_typing()
                
                rospy.loginfo("✅ MoveIt setup complete.")
                rospy.loginfo(f"   Planning Group: arm")
                rospy.loginfo(f"   Planning Frame: {self.move_group.get_planning_frame()}")
                rospy.loginfo(f"   End Effector: {self.move_group.get_end_effector_link()}")

            except Exception as e:
                rospy.logfatal(f"❌ MoveIt setup failed: {e}")
                raise

    def configure_moveit_for_typing(self):
        """Configure MoveIt parameters for precise typing"""
        self.move_group.set_planning_time(10.0)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.allow_replanning(True)
        
        # self.move_group.set_max_velocity_scaling_factor(0.3)  # 30% max speed
        # self.move_group.set_max_acceleration_scaling_factor(0.3)  # 30% max acceleration
        
        self.move_group.set_goal_position_tolerance(0.002)  
        self.move_group.set_goal_orientation_tolerance(0.1)  
        
        self.move_group.set_planner_id("RRTConnect")



    def setup_subscribers_publishers(self):
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.color_topic, Image, self.color_callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(f"/{self.robot_namespace}/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/type_text", String, self.type_text_callback)
        
        self.typing_status_pub = rospy.Publisher("/typing_status", String, queue_size=10)
        self.key_coordinates_pub = rospy.Publisher("/key_coordinates_3d", String, queue_size=10)
        self.annotated_image_pub = rospy.Publisher("/annotated_keyboard", Image, queue_size=10)
        self.trajectory_pub = rospy.Publisher(
            f"/{self.robot_namespace}/move_group/display_planned_path", 
            DisplayTrajectory, queue_size=10
        )
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")

    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.process_keyboard_detection()
        except Exception as e:
            rospy.logerr(f"Color callback error: {e}")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def joint_state_callback(self, msg):
        self.current_joint_state = msg
    def publish_status(self, message):
        try:
            status_msg = String()
            status_msg.data = message
            self.typing_status_pub.publish(status_msg)
            rospy.loginfo(f"Status: {message}")
        except Exception as e:
            rospy.logerr(f"Status publish error: {e}")


    def process_keyboard_detection(self):
        if self.color_image is None or self.depth_image is None or self.camera_info is None:
            return

        try:
            keypoints_2d, keyboard_bbox = self.detect_keyboard_keys(self.color_image)
            
            if not keypoints_2d:
                return

            keypoints_3d_camera = self.calculate_3d_positions(keypoints_2d)
            
            if not keypoints_3d_camera:
                return

            keypoints_3d_base = {}
            for key, pos_3d_cam in keypoints_3d_camera.items():
                pos_3d_base = self.transform_to_base_frame(pos_3d_cam)
                if pos_3d_base is not None:
                    keypoints_3d_base[key] = {
                        'position': pos_3d_base.tolist(),
                        'depth': pos_3d_cam[2]
                    }

            if keypoints_3d_base:
                self.keypoints_3d = keypoints_3d_base
                self.keyboard_detected = True
                
                self.key_coordinates_pub.publish(
                    String(data=json.dumps({"keys": keypoints_3d_base}))
                )
                
                rospy.loginfo_throttle(10, f"✅ Detected {len(keypoints_3d_base)} keys in workspace")

            self.publish_annotated_image(keypoints_2d, keyboard_bbox)

        except Exception as e:
            rospy.logerr(f"Keyboard detection error: {e}")
    def detect_keyboard_keys(self, image):
        keypoints_2d = {}
        keyboard_bbox = []
        
        try:
            if self.model is not None:
                # Use YOLO detection
                results = self.model(image, stream=True, conf=0.4)
                
                for result in results:
                    if result.boxes:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            keyboard_bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            
                            # Scale keyboard layout to detected region
                            self.scale_keyboard_to_detection(keypoints_2d, x1, y1, x2, y2)
                            break
                    break
            else:
                # Fallback: assume keyboard covers central region of image
                h, w = image.shape[:2]
                x1, y1 = w * 0.2, h * 0.3  # Assume keyboard in center 60% width, 40% height
                x2, y2 = w * 0.8, h * 0.7
                keyboard_bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                self.scale_keyboard_to_detection(keypoints_2d, x1, y1, x2, y2)

        except Exception as e:
            rospy.logerr(f"Keyboard detection error: {e}")

        return keypoints_2d, keyboard_bbox
    def scale_keyboard_to_detection(self, keypoints_2d, x1, y1, x2, y2):
        # Reference keyboard dimensions (pixels)
        ref_width, ref_height = 650, 200
        
        # Detected keyboard dimensions  
        det_width = x2 - x1
        det_height = y2 - y1
        
        # Scale factors
        scale_x = det_width / ref_width
        scale_y = det_height / ref_height
        
        # Scale each key position
        for key, (ref_x, ref_y) in self.keyboard_layout.items():
            scaled_x = x1 + (ref_x * scale_x)
            scaled_y = y1 + (ref_y * scale_y)
            keypoints_2d[key] = [int(scaled_x), int(scaled_y)]
    def calculate_3d_positions(self, keypoints_2d):
        keypoints_3d = {}
        
        if self.camera_info is None:
            return keypoints_3d

        # Camera intrinsics
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4] 
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        for key, (x, y) in keypoints_2d.items():
            try:
                if 0 <= x < self.depth_image.shape[1] and 0 <= y < self.depth_image.shape[0]:
                    # Get depth (convert mm to meters)
                    depth = self.depth_image[int(y), int(x)] / 1000.0
                    
                    if 0.1 < depth < 2.0:  # Valid depth range
                        # Convert to 3D in camera frame
                        X = (x - cx) * depth / fx
                        Y = (y - cy) * depth / fy
                        Z = depth
                        
                        keypoints_3d[key] = [X, Y, Z]
                        
            except Exception as e:
                rospy.logwarn_throttle(10, f"3D position error for {key}: {e}")

        return keypoints_3d

    def transform_to_base_frame(self, point_camera):
        try:
            # Create point in camera frame
            camera_point = PointStamped()
            camera_point.header.frame_id = self.camera_frame
            camera_point.header.stamp = rospy.Time(0)
            camera_point.point.x = point_camera[0]
            camera_point.point.y = point_camera[1]  
            camera_point.point.z = point_camera[2]
            
            # Get transform
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            
            # Transform point
            base_point = do_transform_point(camera_point, transform)
            
            return np.array([base_point.point.x, base_point.point.y, base_point.point.z])
            
        except Exception as e:
            rospy.logerr_throttle(10, f"Transform error: {e}")
            return None
    def publish_annotated_image(self, keypoints_2d, keyboard_bbox):
        if self.color_image is None:
            return
            
        try:
            annotated = self.color_image.copy()
            
            if keyboard_bbox:
                pts = np.array(keyboard_bbox, np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
            
            for key, (x, y) in keypoints_2d.items():
                color = (0, 255, 0) if key in self.keypoints_3d else (0, 0, 255)
                cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
                cv2.putText(annotated, key, (int(x)+5, int(y)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            status = f"Typing Ready - 3D Keys: {len(self.keypoints_3d)}"
            cv2.putText(annotated, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            self.annotated_image_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Annotated image error: {e}")

    def type_text_callback(self, msg):
        if self.typing_active:
            rospy.logwarn("Typing in progress, ignoring request")
            return

        text = msg.data.strip()
        if not text:
            return

        rospy.loginfo(f"📝 Typing request: '{text}'")
        
        typing_thread = threading.Thread(target=self.execute_typing, args=(text,))
        typing_thread.daemon = True
        typing_thread.start()


    def execute_typing(self, text):
        self.typing_active = True
        self.publish_status(f"Starting to type: '{text}'")

        try:
            if not self.keypoints_3d:
                self.publish_status("❌ No keyboard detected")
                return

            success_count = 0
            total_chars = len(text)
            
            for i, char in enumerate(text):
                if rospy.is_shutdown():
                    break
                    
                self.publish_status(f"Typing {i+1}/{total_chars}: '{char}'")
                
                key_name = self.char_to_key(char)
                
                if key_name and key_name in self.keypoints_3d:
                    if self.type_key(key_name):
                        success_count += 1
                        rospy.loginfo(f"✅ Typed '{char}'")
                    else:
                        rospy.logwarn(f"❌ Failed to type '{char}'")
                else:
                    rospy.logwarn(f"⚠️ Key '{key_name}' not available for '{char}'")
                
                rospy.sleep(0.5)

            self.publish_status(f"✅ Typing complete: {success_count}/{total_chars}")

        except Exception as e:
            rospy.logerr(f"Typing execution error: {e}")
            self.publish_status(f"❌ Typing error: {str(e)}")
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

    def type_key(self, key_name):
        if key_name not in self.keypoints_3d:
            return False

        try:
            target_pos = self.keypoints_3d[key_name]['position']
            rospy.loginfo(f"⌨️ Typing key '{key_name}' at {target_pos}")

            if self.type_key_cartesian(key_name, target_pos):
                return True
            if self._standard_planning(target_pos, self.move_group.get_current_pose().pose, 5.0):
                return True

            if self.type_key_joint_space(key_name, target_pos):
                return True

            if self.type_key_simple_pose(key_name, target_pos):
                return True

            rospy.logwarn(f"All typing methods failed for key '{key_name}'")
            return False

        except Exception as e:
            rospy.logerr(f"Key typing error for '{key_name}': {e}")
            return False
    def type_key_cartesian(self, key_name, target_pos):
        try:
            rospy.loginfo(f"🎯 Cartesian approach for '{key_name}'")
            
            current_pose = self.move_group.get_current_pose().pose
            
            waypoints = []
            
            approach_pose = Pose()
            approach_pose.position.x = target_pos[0]
            approach_pose.position.y = target_pos[1]
            approach_pose.position.z = target_pos[2] + self.approach_height
            approach_pose.orientation = current_pose.orientation
            waypoints.append(approach_pose)
            
            press_pose = Pose()
            press_pose.position.x = target_pos[0]
            press_pose.position.y = target_pos[1]
            press_pose.position.z = target_pos[2] - self.press_depth
            press_pose.orientation = current_pose.orientation
            waypoints.append(press_pose)
            
            retract_pose = Pose()
            retract_pose.position.x = target_pos[0]
            retract_pose.position.y = target_pos[1]  
            retract_pose.position.z = target_pos[2] + self.approach_height
            retract_pose.orientation = current_pose.orientation
            waypoints.append(retract_pose)
            
            rospy.loginfo("Computing Cartesian path...")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step=0.01, 
            )
            
            rospy.loginfo(f"Cartesian path: {fraction*100:.1f}% complete")
            
            if fraction > 0.8:  
                rospy.loginfo("Executing Cartesian trajectory...")
                success = self.move_group.execute(plan, wait=True)
                
                if success:
                    rospy.loginfo(f"✅ Cartesian typing successful for '{key_name}'")
                    rospy.sleep(self.dwell_time)  
                    self.move_group.stop()
                    return True
                else:
                    rospy.logwarn("Cartesian execution failed")
            else:
                rospy.logwarn(f"Cartesian path only {fraction*100:.1f}% valid")
                
        except Exception as e:
            rospy.logerr(f"Cartesian typing error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def _standard_planning(self, position_3d, current_pose, timeout):
        """Helper function for standard planning with more conservative settings"""
        target_pose = Pose()
        target_pose.position.x = position_3d[0]
        target_pose.position.y = position_3d[1]
        target_pose.position.z = position_3d[2]
        target_pose.orientation = current_pose.orientation

        rospy.loginfo("Setting up standard planning...")

        # Use move_group for planning (not self.arm_group)
        rospy.loginfo(f"Setting planning time: {timeout}s")
        self.move_group.set_planning_time(timeout)

        rospy.loginfo("Setting planning attempts: 5")
        self.move_group.set_num_planning_attempts(5)

        self.move_group.set_goal_position_tolerance(0.01)
        self.move_group.set_goal_orientation_tolerance(0.1)

        rospy.loginfo("Setting pose target...")
        self.move_group.set_pose_target(target_pose)

        rospy.loginfo("Starting motion planning...")
        plan = self.move_group.plan()
        success = plan[0] if isinstance(plan, tuple) else plan

        if not success:
            rospy.logerr("Standard planning failed!")
            rospy.loginfo("Trying with relaxed constraints...")
            self.move_group.set_goal_position_tolerance(0.02)
            self.move_group.set_goal_orientation_tolerance(0.2)
            plan = self.move_group.plan()
            success = plan[0] if isinstance(plan, tuple) else plan
            if success:
                rospy.loginfo("Planning succeeded with relaxed constraints!")
            else:
                rospy.logerr("Planning failed even with relaxed constraints!")

        if success:
            if isinstance(plan, tuple):
                exec_success = self.move_group.execute(plan[1], wait=True)
            else:
                exec_success = self.move_group.execute(plan, wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return exec_success
        else:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return False

    def type_key_joint_space(self, key_name, target_pos):
        try:
            rospy.loginfo(f"🔧 Joint space approach for '{key_name}'")
            
            current_pose = self.move_group.get_current_pose().pose
            
            approach_pose = Pose()
            approach_pose.position.x = target_pos[0]
            approach_pose.position.y = target_pos[1]
            approach_pose.position.z = target_pos[2] + self.approach_height
            approach_pose.orientation = current_pose.orientation
            
            self.move_group.set_pose_target(approach_pose)
            
            rospy.loginfo("Planning approach movement...")
            approach_plan = self.move_group.plan()
            
            if approach_plan[0]: 
                rospy.loginfo("Executing approach...")
                if self.move_group.execute(approach_plan[1], wait=True):
                    rospy.loginfo("✅ Reached approach position")
                    
                    press_pose = Pose()
                    press_pose.position.x = target_pos[0]
                    press_pose.position.y = target_pos[1]
                    press_pose.position.z = target_pos[2] - self.press_depth
                    press_pose.orientation = current_pose.orientation
                    
                    self.move_group.set_pose_target(press_pose)
                    press_plan = self.move_group.plan()
                    
                    if press_plan[0]:
                        rospy.loginfo("Executing key press...")
                        if self.move_group.execute(press_plan[1], wait=True):
                            rospy.loginfo(f"✅ Pressed key '{key_name}'")
                            rospy.sleep(self.dwell_time)
                            
                            self.move_group.set_pose_target(approach_pose)
                            retract_plan = self.move_group.plan()
                            
                            if retract_plan[0]:
                                self.move_group.execute(retract_plan[1], wait=True)
                                rospy.loginfo("✅ Retracted from key")
                            
                            self.move_group.stop()
                            self.move_group.clear_pose_targets()
                            return True
                    else:
                        rospy.logwarn("Press movement planning failed")
                else:
                    rospy.logwarn("Approach movement execution failed")
            else:
                rospy.logwarn("Approach movement planning failed")
                
        except Exception as e:
            rospy.logerr(f"Joint space typing error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def type_key_simple_pose(self, key_name, target_pos):
        try:
            rospy.loginfo(f"🎯 Simple pose approach for '{key_name}'")
            
            current_pose = self.move_group.get_current_pose().pose
            
            target_pose = Pose()
            target_pose.position.x = target_pos[0]
            target_pose.position.y = target_pos[1]
            target_pose.position.z = target_pos[2] + 0.005  
            target_pose.orientation = current_pose.orientation
            
            self.move_group.set_pose_target(target_pose)
            self.move_group.set_planning_time(5.0)
            
            plan = self.move_group.plan()
            
            if plan[0]:
                success = self.move_group.execute(plan[1], wait=True)
                if success:
                    rospy.loginfo(f"✅ Simple pose typing for '{key_name}'")
                    rospy.sleep(self.dwell_time)
                    self.move_group.stop()
                    self.move_group.clear_pose_targets()
                    return True
            
        except Exception as e:
            rospy.logerr(f"Simple pose typing error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    # def move_to_home_position(self):
    #     """Move robot to home/ready position"""
    #     try:
    #         rospy.loginfo("🏠 Moving to home position...")
    #         self.move_group.set_named_target("home")
    #         return self.move_group.go(wait=True)
    #     except Exception as e:
    #         rospy.logerr(f"Home movement error: {e}")
    #         return False

    def get_robot_status(self):
        try:
            status = {
                'timestamp': rospy.Time.now().to_sec(),
                'keyboard_detected': self.keyboard_detected,
                'num_keys_detected': len(self.keypoints_3d),
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
                
            if self.current_joint_state:
                status['joint_states'] = {
                    'names': list(self.current_joint_state.name),
                    'positions': list(self.current_joint_state.position)
                }
                
            return status
            
        except Exception as e:
            rospy.logerr(f"Robot status error: {e}")
            return {}

     def emergency_stop(self):
        rospy.logwarn("🛑 Emergency stop!")
        try:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.typing_active = False
            self.publish_status("🛑 Emergency stop activated")
            rospy.logwarn("All robot motion stopped")
        except Exception as e:
            rospy.logerr(f"Emergency stop error: {e}")

     def shutdown_handler(self):
        rospy.loginfo("🔄 Shutting down Robotic Typing Controller...")
        
        try:
            self.typing_active = False
            
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            self.move_to_home_position()
            
            moveit_commander.roscpp_shutdown()
            
            rospy.loginfo("✅ Shutdown complete")
            
        except Exception as e:
            rospy.logerr(f"Shutdown error: {e}")

def main():
    try:
        controller = RoboticTypingController()
        
        rospy.on_shutdown(controller.shutdown_handler)
        
        rospy.loginfo("🤖 Robotic Typing Controller is ready!")
        rospy.loginfo("📝 Send text to '/type_text' topic to start typing")
        
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("👋 User interrupted")
    except Exception as e:
        rospy.logfatal(f"❌ Fatal error: {e}")
    finally:
        rospy.loginfo("🔄 Exiting...")

if __name__ == "__main__":
    main()

    


    



