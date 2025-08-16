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
import tf.transformations as tf_trans

class VerticalKeyboardTypingController:
    
    def __init__(self):
        rospy.init_node("vertical_keyboard_typing_controller", log_level=rospy.INFO)
        logging.basicConfig(level=logging.INFO)

        rospy.loginfo("🤖 Initializing Vertical Keyboard Typing Controller...")

        self.robot_namespace = "my_gen3"  
        self.base_frame = f'{self.robot_namespace}/base_link'
        self.camera_frame = 'camera_link'

        # Camera topics
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.color_topic = '/camera/color/image_raw'
        self.camera_info_topic = '/camera/color/camera_info'

        # MODIFIED: Vertical keyboard typing parameters
        self.press_depth = 0.015  # Deeper press for pen on vertical surface
        self.dwell_time = 0.3     # Longer dwell for reliable contact
        self.approach_distance = 0.03  # Distance before touching keyboard
        self.safe_distance = 0.08      # Safe distance from keyboard
        
        # MODIFIED: Pen tip offset from end effector (adjust based on your pen holder)
        self.pen_tip_offset = [0.0, 0.0, -0.05]  # 5cm extension in negative Z (tool frame)

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
        self.keyboard_normal = np.array([0, 0, -1])  # MODIFIED: Vertical keyboard normal (pointing away from keyboard)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("✅ Vertical Keyboard Typing Controller initialized successfully!")

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
            
            moveit_commander.roscpp_initialize([])
            
            self.robot = moveit_commander.RobotCommander(
                robot_description=f"{self.robot_namespace}/robot_description",
                ns=self.robot_namespace
            )
            self.scene = moveit_commander.PlanningSceneInterface(ns=self.robot_namespace)

            available_groups = self.robot.get_group_names()
            rospy.loginfo(f"Available planning groups: {available_groups}")
            
            self.move_group = moveit_commander.MoveGroupCommander(
                "arm",
                robot_description=f"{self.robot_namespace}/robot_description",
                ns=self.robot_namespace
            )
            
            self.configure_moveit_for_vertical_typing()
            
            rospy.loginfo("✅ MoveIt setup complete.")
            rospy.loginfo(f"   Planning Group: arm")
            rospy.loginfo(f"   Planning Frame: {self.move_group.get_planning_frame()}")
            rospy.loginfo(f"   End Effector: {self.move_group.get_end_effector_link()}")

        except Exception as e:
            rospy.logfatal(f"❌ MoveIt setup failed: {e}")
            raise

    def configure_moveit_for_vertical_typing(self):
        """Configure MoveIt parameters for precise vertical keyboard typing"""
        self.move_group.set_planning_time(15.0)  # More time for complex vertical motions
        self.move_group.set_num_planning_attempts(15)
        self.move_group.allow_replanning(True)
        
        # MODIFIED: Slower speeds for precision with pen
        self.move_group.set_max_velocity_scaling_factor(0.2)  # 20% max speed
        self.move_group.set_max_acceleration_scaling_factor(0.2)  # 20% max acceleration
        
        # MODIFIED: Tighter tolerances for pen tip accuracy
        self.move_group.set_goal_position_tolerance(0.001)  # 1mm tolerance
        self.move_group.set_goal_orientation_tolerance(0.05)  # Stricter orientation
        
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

    # MODIFIED: Keyboard detection remains similar but processes for vertical orientation
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
                
                # MODIFIED: Estimate keyboard normal from key positions
                self.estimate_keyboard_normal(keypoints_3d_base)
                
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
                results = self.model(image, stream=True, conf=0.4)
                
                for result in results:
                    if result.boxes:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            keyboard_bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            
                            self.scale_keyboard_to_detection(keypoints_2d, x1, y1, x2, y2)
                            break
                    break
            else:
                # Fallback for vertical keyboard
                h, w = image.shape[:2]
                x1, y1 = w * 0.1, h * 0.2  
                x2, y2 = w * 0.9, h * 0.8
                keyboard_bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                self.scale_keyboard_to_detection(keypoints_2d, x1, y1, x2, y2)

        except Exception as e:
            rospy.logerr(f"Keyboard detection error: {e}")

        return keypoints_2d, keyboard_bbox

    def scale_keyboard_to_detection(self, keypoints_2d, x1, y1, x2, y2):
        # MODIFIED: Adjusted for vertical keyboard layout
        ref_width, ref_height = 650, 200
        
        det_width = x2 - x1
        det_height = y2 - y1
        
        scale_x = det_width / ref_width
        scale_y = det_height / ref_height
        
        for key, (ref_x, ref_y) in self.keyboard_layout.items():
            scaled_x = x1 + (ref_x * scale_x)
            scaled_y = y1 + (ref_y * scale_y)
            keypoints_2d[key] = [int(scaled_x), int(scaled_y)]

    def calculate_3d_positions(self, keypoints_2d):
        keypoints_3d = {}
        
        if self.camera_info is None:
            return keypoints_3d

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4] 
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        for key, (x, y) in keypoints_2d.items():
            try:
                if 0 <= x < self.depth_image.shape[1] and 0 <= y < self.depth_image.shape[0]:
                    depth = self.depth_image[int(y), int(x)] / 1000.0
                    
                    if 0.1 < depth < 2.0:
                        X = (x - cx) * depth / fx
                        Y = (y - cy) * depth / fy
                        Z = depth
                        
                        keypoints_3d[key] = [X, Y, Z]
                        
            except Exception as e:
                rospy.logwarn_throttle(10, f"3D position error for {key}: {e}")

        return keypoints_3d

    # MODIFIED: New function to estimate keyboard normal from detected keys
    def estimate_keyboard_normal(self, keypoints_3d_base):
        """Estimate the normal vector of the vertical keyboard surface"""
        try:
            if len(keypoints_3d_base) >= 3:
                # Get positions of first 3 keys to estimate plane
                positions = []
                for key_data in list(keypoints_3d_base.values())[:3]:
                    positions.append(np.array(key_data['position']))
                
                # Calculate plane normal using cross product
                v1 = positions[1] - positions[0]
                v2 = positions[2] - positions[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                
                self.keyboard_normal = normal
                rospy.loginfo(f"Estimated keyboard normal: {normal}")
            
        except Exception as e:
            rospy.logwarn(f"Could not estimate keyboard normal: {e}")
            # Default to facing negative X (typical vertical setup)
            self.keyboard_normal = np.array([-1, 0, 0])

    def transform_to_base_frame(self, point_camera):
        try:
            camera_point = PointStamped()
            camera_point.header.frame_id = self.camera_frame
            camera_point.header.stamp = rospy.Time(0)
            camera_point.point.x = point_camera[0]
            camera_point.point.y = point_camera[1]  
            camera_point.point.z = point_camera[2]
            
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            
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
            
            status = f"Vertical Typing Ready - 3D Keys: {len(self.keypoints_3d)}"
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
        self.publish_status(f"Starting to type on vertical keyboard: '{text}'")

        try:
            if not self.keypoints_3d:
                self.publish_status("❌ No vertical keyboard detected")
                return

            success_count = 0
            total_chars = len(text)
            
            for i, char in enumerate(text):
                if rospy.is_shutdown():
                    break
                    
                self.publish_status(f"Typing {i+1}/{total_chars}: '{char}' on vertical keyboard")
                
                key_name = self.char_to_key(char)
                
                if key_name and key_name in self.keypoints_3d:
                    if self.type_key_vertical(key_name):
                        success_count += 1
                        rospy.loginfo(f"✅ Typed '{char}' on vertical keyboard")
                    else:
                        rospy.logwarn(f"❌ Failed to type '{char}' on vertical keyboard")
                else:
                    rospy.logwarn(f"⚠️ Key '{key_name}' not available for '{char}'")
                
                rospy.sleep(0.5)

            self.publish_status(f"✅ Vertical typing complete: {success_count}/{total_chars}")

        except Exception as e:
            rospy.logerr(f"Vertical typing execution error: {e}")
            self.publish_status(f"❌ Vertical typing error: {str(e)}")
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

    # MODIFIED: New vertical keyboard typing function
    def type_key_vertical(self, key_name):
        """Type a key on vertical keyboard using pen in gripper"""
        if key_name not in self.keypoints_3d:
            return False

        try:
            target_pos = np.array(self.keypoints_3d[key_name]['position'])
            rospy.loginfo(f"⌨️ Typing key '{key_name}' on vertical keyboard at {target_pos}")

            # Try cartesian path first (most precise)
            if self.type_key_vertical_cartesian(key_name, target_pos):
                return True
            
            # Fallback to pose-based approach
            if self.type_key_vertical_pose_based(key_name, target_pos):
                return True

            rospy.logwarn(f"All vertical typing methods failed for key '{key_name}'")
            return False

        except Exception as e:
            rospy.logerr(f"Vertical key typing error for '{key_name}': {e}")
            return False

    def type_key_vertical_cartesian(self, key_name, target_pos):
        """Use cartesian path for vertical keyboard typing with pen"""
        try:
            rospy.loginfo(f"🎯 Vertical Cartesian approach for '{key_name}'")
            
            current_pose = self.move_group.get_current_pose().pose
            
            # MODIFIED: Calculate poses for vertical keyboard
            waypoints = []
            
            # 1. Approach pose - pen tip positioned away from key
            approach_pose = self.calculate_vertical_typing_pose(
                target_pos, -self.approach_distance, current_pose.orientation
            )
            waypoints.append(approach_pose)
            
            # 2. Contact pose - pen tip touches key surface
            contact_pose = self.calculate_vertical_typing_pose(
                target_pos, 0.0, current_pose.orientation
            )
            waypoints.append(contact_pose)
            
            # 3. Press pose - pen tip presses into key
            press_pose = self.calculate_vertical_typing_pose(
                target_pos, self.press_depth, current_pose.orientation
            )
            waypoints.append(press_pose)
            
            # 4. Release pose - pen tip pulls back to contact
            waypoints.append(contact_pose)
            
            # 5. Retract pose - pen tip moves away from keyboard
            waypoints.append(approach_pose)
            
            rospy.loginfo("Computing vertical Cartesian path...")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step=0.005,  # Smaller steps for precision
                jump_threshold=0.0  # No joint jumps allowed
            )
            
            rospy.loginfo(f"Vertical Cartesian path: {fraction*100:.1f}% complete")
            
            if fraction > 0.9:  # Higher threshold for vertical precision
                rospy.loginfo("Executing vertical Cartesian trajectory...")
                success = self.move_group.execute(plan, wait=True)
                
                if success:
                    rospy.loginfo(f"✅ Vertical Cartesian typing successful for '{key_name}'")
                    rospy.sleep(self.dwell_time)
                    self.move_group.stop()
                    return True
                else:
                    rospy.logwarn("Vertical Cartesian execution failed")
            else:
                rospy.logwarn(f"Vertical Cartesian path only {fraction*100:.1f}% valid")
                
        except Exception as e:
            rospy.logerr(f"Vertical Cartesian typing error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def calculate_vertical_typing_pose(self, key_position, offset_distance, base_orientation):
        """Calculate pose for vertical keyboard typing with pen tip positioning"""
        
        # Apply pen tip offset to reach the actual key surface
        pen_tip_position = key_position.copy()
        
        # MODIFIED: For vertical keyboard, offset is typically in X direction (towards/away from keyboard)
        # Adjust this based on your keyboard orientation
        pen_tip_position[0] += offset_distance * self.keyboard_normal[0]
        pen_tip_position[1] += offset_distance * self.keyboard_normal[1] 
        pen_tip_position[2] += offset_distance * self.keyboard_normal[2]
        
        # Apply pen offset from end effector to pen tip
        pen_tip_position[0] += self.pen_tip_offset[0]
        pen_tip_position[1] += self.pen_tip_offset[1]
        pen_tip_position[2] += self.pen_tip_offset[2]
        
        # MODIFIED: Calculate orientation to point pen perpendicular to vertical keyboard
        pose = Pose()
        pose.position.x = pen_tip_position[0]
        pose.position.y = pen_tip_position[1]
        pose.position.z = pen_tip_position[2]
        
        # Create orientation perpendicular to keyboard surface
        # For vertical keyboard, pen should point horizontally towards keyboard
        target_direction = -self.keyboard_normal  # Point towards keyboard
        
        # Calculate quaternion for desired orientation
        # This assumes your end effector's Z-axis should align with target direction
        z_axis = target_direction / np.linalg.norm(target_direction)
        
        # Create a coordinate frame
        if abs(z_axis[2]) < 0.9:  # Not pointing straight up/down
            x_axis = np.array([0, 0, 1])  # Use world Z as reference
        else:
            x_axis = np.array([1, 0, 0])  # Use world X as reference
            
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to quaternion
        quaternion = tf_trans.quaternion_from_matrix(
            np.vstack([
                np.column_stack([rotation_matrix, [0, 0, 0]]),
                [0, 0, 0, 1]
            ])
        )
        
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        
        return pose

    def type_key_vertical_pose_based(self, key_name, target_pos):
        """Fallback method using pose-based planning for vertical keyboard"""
        try:
            rospy.loginfo(f"🔧 Vertical pose-based approach for '{key_name}'")
            
            current_pose = self.move_group.get_current_pose().pose
            
            # Calculate target pose for key press
            target_pose = self.calculate_vertical_typing_pose(
                target_pos, 0.0, current_pose.orientation
            )
            
            self.move_group.set_pose_target(target_pose)
            self.move_group.set_planning_time(10.0)
            
            plan = self.move_group.plan()
            
            if plan[0]:
                success = self.move_group.execute(plan[1], wait=True)
                if success:
                    rospy.loginfo(f"✅ Vertical pose-based typing for '{key_name}'")
                    rospy.sleep(self.dwell_time)
                    
                    # Move back to safe distance
                    retract_pose = self.calculate_vertical_typing_pose(
                        target_pos, -self.safe_distance, current_pose.orientation
                    )
                    self.move_group.set_pose_target(retract_pose)
                    retract_plan = self.move_group.plan()
                    if retract_plan[0]:
                        self.move_group.execute(retract_plan[1], wait=True)
                    
                    self.move_group.stop()
                    self.move_group.clear_pose_targets()
                    return True
            
        except Exception as e:
            rospy.logerr(f"Vertical pose-based typing error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def move_to_home_position(self):
        """Move robot to home/ready position for vertical keyboard"""
        try:
            rospy.loginfo("🏠 Moving to vertical keyboard ready position...")
            
            # Try named target first
            try:
                self.move_group.set_named_target("home")
                return self.move_group.go(wait=True)
            except:
                # Fallback: move to a safe position in front of vertical keyboard
                safe_pose = self.move_group.get_current_pose().pose
                safe_pose.position.x -= 0.2  # Move back from keyboard
                safe_pose.position.z += 0.1  # Move up
                
                self.move_group.set_pose_target(safe_pose)
                plan = self.move_group.plan()
                if plan[0]:
                    return self.move_group.execute(plan[1], wait=True)
                    
        except Exception as e:
            rospy.logerr(f"Home movement error: {e}")
            return False

    def get_robot_status(self):
        try:
            status = {
                'timestamp': rospy.Time.now().to_sec(),
                'keyboard_detected': self.keyboard_detected,
                'num_keys_detected': len(self.keypoints_3d),
                'typing_active': self.typing_active,
                'keyboard_orientation': 'vertical',
                'keyboard_normal': self.keyboard_normal.tolist(),
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
        rospy.loginfo("🔄 Shutting down Vertical Keyboard Typing Controller...")
        
        try:
            self.typing_active = False
            
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            self.move_to_home_position()
            
            moveit_commander.roscpp_shutdown()
            
            rospy.loginfo("✅ Shutdown complete")
            
        except Exception as e:
            rospy.logerr(f"Shutdown error: {e}")

    def calibrate_pen_offset(self):
        """Helper function to calibrate pen tip offset from end effector"""
        rospy.loginfo("🖊️ Starting pen tip calibration...")
        rospy.loginfo("Move the robot so the pen tip touches a known point, then call this service")
        
        try:
            current_pose = self.move_group.get_current_pose().pose
            
            # You can manually adjust these values based on your pen holder setup
            rospy.loginfo(f"Current end effector position: [{current_pose.position.x:.3f}, {current_pose.position.y:.3f}, {current_pose.position.z:.3f}]")
            rospy.loginfo(f"Current pen offset: {self.pen_tip_offset}")
            rospy.loginfo("Adjust pen_tip_offset in code based on the difference between end effector and actual pen tip position")
            
        except Exception as e:
            rospy.logerr(f"Calibration error: {e}")

    def validate_vertical_setup(self):
        """Validate that the setup is correct for vertical keyboard typing"""
        try:
            rospy.loginfo("🔍 Validating vertical keyboard setup...")
            
            # Check if robot can reach typical vertical keyboard positions
            if not self.keypoints_3d:
                rospy.logwarn("No keys detected - ensure keyboard is visible to camera")
                return False
            
            # Check keyboard normal vector
            if np.allclose(self.keyboard_normal, [0, 0, -1], atol=0.5):
                rospy.logwarn("Keyboard appears horizontal, not vertical!")
                rospy.loginfo("Expected vertical keyboard normal to be roughly [-1,0,0] or [1,0,0]")
                rospy.loginfo(f"Detected normal: {self.keyboard_normal}")
            
            # Test reachability
            reachable_keys = 0
            for key_name, key_data in self.keypoints_3d.items():
                target_pos = np.array(key_data['position'])
                test_pose = self.calculate_vertical_typing_pose(target_pos, 0.0, 
                    self.move_group.get_current_pose().pose.orientation)
                
                # Simple workspace check
                if (0.2 < test_pose.position.x < 1.0 and 
                    -0.5 < test_pose.position.y < 0.5 and
                    0.1 < test_pose.position.z < 1.0):
                    reachable_keys += 1
            
            rospy.loginfo(f"Reachability check: {reachable_keys}/{len(self.keypoints_3d)} keys appear reachable")
            
            if reachable_keys < len(self.keypoints_3d) * 0.7:
                rospy.logwarn("Less than 70% of keys appear reachable - check robot positioning")
                return False
            
            rospy.loginfo("✅ Vertical keyboard setup validation passed")
            return True
            
        except Exception as e:
            rospy.logerr(f"Setup validation error: {e}")
            return False


def main():
    try:
        controller = VerticalKeyboardTypingController()
        
        rospy.on_shutdown(controller.shutdown_handler)
        
        # Validate setup
        rospy.sleep(2.0)  # Wait for initialization
        controller.validate_vertical_setup()
        
        rospy.loginfo("🤖 Vertical Keyboard Typing Controller is ready!")
        rospy.loginfo("📝 Send text to '/type_text' topic to start typing")
        rospy.loginfo("🖊️ Ensure pen is properly mounted in gripper")
        rospy.loginfo("📐 Keyboard should be mounted vertically")
        
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("👋 User interrupted")
    except Exception as e:
        rospy.logfatal(f"❌ Fatal error: {e}")
    finally:
        rospy.loginfo("🔄 Exiting...")

if __name__ == "__main__":
    main()