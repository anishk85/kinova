#!/usr/bin/env python3

import rospy
import logging
import os
import json
import numpy as np
import torch  # Add this import
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
        self.base_frame = 'base_link'
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
        """Setup YOLO key detection model"""
        try:
            rospy.loginfo("🔧 Setting up YOLO key detection...")
            
            # Use the new best.pt model for key detection
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(current_dir, '../src/best.pt')
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                rospy.logwarn(f"YOLO model not found at {self.model_path}")
                rospy.logwarn("Please ensure best.pt is in the models directory")
                self.model = None
                self.use_yolo = False
            else:
                # Load YOLO model
                rospy.loginfo(f"Loading YOLO key detection model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    rospy.loginfo("🚀 Using GPU acceleration")
                else:
                    rospy.loginfo("🔄 Using CPU inference")
                
                self.use_yolo = True
                
                # Set detection parameters
                self.detection_confidence = 0.5
                self.iou_threshold = 0.45
                
                # Print available classes
                rospy.loginfo(f"Model classes: {list(self.model.names.values())}")
                
                # Suppress ultralytics logging
                import logging
                logging.getLogger("ultralytics").setLevel(logging.ERROR)
                
                rospy.loginfo("✅ YOLO key detection model loaded successfully")
            
            # Initialize detection state
            self.last_detection_time = rospy.Time.now()
            self.detection_timeout = 5.0  # seconds
            
        except Exception as e:
            rospy.logerr(f"Vision setup error: {e}")
            self.model = None
            self.use_yolo = False

    def load_keyboard_layout(self):
        """Initialize key detection - no layout file needed"""
        try:
            rospy.loginfo("🔧 Initializing key detection system...")
            
            self.detected_keys = {}
            
            rospy.loginfo("✅ Key detection system ready")
                
        except Exception as e:
            rospy.logerr(f"Error initializing key detection: {e}")

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
        """Process key detection and transform to 3D coordinates"""
        if self.color_image is None or self.depth_image is None or self.camera_info is None:
            return

        try:
            # Run YOLO detection on individual keys
            keypoints_2d, keyboard_bbox = self.detect_keyboard_keys(self.color_image)
            
            # Always publish annotated image for debugging
            self.publish_annotated_image(keypoints_2d, keyboard_bbox)
            
            if not keypoints_2d:
                # Check detection timeout
                time_since_detection = (rospy.Time.now() - self.last_detection_time).to_sec()
                if time_since_detection > self.detection_timeout:
                    self.keyboard_detected = False
                return

            # Calculate 3D positions for detected keys
            keypoints_3d_camera = self.calculate_3d_positions(keypoints_2d)
            
            if not keypoints_3d_camera:
                rospy.logwarn_throttle(5, "No 3D positions calculated - check depth image")
                return

            # Transform to base frame and validate workspace
            keypoints_3d_base = {}
            for key, pos_3d_cam in keypoints_3d_camera.items():
                pos_3d_base = self.transform_to_base_frame(pos_3d_cam)
                if pos_3d_base is not None and self.validate_workspace(pos_3d_base):
                    keypoints_3d_base[key] = {
                        'position': pos_3d_base.tolist(),
                        'depth': pos_3d_cam[2],
                        'pixel_coords': keypoints_2d[key]
                    }
                    
                    # Print coordinates for debugging
                    rospy.loginfo_throttle(5, 
                        f"🎯 Key '{key}': "
                        f"Pixel({keypoints_2d[key][0]}, {keypoints_2d[key][1]}) -> "
                        f"Camera({pos_3d_cam[0]:.3f}, {pos_3d_cam[1]:.3f}, {pos_3d_cam[2]:.3f}) -> "
                        f"Base({pos_3d_base[0]:.3f}, {pos_3d_base[1]:.3f}, {pos_3d_base[2]:.3f})"
                    )

            if keypoints_3d_base:
                self.keypoints_3d = keypoints_3d_base
                self.keyboard_detected = True
                
                # Publish key coordinates with detection info
                detection_data = {
                    "timestamp": rospy.Time.now().to_sec(),
                    "detection_method": "YOLO_individual_keys",
                    "total_keys_detected": len(keypoints_3d_base),
                    "keys": keypoints_3d_base
                }
                
                self.key_coordinates_pub.publish(String(data=json.dumps(detection_data)))
                
                rospy.loginfo_throttle(10, f"✅ Detected {len(keypoints_3d_base)} keys with 3D coordinates")
                
                # Print summary of available keys
                available_keys = list(keypoints_3d_base.keys())
                rospy.loginfo_throttle(15, f"📋 Available keys for typing: {sorted(available_keys)}")

        except Exception as e:
            rospy.logerr(f"Key detection processing error: {e}")

    def validate_workspace(self, position):
        """Validate if position is within robot's safe workspace"""
        try:
            x, y, z = position
            
            # Define safe workspace bounds for keyboard typing
            workspace_bounds = {
                'x_min': 0.2, 'x_max': 0.8,   # 20cm to 80cm forward
                'y_min': -0.4, 'y_max': 0.4,  # ±40cm left/right  
                'z_min': 0.1, 'z_max': 0.5    # 10cm to 50cm height
            }
            
            in_bounds = (
                workspace_bounds['x_min'] <= x <= workspace_bounds['x_max'] and
                workspace_bounds['y_min'] <= y <= workspace_bounds['y_max'] and
                workspace_bounds['z_min'] <= z <= workspace_bounds['z_max']
            )
            
            if not in_bounds:
                rospy.logdebug_throttle(10, f"Position {position} outside workspace bounds")
                
            return in_bounds
            
        except Exception as e:
            rospy.logerr(f"Workspace validation error: {e}")
            return False

    def detect_keyboard_keys(self, image):
        """Detect individual keys using YOLO model"""
        keypoints_2d = {}
        keyboard_bbox = []
        
        try:
            if self.use_yolo and self.model is not None:
                rospy.logdebug("Running YOLO key detection...")
                
                # Run YOLO inference on individual keys
                results = self.model(
                    image, 
                    conf=self.detection_confidence,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                if len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        rospy.loginfo_throttle(3, f"YOLO detected {len(boxes)} keys")
                        
                        detected_keys_list = []
                        
                        # Process each detected key
                        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box
                            confidence = float(conf)
                            class_id = int(cls)
                            
                            # Get class name from model
                            if class_id < len(self.model.names):
                                class_name = self.model.names[class_id]
                            else:
                                continue
                            
                            # Calculate center point of detected key
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Map detected class to usable key name
                            key_name = self.normalize_key_name(class_name)
                            
                            if key_name and confidence > self.detection_confidence:
                                keypoints_2d[key_name] = [center_x, center_y]
                                detected_keys_list.append(f"{key_name}({confidence:.2f})")
                                
                                rospy.logdebug(f"Detected key '{key_name}' at ({center_x}, {center_y}) confidence: {confidence:.2f}")
                        
                        # Print detected keys periodically
                        if detected_keys_list:
                            rospy.loginfo_throttle(2, f"🔍 Detected keys: {detected_keys_list}")
                        
                        # Calculate overall keyboard region
                        if keypoints_2d:
                            all_x = [pos[0] for pos in keypoints_2d.values()]
                            all_y = [pos[1] for pos in keypoints_2d.values()]
                            
                            margin = 50
                            x1 = max(0, min(all_x) - margin)
                            y1 = max(0, min(all_y) - margin) 
                            x2 = min(image.shape[1], max(all_x) + margin)
                            y2 = min(image.shape[0], max(all_y) + margin)
                            
                            keyboard_bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            self.last_detection_time = rospy.Time.now()
            
            else:
                rospy.logwarn_throttle(10, "YOLO model not available")
                
        except Exception as e:
            rospy.logerr(f"YOLO key detection error: {e}")

        return keypoints_2d, keyboard_bbox
    def normalize_key_name(self, detected_class):
        """Normalize detected class name to standard key name"""
        
        # Convert to uppercase for consistency
        key_upper = detected_class.upper().strip()
        
        # Handle common key name variations
        key_mappings = {
            # Letters (should be direct)
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
            'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 
            'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
            'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
            
            # Numbers
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
            '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
            
            # Special keys - handle variations
            'SPACE': 'SPACE', 'SPACEBAR': 'SPACE', 'SPACE_BAR': 'SPACE',
            'ENTER': 'ENTER', 'RETURN': 'ENTER',
            'BACKSPACE': 'BACKSPACE', 'DELETE': 'BACKSPACE', 'DEL': 'BACKSPACE',
            'SHIFT': 'SHIFT', 'SHIFT_L': 'SHIFT', 'SHIFT_R': 'SHIFT',
            'CTRL': 'CTRL', 'CONTROL': 'CTRL',
            'ALT': 'ALT',
            'TAB': 'TAB',
            'ESC': 'ESC', 'ESCAPE': 'ESC'
        }
        
        # Direct mapping
        if key_upper in key_mappings:
            return key_mappings[key_upper]
        
        # Single character keys
        if len(key_upper) == 1 and (key_upper.isalpha() or key_upper.isdigit()):
            return key_upper
        
        # Log unknown keys for debugging
        rospy.logwarn_throttle(30, f"Unknown key detected: '{detected_class}' -> normalized to '{key_upper}'")
        return key_upper


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
            camera_point.header.stamp = rospy.Time(0)  # Use latest available
            camera_point.point.x = point_camera[0]
            camera_point.point.y = point_camera[1]  
            camera_point.point.z = point_camera[2]
            
            # Get transform with timeout
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, 
                rospy.Time(0), rospy.Duration(2.0)  # Increased timeout
            )
            
            # Transform point
            base_point = do_transform_point(camera_point, transform)
            
            result = np.array([base_point.point.x, base_point.point.y, base_point.point.z])
            
            # Log successful transform periodically
            rospy.logdebug_throttle(5, f"Transform successful: {self.camera_frame} -> {self.base_frame}")
            
            return result
            
        except Exception as e:
            rospy.logerr_throttle(5, f"Transform error ({self.camera_frame} -> {self.base_frame}): {e}")
            
            # Try to auto-detect correct frame on first error
            if not hasattr(self, '_frame_detection_attempted'):
                self._frame_detection_attempted = True
                new_frame = self.detect_camera_frame()
                if new_frame != self.camera_frame:
                    rospy.loginfo(f"Switching camera frame from {self.camera_frame} to {new_frame}")
                    self.camera_frame = new_frame
                    # Retry with new frame
                    return self.transform_to_base_frame(point_camera)
            
            return None


    def detect_camera_frame(self):
        """Automatically detect the correct camera frame"""
        possible_frames = [
            f'{self.robot_namespace}/camera_link',
            f'{self.robot_namespace}/camera_color_optical_frame',
            f'{self.robot_namespace}/camera_depth_optical_frame',
            'camera_link',
            'camera_color_optical_frame',
            'camera_depth_optical_frame',
            'camera_color_frame',
            'camera_depth_frame'
        ]
        
        for frame in possible_frames:
            try:
                # Test if transform exists
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, frame, rospy.Time(0), rospy.Duration(1.0)
                )
                rospy.loginfo(f"✅ Found working camera frame: {frame}")
                return frame
            except Exception:
                continue
        
        rospy.logwarn("❌ No working camera frame found")
        return self.camera_frame  # Return original as fallback

    def publish_annotated_image(self, keypoints_2d, keyboard_bbox):
        """Enhanced detection visualization with always-on mode"""
        # Always publish for debugging - remove the parameter check
        # if not rospy.get_param('~publish_debug_image', False):
        #     return  # Skip if debug images not requested
            
        if self.color_image is None:
            return
            
        try:
            # Enhanced annotation for debugging
            annotated = self.color_image.copy()
            
            # Draw keyboard bounding box if available
            if keyboard_bbox:
                pts = np.array(keyboard_bbox, np.int32)
                cv2.polylines(annotated, [pts], True, (255, 0, 0), 3)
                cv2.putText(annotated, "Keyboard Region", 
                           (int(pts[0][0]), int(pts[0][1] - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Draw detected keys with enhanced visualization
            for key, (x, y) in keypoints_2d.items():
                # Color coding: green if 3D available, red if only 2D
                color = (0, 255, 0) if key in self.keypoints_3d else (0, 0, 255)
                
                # Draw key center with larger circle
                cv2.circle(annotated, (int(x), int(y)), 10, color, -1)
                cv2.circle(annotated, (int(x), int(y)), 12, (255, 255, 255), 2)
                
                # Draw key label with background
                font_scale = 0.6
                thickness = 2
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Background rectangle for text
                cv2.rectangle(annotated, 
                             (int(x) - text_size[0]//2 - 3, int(y) - text_size[1] - 15),
                             (int(x) + text_size[0]//2 + 3, int(y) - 5),
                             (0, 0, 0), -1)
                
                # Text
                cv2.putText(annotated, key, 
                           (int(x) - text_size[0]//2, int(y) - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Enhanced status information
            status_lines = [
                f"YOLO Model: {'Active' if self.use_yolo else 'Inactive'}",
                f"2D Keys Detected: {len(keypoints_2d)}",
                f"3D Keys Available: {len(self.keypoints_3d)}",
                f"Keyboard Status: {'Ready' if self.keyboard_detected else 'Not Ready'}",
                f"Transform Frame: {self.camera_frame}"
            ]
            
            for i, line in enumerate(status_lines):
                y_pos = 30 + i * 25
                # Background for text
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated, (5, y_pos - 20), (text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(annotated, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Publish debug image
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            self.annotated_image_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Debug image error: {e}")


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

    


    



