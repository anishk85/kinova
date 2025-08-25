#!/usr/bin/env python3

import rospy
import logging
import threading
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge
import tf2_ros

# Import our modular components
from vision_detection import VisionDetection
from depth_logic import DepthLogic  
from motion_planner import MotionPlanner


class RoboticTypingController:
    
    def __init__(self):
        rospy.init_node("robotic_typing_controller", log_level=rospy.INFO)
        logging.basicConfig(level=logging.INFO)

        rospy.loginfo("🤖 Initializing Robotic Typing Controller...")

        self.robot_namespace = "my_gen3"  
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_depth_frame'

        self.depth_topic = '/camera/depth/image_rect_raw'
        self.color_topic = '/camera/color/image_raw'
        self.camera_info_topic = '/camera/color/camera_info'

        self.bridge = CvBridge()
        
        # In-memory coordinate storage (no file saving)
        self.keypoints_3d = {}  # Current detected coordinates
        self.session_coordinates = {}  # Accumulated coordinates during session
        
        self.keyboard_detected = False
        self.depth_image = None
        self.color_image = None
        self.camera_info = None
        self.current_joint_state = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Wait for TF and auto-detect camera frame
        rospy.sleep(1.0)
        # self.camera_frame = self.detect_camera_frame()
        rospy.loginfo(f"Using camera frame: {self.camera_frame}")

        # Initialize modular components
        self.vision_detector = VisionDetection()
        self.depth_processor = DepthLogic(self.tf_buffer, self.base_frame, self.camera_frame)
        self.motion_planner = MotionPlanner(self.robot_namespace)
        
        self.load_keyboard_layout()
        self.setup_subscribers_publishers()

        rospy.loginfo("✅ Robotic Typing Controller initialized successfully!")

    def load_keyboard_layout(self):
        """Initialize key detection - no layout file needed"""
        try:
            rospy.loginfo("🔧 Initializing key detection system...")
            self.detected_keys = {}
            rospy.loginfo("✅ Key detection system ready")
        except Exception as e:
            rospy.logerr(f"Error initializing key detection: {e}")

    def setup_subscribers_publishers(self):
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.color_topic, Image, self.color_callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(f"/{self.robot_namespace}/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/type_text", String, self.type_text_callback)
        
        self.typing_status_pub = rospy.Publisher("/typing_status", String, queue_size=10)
        self.key_coordinates_pub = rospy.Publisher("/key_coordinates_3d", String, queue_size=10)
        self.annotated_image_pub = rospy.Publisher("/annotated_keyboard", Image, queue_size=10)

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
            keypoints_2d, keyboard_bbox = self.vision_detector.detect_keyboard_keys(self.color_image)
            
            # Always publish annotated image for debugging
            self.vision_detector.publish_annotated_image(
                self.color_image, keypoints_2d, keyboard_bbox, 
                self.keypoints_3d, self.session_coordinates, 
                self.camera_frame, self.vision_detector.use_yolo, 
                self.annotated_image_pub
            )
            
            if not keypoints_2d:
                # Check detection timeout
                time_since_detection = (rospy.Time.now() - self.vision_detector.last_detection_time).to_sec()
                if time_since_detection > self.vision_detector.detection_timeout:
                    self.keyboard_detected = False
                return

            # Calculate 3D positions for detected keys
            keypoints_3d_camera = self.depth_processor.calculate_3d_positions(
                keypoints_2d, self.camera_info, self.depth_image, self.color_image
            )

            if not keypoints_3d_camera:
                rospy.logwarn_throttle(5, "No 3D positions calculated - check depth image")
                return

            # Transform to base frame and validate workspace
            keypoints_3d_base = {}
            for key, pos_3d_cam in keypoints_3d_camera.items():
                pos_3d_base = self.depth_processor.transform_to_base_frame(pos_3d_cam)
                if pos_3d_base is not None and self.depth_processor.validate_workspace(pos_3d_base):
                    keypoints_3d_base[key] = {
                        'position': pos_3d_base.tolist(),
                        'depth': pos_3d_cam[2],
                        'pixel_coords': keypoints_2d[key],
                        'detection_time': rospy.Time.now().to_sec(),
                        'source': 'live_detection'
                    }
                    
                    # Store in session coordinates (accumulating during session)
                    self.session_coordinates[key] = keypoints_3d_base[key].copy()
                    
                    rospy.loginfo_throttle(5, 
                        f"🎯 Key '{key}': "
                        f"Pixel({keypoints_2d[key][0]}, {keypoints_2d[key][1]}) -> "
                        f"Camera({pos_3d_cam[0]:.3f}, {pos_3d_cam[1]:.3f}, {pos_3d_cam[2]:.3f}) -> "
                        f"Base({pos_3d_base[0]:.3f}, {pos_3d_base[1]:.3f}, {pos_3d_base[2]:.3f})"
                    )

            if keypoints_3d_base:
                # Store fresh detections
                self.keypoints_3d = keypoints_3d_base
                self.keyboard_detected = True
                
                rospy.loginfo_throttle(10, f"✅ Fresh: {len(keypoints_3d_base)} keys | Session total: {len(self.session_coordinates)} keys")
                
                # Print summary of available keys
                fresh_keys = list(keypoints_3d_base.keys())
                all_session_keys = list(self.session_coordinates.keys())
                rospy.loginfo_throttle(15, f"🆕 Fresh keys: {sorted(fresh_keys)}")
                rospy.loginfo_throttle(15, f"📋 Session keys: {sorted(all_session_keys)}")

        except Exception as e:
            rospy.logerr(f"Key detection processing error: {e}")

    def type_text_callback(self, msg):
        if self.motion_planner.typing_active:
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
        try:
            # Use all available coordinates (fresh + session)
            available_coords = self.session_coordinates.copy()
            available_coords.update(self.keypoints_3d)  # Fresh coords override session coords
            
            self.motion_planner.execute_typing(text, available_coords, self.publish_status)
            
        except Exception as e:
            rospy.logerr(f"Typing execution error: {e}")
            self.publish_status(f"❌ Typing error: {str(e)}")

    def get_robot_status(self):
        try:
            status = {
                'timestamp': rospy.Time.now().to_sec(),
                'keyboard_detected': self.keyboard_detected,
                'num_keys_detected': len(self.keypoints_3d),
                'num_session_keys': len(self.session_coordinates),
            }
            
            # Get motion planner status
            motion_status = self.motion_planner.get_robot_status(self.current_joint_state)
            status.update(motion_status)
                
            return status
            
        except Exception as e:
            rospy.logerr(f"Robot status error: {e}")
            return {}

    def emergency_stop(self):
        rospy.logwarn("🛑 Emergency stop!")
        self.motion_planner.emergency_stop(self.publish_status)

    def shutdown_handler(self):
        rospy.loginfo("🔄 Shutting down Robotic Typing Controller...")
        
        try:
            self.motion_planner.shutdown()
            
            rospy.loginfo(f"✅ Session ended with {len(self.session_coordinates)} keys learned")
            rospy.loginfo("✅ Shutdown complete")
            
        except Exception as e:
            rospy.logerr(f"Shutdown error: {e}")


def main():
    try:
        controller = RoboticTypingController()
        
        rospy.on_shutdown(controller.shutdown_handler)
        
        rospy.loginfo("🤖 Robotic Typing Controller is ready!")
        rospy.loginfo("📝 Send text to '/type_text' topic to start typing")
        rospy.loginfo("🎨 Watch '/annotated_keyboard' topic for visual feedback")
        rospy.loginfo("   - GREEN keys: Fresh detection with good depth")
        rospy.loginfo("   - YELLOW keys: Session coordinates available")
        rospy.loginfo("   - RED keys: Poor or no depth data")
        rospy.loginfo("💾 Coordinates are stored in memory during this session only")
        
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("👋 User interrupted")
    except Exception as e:
        rospy.logfatal(f"❌ Fatal error: {e}")
    finally:
        rospy.loginfo("🔄 Exiting...")


if __name__ == "__main__":
    main()