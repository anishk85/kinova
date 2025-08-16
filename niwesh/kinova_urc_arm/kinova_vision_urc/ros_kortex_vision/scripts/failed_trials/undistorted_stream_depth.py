#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from ros_kortex_vision.scripts.failed_trials.radialTangentialDistortion import RadialTangentialDistortion, PinholeCamera

class ObjectDepthEstimator:
    def __init__(self):
        rospy.init_node('object_depth_estimator', anonymous=True)

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")  # Load YOLOv8 model

        # Hardcoded camera parameters (replace with actual values)
        self.image_width = 640
        self.image_height = 480
        self.distortion_model = "plumb_bob"
        self.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.K = [1297.672904, 0.0, 620.914026, 0.0, 1298.631344, 238.280325, 0.0, 0.0, 1.0]

        # Extract focal lengths and principal point from K matrix
        fx = self.K[0]
        fy = self.K[4]
        cx = self.K[2]
        cy = self.K[5]
        
        # Initialize camera objects
        self.camera = PinholeCamera(width=self.image_width, height=self.image_height, f1=fx, f2=fy, c1=cx, c2=cy, distortion=self.distortion_model)
        self.distortion = RadialTangentialDistortion(self.D[0], self.D[1], self.D[2], self.D[3])

        self.camera_matrix = np.array(self.K).reshape((3, 3))
        self.dist_coeffs = np.array(self.D)

        # Compute new camera matrix
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (self.image_width, self.image_height), 1, (self.image_width, self.image_height)
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (self.image_width, self.image_height), cv2.CV_32FC1
        )

        self.image_received = False
        self.camera_info_received = False  # Initially set to False
        self.depth_image = None
        self.depth_encoding = None

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        self.rate = rospy.Rate(30)  # 30 Hz

    def camera_info_callback(self, data):
        """
        Camera info callback function.
        """
        # Extract camera parameters
        fx = data.K[0]
        fy = data.K[4]
        cx = data.K[2]
        cy = data.K[5]

        # Update PinholeCamera object
        self.camera = PinholeCamera(width=data.width, height=data.height, f1=fx, f2=fy, c1=cx, c2=cy, distortion=self.distortion_model)

        # Update distortion parameters
        if len(data.D) >= 4:
            self.distortion = RadialTangentialDistortion(data.D[0], data.D[1], data.D[2], data.D[3])
        
        # Convert parameters for OpenCV
        self.camera_matrix = np.array(data.K).reshape((3, 3))
        self.dist_coeffs = np.array(data.D)

        # Compute new camera matrix
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (data.width, data.height), 1, (data.width, data.height)
        )

        # Precompute remap matrices
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (data.width, data.height), cv2.CV_32FC1
        )

        self.camera_info_received = True
        rospy.loginfo("Camera info received and processed.")

    def depth_callback(self, data):
        """
        Depth image callback function.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self.depth_encoding = data.encoding
            rospy.loginfo(f"Depth image encoding: {self.depth_encoding}")
        except Exception as e:
            rospy.logerr(f"Error converting depth image: {e}")

    def undistort_image(self, image):
        """
        Undistorts an image using precomputed remap matrices.
        """
        if self.map1 is None or self.map2 is None:
            rospy.logwarn("Remap matrices not initialized. Returning original image.")
            return image
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def image_callback(self, data):
        """
        Image callback function.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_received = True
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        if not self.camera_info_received:
            rospy.logwarn_throttle(10, "Waiting for camera info...")
            return

        undistorted_image = self.undistort_image(cv_image)

        results = self.model.predict(undistorted_image, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                if class_name == "cup" and confidence > 0.7:
                    cv2.rectangle(undistorted_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(undistorted_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    if self.depth_image is None:
                        rospy.logwarn_throttle(10, "Depth image not yet received.")
                        continue

                    height_depth, width_depth = self.depth_image.shape[:2]
                    center_x_depth, center_y_depth = int(center_x), int(center_y)

                    if 0 <= center_x_depth < width_depth and 0 <= center_y_depth < height_depth:
                        depth_value = self.depth_image[center_y_depth, center_x_depth]
                        depth_value = -1 if np.isnan(depth_value) or depth_value <= 0.001 else depth_value
                    else:
                        depth_value = -1  

                    rospy.loginfo(f"Depth of {class_name} at ({center_x}, {center_y}): {depth_value:.3f}" if depth_value > 0 else f"Depth of {class_name} at ({center_x}, {center_y}): Invalid")

        cv2.imshow("Object Detection with Depth", undistorted_image)
        cv2.waitKey(1)

    def run(self):
        """
        Main loop.
        """
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == '__main__':
    try:
        estimator = ObjectDepthEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
