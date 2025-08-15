#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.optimize import minimize

class RadialTangentialDistortion:
    def __init__(self, k1, k2, p1, p2, k3=0, k4=0, k5=0, k6=0):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6

    def distort(self, u_undistorted):
        u_undistorted = u_undistorted.reshape(-1, 2)
        u, v = u_undistorted.T
        r2 = u**2 + v**2

        u_residual = 2 * self.p1 * u * v + self.p2 * (r2 + 2 * u**2)
        v_residual = 2 * self.p1 * (r2 + 2 * v**2) + 2 * self.p2 * u * v
        residual = np.stack([u_residual, v_residual], axis=1)

        radial_coeff = (
            (1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3)
            / (1 + self.k4 * r2 + self.k5 * r2**2 + self.k6 * r2**3)
        ).reshape(-1, 1)
        return radial_coeff * u_undistorted + residual

    def undistort(self, uDistorted):
        num_points = uDistorted.shape[
            0
        ]  # Axis 0 contains the number of points/images. Pass a single element in an array for a single image
        uUnDistorted = np.zeros_like(uDistorted)
        diff = lambda uUndistortedk, uDistorted_k: np.linalg.norm(
            self.distort(uUndistortedk) - uDistorted_k
        )
        # make it work for many points:
        for k in range(num_points):
            uDistortedk = uDistorted[k, :]
            res = minimize(
                diff, uDistortedk, args=(uDistortedk), method="Nelder-Mead", tol=1e-6
            )
            uUnDistorted[k, :] = res.x
        return uUnDistorted

class PinholeCamera:
    def __init__(self, width=0, height=0, f1=0, f2=0, c1=0, c2=0, distortion=None):
        self.width = width
        self.height = height
        self.f1 = f1
        self.f2 = f2
        self.c1 = c1
        self.c2 = c2
        self.distortion = distortion

class ObjectDepthEstimator:
    def __init__(self):
        rospy.init_node('object_depth_estimator', anonymous=True)

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")  # Load YOLOv8 model

        # Hardcoded camera parameters (replace with your actual values)
        self.image_width = 640  # Example, adjust based on your camera info
        self.image_height = 480 # Example, adjust based on your camera info
        self.distortion_model = "plumb_bob"
        self.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.K = [1297.672904, 0.0, 620.914026, 0.0, 1298.631344, 238.280325, 0.0, 0.0, 1.0]
        self.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.P = [1297.008057, 0.0, 620.336773, 0.0, 0.0, 1304.157593, 238.813876, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.binning_x = 0
        self.binning_y = 0
        self.roi_x_offset = 0
        self.roi_y_offset = 0
        self.roi_height = 0
        self.roi_width = 0
        self.roi_do_rectify = False

        # Initialize camera parameters (use hardcoded values)
        # Extract focal lengths and principal point from K matrix
        fx = self.K[0]
        fy = self.K[4]
        cx = self.K[2]
        cy = self.K[5]

        # Initialize RadialTangentialDistortion with your D values
        self.distortion = RadialTangentialDistortion(k1=self.D[0], k2=self.D[1], p1=self.D[2], p2=self.D[3], k3=self.D[4])

        self.camera = PinholeCamera(width=self.image_width, height=self.image_height, f1=fx, f2=fy, c1=cx, c2=cy, distortion=self.distortion)

        self.camera_matrix = np.array(self.K).reshape((3, 3))
        self.dist_coeffs = np.array(self.D)

        # Set default value, will be overwritten if camera_info is called.
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (self.image_width, self.image_height), 1, (self.image_width, self.image_height))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (self.image_width, self.image_height), cv2.CV_32FC1)

        self.image_received = False  # Flag to check if image is received
        self.camera_info_received = True  # Flag to check if camera info is received, set to true since we added default values
        self.depth_image = None # Depth image
        self.depth_encoding = None # Depth image encoding

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback) # Modified topic name

        # Rate
        self.rate = rospy.Rate(30)  # 30 Hz

    def camera_info_callback(self, data):
        """
        Camera info callback function.
        """
        self.camera.from_camera_info(data)
        self.distortion.from_camera_info(data)

        # Convert camera parameters to OpenCV format
        self.camera_matrix = np.array(data.K).reshape((3, 3))
        self.dist_coeffs = np.array(data.D)

        # Refine camera matrix (optional, but often improves results)
        image_width = data.width
        image_height = data.height
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (image_width, image_height), 1, (image_width, image_height))

        # Precompute remap matrices
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (image_width, image_height), cv2.CV_32FC1)

        self.camera_info_received = True
        rospy.loginfo("Camera info received and processed.")

    def depth_callback(self, data):
        """
        Depth image callback function.
        """
        try:
            # Convert the depth image using CvBridge
            self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self.depth_encoding = data.encoding
            rospy.loginfo(f"Depth image encoding: {self.depth_encoding}")
        except Exception as e:
            rospy.logerr(f"Error converting depth image: {e}")
            return

    def undistort_image(self, image):
        """
        Undistorts the given image using precomputed remap matrices.
        """
        if self.map1 is None or self.map2 is None:
            rospy.logwarn("Remap matrices not initialized. Returning original image.")
            return image

        undistorted_image = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
        return undistorted_image

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

        # Check if camera info has been received before proceeding
        if not self.camera_info_received:
            rospy.logwarn_throttle(10, "Waiting for camera info...") #only log once every 10 seconds
            return

        # Undistort the image
        undistorted_image = self.undistort_image(cv_image)

        # Object detection using YOLOv8
        results = self.model.predict(undistorted_image, verbose=False)

        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box information
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Get class name
                class_name = result.names[class_id]

                # Filter for "cup" detections (or any class you are interested in)
                if class_name == "cup" and confidence > 0.7:  # Adjust confidence threshold as needed
                    # Draw bounding box
                    cv2.rectangle(undistorted_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(undistorted_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Estimate depth (take depth from center of bounding box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Check if depth image is available
                    if self.depth_image is None:
                        rospy.logwarn_throttle(10, "Depth image not yet received.")
                        continue

                    # Determine depth image dimensions
                    height_depth, width_depth = self.depth_image.shape[:2]

                    # Project center point to depth image coordinates (assuming they are aligned)
                    center_x_depth = int(center_x)
                    center_y_depth = int(center_y)

                    # Ensure valid depth image coordinates
                    if 0 <= center_x_depth < width_depth and 0 <= center_y_depth < height_depth:
                        depth_value = self.depth_image[center_y_depth, center_x_depth]
                        if np.isnan(depth_value) or depth_value <= 0.001:  # Check for NaN or near-zero values
                            depth_value = -1 #consider depth value invalid
                        #rospy.loginfo(f"Type of depth value: {type(depth_value)}") #debugging code
                    else:
                        depth_value = -1  # Invalid depth

                    if depth_value > 0:
                        rospy.loginfo(f"Depth of {class_name} at ({center_x}, {center_y}): {depth_value:.3f}")
                    else:
                        rospy.loginfo(f"Depth of {class_name} at ({center_x}, {center_y}): Invalid")

        # Display the resulting image
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
