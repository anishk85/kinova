#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np

class YOLOv5ROSNode:
    def __init__(self):
        rospy.init_node('yolov5_ros_node', anonymous=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.bridge = CvBridge()

        # Camera intrinsic parameters
        self.color_info = {
            'K': [1297.672904, 0.0, 620.914026, 0.0, 1298.631344, 238.280325, 0.0, 0.0, 1.0],
            'P': [1297.008057, 0.0, 620.336773, 0.0, 0.0, 1304.157593, 238.813876, 0.0, 0.0, 0.0, 1.0, 0.0]
        }
        
        self.depth_info = {
            'K': [360.01333, 0.0, 243.87228, 0.0, 360.013366699, 137.9218444, 0.0, 0.0, 1.0],
            'P': [360.01333, 0.0, 243.87228, 0.0, 0.0, 360.013366699, 137.9218444, 0.0, 0.0, 0.0, 1.0, 0.0]
        }

        # ROS subscribers/publishers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.image_pub = rospy.Publisher("/yolo/detected_image", Image, queue_size=10)
        self.depth_image = None

    def depth_callback(self, msg):
        try:
            if msg.encoding == "16UC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) * 0.001
            elif msg.encoding == "32FC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            rospy.logerr(f"Depth processing error: {e}")


    def compute_3d_coordinates(self, u, v, depth):
        K = np.array(self.depth_info['K']).reshape(3, 3)
        uv1 = np.array([u, v, 1.0])
        xyz = np.linalg.inv(K) @ uv1 * depth
        return xyz[0], xyz[1], xyz[2]


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(cv_image)
            rendered_image = cv_image.copy()

            if self.depth_image is not None:
                depth_vis = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = cv2.applyColorMap(np.uint8(depth_vis), cv2.COLORMAP_JET)

                for *xyxy, conf, cls in results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, xyxy)
                    center_rgb = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    u_rgb, v_rgb = center_rgb
                    u_depth = int(((u_rgb - self.color_info['K'][2]) * 
                                 (self.depth_info['K'][0] / self.color_info['K'][0])) + 
                                self.depth_info['K'][2])
                    v_depth = int(((v_rgb - self.color_info['K'][5]) * 
                                 (self.depth_info['K'][4] / self.color_info['K'][4])) + 
                                self.depth_info['K'][5])

                    if (0 <= u_depth < 480 and 0 <= v_depth < 270):
                        depth_value = self.depth_image[v_depth, u_depth]
                        x, y, z = self.compute_3d_coordinates(u_depth, v_depth, depth_value)
                        
                    else:
                        depth_value = -1
                        x, y, z = None, None, None

                    self.draw_detections(rendered_image, depth_vis, 
                                         (x1, y1, x2, y2), 
                                         (u_rgb, v_rgb), 
                                         (u_depth, v_depth), 
                                         depth_value, x, y, z)
            
            self.publish_and_display(rendered_image, depth_vis)
        
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")

    def draw_detections(self, rgb_img, depth_img, bbox, center_rgb, center_depth, depth, x, y, z):
        cv2.rectangle(rgb_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.circle(rgb_img, center_rgb, 5, (255, 0, 0), -1)
        cv2.putText(rgb_img, f"{depth:.2f}m", (bbox[2]-60, bbox[3]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if x is not None:
            cv2.putText(rgb_img, f"3D: ({x:.2f}, {y:.2f}, {z:.2f})m", (bbox[2]-150, bbox[3]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
    def publish_and_display(self, rgb_img, depth_img):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img, "bgr8"))
        cv2.imshow("YOLOv5 Detection", rgb_img)
        cv2.imshow("Depth View", depth_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    YOLOv5ROSNode()
    rospy.spin()
