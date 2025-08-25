#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
import tf2_ros


# Camera parameters
color_info = {
    'K': [1297.672904, 0.0, 620.914026, 0.0, 1298.631344, 238.280325, 0.0, 0.0, 1.0]
}

depth_info = {
    'K': [360.01333, 0.0, 243.87228, 0.0, 360.013366699, 137.9218444, 0.0, 0.0, 1.0]
}


class DepthLogic:
    
    def __init__(self, tf_buffer, base_frame, camera_frame):
        self.tf_buffer = tf_buffer
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        
        # Workspace bounds
        self.workspace_bounds = {
            'x_min': 0.2, 'x_max': 0.8,   # 20cm to 80cm forward
            'y_min': -0.4, 'y_max': 0.4,  # ±40cm left/right  
            'z_min': 0.1, 'z_max': 0.5    # 10cm to 50cm height
        }

    def compute_3d_coordinates_from_uv(self, u, v, depth, camera_info):
        """Compute 3D coordinates from depth using actual camera info."""
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None

        # Use actual camera intrinsics instead of hardcoded values
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        
        # Convert pixel coordinates to 3D camera coordinates
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        
        return X, Y, Z

    def depth_helper(self, x, y, depth_image, camera_info):
        """Get 3D coordinates with proper depth sampling and validation"""
        try:
            # For aligned depth and color images, coordinates should match directly
            # If your depth and color are aligned, no projection needed
            u_depth = int(x)
            v_depth = int(y)
            
            # Validate bounds
            if not (0 <= u_depth < depth_image.shape[1] and 0 <= v_depth < depth_image.shape[0]):
                rospy.logdebug(f"Depth coordinates out of bounds: ({u_depth}, {v_depth})")
                return None

            # Enhanced multi-point depth sampling
            depth_samples = []
            sample_radius = 3  # Increased sample area
            
            for dx in range(-sample_radius, sample_radius + 1):
                for dy in range(-sample_radius, sample_radius + 1):
                    px, py = u_depth + dx, v_depth + dy
                    if (0 <= px < depth_image.shape[1] and 
                        0 <= py < depth_image.shape[0]):
                        
                        depth_val = depth_image[py, px]  # Keep in original units
                        
                        # Check if depth value is valid (adjust range as needed)
                        if depth_val > 0 and not np.isnan(depth_val) and not np.isinf(depth_val):
                            depth_m = depth_val / 1000.0  # Convert mm to meters
                            if 0.1 < depth_m < 3.0:  # Valid depth range in meters
                                depth_samples.append(depth_m)

            if len(depth_samples) < 3:  # Need at least 3 valid samples
                rospy.logdebug(f"Insufficient depth samples: {len(depth_samples)} for key at ({x}, {y})")
                return None

            # Use median depth for robustness
            depth_value = np.median(depth_samples)
            
            rospy.logdebug(f"Depth sampling: {len(depth_samples)} samples, median: {depth_value:.3f}m")
            
            # Compute 3D coordinates
            camera_coords = self.compute_3d_coordinates_from_uv(u_depth, v_depth, depth_value, camera_info)
            return camera_coords
            
        except Exception as e:
            rospy.logdebug(f"Depth helper error: {e}")
            return None

    def calculate_3d_positions(self, keypoints_2d, camera_info, depth_image, color_image):
        """Enhanced 3D position calculation with better validation"""
        keypoints_3d = {}
        
        if camera_info is None:
            rospy.logwarn("No camera info available for 3D calculation")
            return keypoints_3d
            
        if depth_image is None:
            rospy.logwarn("No depth image available for 3D calculation")
            return keypoints_3d

        rospy.loginfo(f"Computing 3D positions for {len(keypoints_2d)} keys")
        rospy.loginfo(f"Depth image shape: {depth_image.shape}")
        rospy.loginfo(f"Color image shape: {color_image.shape}")

        for key, (x, y) in keypoints_2d.items():
            try:
                # Validate input coordinates
                if not (0 <= x < color_image.shape[1] and 0 <= y < color_image.shape[0]):
                    rospy.logwarn(f"Key '{key}' coordinates ({x}, {y}) out of color image bounds")
                    continue
                    
                camera_coords = self.depth_helper(x, y, depth_image, camera_info)

                if camera_coords is not None:
                    keypoints_3d[key] = [camera_coords[0], camera_coords[1], camera_coords[2]]

                    # Enhanced logging
                    rospy.loginfo(f"✅ 3D calc for '{key}': "
                                f"2D({x}, {y}) -> "
                                f"3D({camera_coords[0]:.3f}, {camera_coords[1]:.3f}, {camera_coords[2]:.3f})")
                # else:
                    # rospy.logwarn(f"❌ No valid depth for key '{key}' at ({x}, {y})")
                    
                    # Debug: Check what's at that pixel
                    if (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
                        raw_depth = depth_image[y, x]
                        rospy.loginfo(f"   Raw depth at ({x}, {y}): {raw_depth}")
                                
            except Exception as e:
                rospy.logerr(f"3D position error for '{key}': {e}")

        rospy.loginfo(f"Successfully calculated 3D positions for {len(keypoints_3d)}/{len(keypoints_2d)} keys")
        return keypoints_3d

    def transform_to_base_frame(self, point_camera):
        try:
            camera_point = PointStamped()
            camera_point.header.frame_id = self.camera_frame
            camera_point.header.stamp = rospy.Time(0)
            camera_point.point.x = point_camera[0]
            camera_point.point.y = point_camera[1]
            camera_point.point.z = point_camera[2]

            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(2.0)
            )

            base_point = do_transform_point(camera_point, transform)
            result = np.array([base_point.point.x, base_point.point.y, base_point.point.z])

            return result

        except Exception as e:
            rospy.logerr_throttle(5, f"Transform error ({self.camera_frame} -> {self.base_frame}): {e}")
            return None

    def validate_workspace(self, position):
        """Validate if position is within robot's safe workspace"""
        try:
            x, y, z = position
            
            in_bounds = (
                self.workspace_bounds['x_min'] <= x <= self.workspace_bounds['x_max'] and
                self.workspace_bounds['y_min'] <= y <= self.workspace_bounds['y_max'] and
                self.workspace_bounds['z_min'] <= z <= self.workspace_bounds['z_max']
            )
            
            if not in_bounds:
                rospy.logdebug_throttle(10, f"Position {position} outside workspace bounds")
                
            return in_bounds
            
        except Exception as e:
            rospy.logerr(f"Workspace validation error: {e}")
            return False