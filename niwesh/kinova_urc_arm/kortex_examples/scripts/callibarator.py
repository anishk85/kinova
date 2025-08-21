#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from std_msgs.msg import String
import moveit_commander
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import tf.transformations as tf_trans

class PenOffsetCalibrator:
    
    def __init__(self):
        rospy.init_node("pen_offset_calibrator")
        
        self.robot_namespace = "my_gen3"
        self.move_group = moveit_commander.MoveGroupCommander(
            "arm",
            robot_description=f"{self.robot_namespace}/robot_description",
            ns=self.robot_namespace
        )
        
        # Current pen offset (start with initial guess)
        self.pen_tip_offset = [0.0, 0.0, -0.05]  # Initial guess: 5cm down from end effector
        
        # Calibration points storage
        self.calibration_points = []
        
        # Publishers for visualization
        self.status_pub = rospy.Publisher("/calibration_status", String, queue_size=10)
        
        rospy.loginfo("🖊️ Pen Offset Calibrator initialized")
        rospy.loginfo("Available commands:")
        rospy.loginfo("  1. python calibrator.py --method frame")
        rospy.loginfo("  2. python calibrator.py --method target")  
        rospy.loginfo("  3. python calibrator.py --method visual")

    def method_1_coordinate_frame_analysis(self):
        """
        Method 1: Analyze end effector coordinate frame to understand pen direction
        This helps you understand which axis the pen extends along
        """
        rospy.loginfo("🔍 Method 1: Coordinate Frame Analysis")
        
        try:
            current_pose = self.move_group.get_current_pose().pose
            
            rospy.loginfo("Current End Effector Pose:")
            rospy.loginfo(f"  Position: [{current_pose.position.x:.3f}, {current_pose.position.y:.3f}, {current_pose.position.z:.3f}]")
            rospy.loginfo(f"  Orientation (XYZW): [{current_pose.orientation.x:.3f}, {current_pose.orientation.y:.3f}, {current_pose.orientation.z:.3f}, {current_pose.orientation.w:.3f}]")
            
            # Convert quaternion to rotation matrix to understand frame orientation
            quaternion = [current_pose.orientation.x, current_pose.orientation.y, 
                         current_pose.orientation.z, current_pose.orientation.w]
            rotation_matrix = tf_trans.quaternion_matrix(quaternion)[:3, :3]
            
            rospy.loginfo("\nEnd Effector Frame Axes (in base frame):")
            rospy.loginfo(f"  X-axis (Red):   [{rotation_matrix[0,0]:.3f}, {rotation_matrix[1,0]:.3f}, {rotation_matrix[2,0]:.3f}]")
            rospy.loginfo(f"  Y-axis (Green): [{rotation_matrix[0,1]:.3f}, {rotation_matrix[1,1]:.3f}, {rotation_matrix[2,1]:.3f}]")
            rospy.loginfo(f"  Z-axis (Blue):  [{rotation_matrix[0,2]:.3f}, {rotation_matrix[1,2]:.3f}, {rotation_matrix[2,2]:.3f}]")
            
            rospy.loginfo("\n🔧 INTERPRETATION:")
            rospy.loginfo("Look at your pen holder and determine which axis the pen extends along:")
            rospy.loginfo("- If pen points along +X axis: pen_tip_offset = [+length, 0, 0]")
            rospy.loginfo("- If pen points along -X axis: pen_tip_offset = [-length, 0, 0]")
            rospy.loginfo("- If pen points along +Y axis: pen_tip_offset = [0, +length, 0]")
            rospy.loginfo("- If pen points along -Y axis: pen_tip_offset = [0, -length, 0]")
            rospy.loginfo("- If pen points along +Z axis: pen_tip_offset = [0, 0, +length]")
            rospy.loginfo("- If pen points along -Z axis: pen_tip_offset = [0, 0, -length]")
            rospy.loginfo("\nwhere 'length' is the distance from end effector center to pen tip")
            
        except Exception as e:
            rospy.logerr(f"Frame analysis error: {e}")

    def method_2_target_point_calibration(self):
        """
        Method 2: Use a known target point to calibrate pen offset
        Move pen tip to touch a known point, then calculate offset
        """
        rospy.loginfo("🎯 Method 2: Target Point Calibration")
        rospy.loginfo("Instructions:")
        rospy.loginfo("1. Place a distinctive target (tape mark, coin, etc.) on your workspace")
        rospy.loginfo("2. Manually move the robot so the pen tip EXACTLY touches this target")
        rospy.loginfo("3. Measure the target's position relative to robot base")
        rospy.loginfo("4. Run this calibration")
        
        input("Press Enter when pen tip is touching the target point...")
        
        try:
            # Get current end effector pose when pen tip is at target
            current_pose = self.move_group.get_current_pose().pose
            end_effector_pos = np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z
            ])
            
            rospy.loginfo(f"End effector position: {end_effector_pos}")
            
            # Get target position (you need to measure this)
            target_x = float(input("Enter target X coordinate (meters): "))
            target_y = float(input("Enter target Y coordinate (meters): "))
            target_z = float(input("Enter target Z coordinate (meters): "))
            target_pos = np.array([target_x, target_y, target_z])
            
            rospy.loginfo(f"Target position: {target_pos}")
            
            # Calculate offset in base frame
            offset_base_frame = target_pos - end_effector_pos
            rospy.loginfo(f"Offset in base frame: {offset_base_frame}")
            
            # Transform offset to end effector frame
            quaternion = [current_pose.orientation.x, current_pose.orientation.y,
                         current_pose.orientation.z, current_pose.orientation.w]
            rotation_matrix = tf_trans.quaternion_matrix(quaternion)[:3, :3]
            
            # Offset in end effector frame = R^T * offset_base_frame
            offset_ee_frame = rotation_matrix.T @ offset_base_frame
            
            rospy.loginfo("🎉 CALIBRATION RESULT:")
            rospy.loginfo(f"Pen tip offset (end effector frame): [{offset_ee_frame[0]:.4f}, {offset_ee_frame[1]:.4f}, {offset_ee_frame[2]:.4f}]")
            rospy.loginfo("Update your code with:")
            rospy.loginfo(f"self.pen_tip_offset = [{offset_ee_frame[0]:.4f}, {offset_ee_frame[1]:.4f}, {offset_ee_frame[2]:.4f}]")
            
            return offset_ee_frame
            
        except Exception as e:
            rospy.logerr(f"Target calibration error: {e}")
            return None

    def method_3_visual_validation(self, test_offset):
        """
        Method 3: Visual validation of pen offset
        Move to calculated positions and verify pen tip location
        """
        rospy.loginfo("👀 Method 3: Visual Validation")
        rospy.loginfo(f"Testing offset: {test_offset}")
        
        try:
            # Get current pose
            current_pose = self.move_group.get_current_pose().pose
            
            # Calculate where pen tip should be with this offset
            quaternion = [current_pose.orientation.x, current_pose.orientation.y,
                         current_pose.orientation.z, current_pose.orientation.w]
            rotation_matrix = tf_trans.quaternion_matrix(quaternion)[:3, :3]
            
            end_effector_pos = np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z
            ])
            
            # Transform offset to base frame
            offset_base_frame = rotation_matrix @ np.array(test_offset)
            predicted_pen_tip = end_effector_pos + offset_base_frame
            
            rospy.loginfo(f"End effector position: {end_effector_pos}")
            rospy.loginfo(f"Predicted pen tip position: {predicted_pen_tip}")
            
            # Create test poses to move pen tip to specific locations
            test_positions = [
                [predicted_pen_tip[0] + 0.05, predicted_pen_tip[1], predicted_pen_tip[2]],  # 5cm right
                [predicted_pen_tip[0] - 0.05, predicted_pen_tip[1], predicted_pen_tip[2]],  # 5cm left
                [predicted_pen_tip[0], predicted_pen_tip[1] + 0.05, predicted_pen_tip[2]],  # 5cm forward
                [predicted_pen_tip[0], predicted_pen_tip[1] - 0.05, predicted_pen_tip[2]],  # 5cm back
            ]
            
            for i, test_pos in enumerate(test_positions):
                rospy.loginfo(f"\nTest {i+1}: Moving pen tip to {test_pos}")
                
                # Calculate required end effector pose
                required_ee_pos = np.array(test_pos) - offset_base_frame
                
                test_pose = Pose()
                test_pose.position.x = required_ee_pos[0]
                test_pose.position.y = required_ee_pos[1]
                test_pose.position.z = required_ee_pos[2]
                test_pose.orientation = current_pose.orientation
                
                self.move_group.set_pose_target(test_pose)
                plan = self.move_group.plan()
                
                if plan[0]:
                    rospy.loginfo("Plan successful - executing movement...")
                    self.move_group.execute(plan[1], wait=True)
                    
                    input(f"Visually verify: Is pen tip at position {test_pos}? Press Enter to continue...")
                else:
                    rospy.logwarn(f"Could not plan to test position {i+1}")
                
                self.move_group.stop()
                self.move_group.clear_pose_targets()
            
            rospy.loginfo("✅ Visual validation complete!")
            
        except Exception as e:
            rospy.logerr(f"Visual validation error: {e}")

    def method_4_systematic_calibration(self):
        """
        Method 4: Systematic calibration using multiple known points
        Most accurate method for complex pen holders
        """
        rospy.loginfo("🔬 Method 4: Systematic Multi-Point Calibration")
        
        calibration_points = []
        
        for i in range(3):  # Collect 3 calibration points
            rospy.loginfo(f"\nCalibration Point {i+1}/3:")
            rospy.loginfo("1. Manually move robot so pen tip touches a known point")
            rospy.loginfo("2. Measure the coordinates of that point")
            
            input("Press Enter when pen tip is at calibration point...")
            
            try:
                # Get end effector pose
                current_pose = self.move_group.get_current_pose().pose
                ee_pos = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
                ee_quat = [current_pose.orientation.x, current_pose.orientation.y,
                          current_pose.orientation.z, current_pose.orientation.w]
                
                # Get target position
                target_x = float(input("Enter target X coordinate: "))
                target_y = float(input("Enter target Y coordinate: "))
                target_z = float(input("Enter target Z coordinate: "))
                target_pos = np.array([target_x, target_y, target_z])
                
                calibration_points.append({
                    'ee_pos': ee_pos,
                    'ee_quat': ee_quat,
                    'target_pos': target_pos
                })
                
                rospy.loginfo(f"Point {i+1} recorded")
                
            except Exception as e:
                rospy.logerr(f"Error recording point {i+1}: {e}")
                return None
        
        # Calculate average offset
        offsets_ee_frame = []
        
        for point in calibration_points:
            ee_pos = point['ee_pos']
            target_pos = point['target_pos']
            ee_quat = point['ee_quat']
            
            # Offset in base frame
            offset_base = target_pos - ee_pos
            
            # Transform to end effector frame
            rotation_matrix = tf_trans.quaternion_matrix(ee_quat)[:3, :3]
            offset_ee = rotation_matrix.T @ offset_base
            
            offsets_ee_frame.append(offset_ee)
        
        # Average the offsets
        avg_offset = np.mean(offsets_ee_frame, axis=0)
        std_offset = np.std(offsets_ee_frame, axis=0)
        
        rospy.loginfo("🎉 SYSTEMATIC CALIBRATION RESULT:")
        rospy.loginfo(f"Average offset: [{avg_offset[0]:.4f}, {avg_offset[1]:.4f}, {avg_offset[2]:.4f}]")
        rospy.loginfo(f"Standard deviation: [{std_offset[0]:.4f}, {std_offset[1]:.4f}, {std_offset[2]:.4f}]")
        rospy.loginfo("Update your code with:")
        rospy.loginfo(f"self.pen_tip_offset = [{avg_offset[0]:.4f}, {avg_offset[1]:.4f}, {avg_offset[2]:.4f}]")
        
        if np.max(std_offset) > 0.005:  # 5mm threshold
            rospy.logwarn("⚠️  High standard deviation detected - check measurement consistency")
        
        return avg_offset

    def validate_pen_offset_direction(self, pen_offset):
        """
        Validate that pen offset direction makes physical sense
        """
        rospy.loginfo("🔍 Validating pen offset direction...")
        
        offset_magnitude = np.linalg.norm(pen_offset)
        offset_direction = pen_offset / offset_magnitude
        
        rospy.loginfo(f"Pen offset magnitude: {offset_magnitude:.3f} m")
        rospy.loginfo(f"Pen offset direction (unit vector): {offset_direction}")
        
        # Common sense checks
        if offset_magnitude < 0.01:
            rospy.logwarn("⚠️  Offset magnitude very small (<1cm) - is pen holder very short?")
        elif offset_magnitude > 0.20:
            rospy.logwarn("⚠️  Offset magnitude very large (>20cm) - double check measurements")
        
        # Direction analysis
        dominant_axis = np.argmax(np.abs(offset_direction))
        axis_names = ['X', 'Y', 'Z']
        sign = "+" if offset_direction[dominant_axis] > 0 else "-"
        
        rospy.loginfo(f"Pen primarily extends along {sign}{axis_names[dominant_axis]} axis")
        
        # Provide interpretation
        interpretations = {
            0: "X-axis: Pen extends forward/backward from gripper",
            1: "Y-axis: Pen extends left/right from gripper", 
            2: "Z-axis: Pen extends up/down from gripper"
        }
        rospy.loginfo(f"Interpretation: {interpretations[dominant_axis]}")
        
        return True

def main():
    import sys
    
    calibrator = PenOffsetCalibrator()
    
    if len(sys.argv) > 1:
        method = sys.argv[1].replace('--method=', '')
        
        if method == "frame":
            calibrator.method_1_coordinate_frame_analysis()
            
        elif method == "target":
            offset = calibrator.method_2_target_point_calibration()
            if offset is not None:
                calibrator.validate_pen_offset_direction(offset)
                
        elif method == "visual":
            if len(sys.argv) > 4:
                test_offset = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
                calibrator.method_3_visual_validation(test_offset)
            else:
                rospy.logwarn("Usage: python calibrator.py --method=visual <x> <y> <z>")
                
        elif method == "systematic":
            offset = calibrator.method_4_systematic_calibration()
            if offset is not None:
                calibrator.validate_pen_offset_direction(offset)
                
        else:
            rospy.logwarn("Unknown method. Use: frame, target, visual, or systematic")
    
    else:
        rospy.loginfo("🖊️ Pen Offset Calibration Methods:")
        rospy.loginfo("Run with one of these options:")
        rospy.loginfo("  python calibrator.py --method=frame      # Analyze coordinate frames")
        rospy.loginfo("  python calibrator.py --method=target     # Single point calibration")
        rospy.loginfo("  python calibrator.py --method=visual     # Visual validation")
        rospy.loginfo("  python calibrator.py --method=systematic # Multi-point calibration")

if __name__ == "__main__":
    main()