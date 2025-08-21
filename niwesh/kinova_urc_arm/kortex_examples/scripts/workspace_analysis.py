#!/usr/bin/env python3

import rospy
import numpy as np
import json
import os
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from sensor_msgs.msg import JointState
import moveit_commander
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import tf.transformations as tf_trans
from datetime import datetime

class WorkspaceMapper:
    
    def __init__(self):
        rospy.init_node("workspace_mapper", log_level=rospy.INFO)
        
        self.robot_namespace = "my_gen3"
        self.base_frame = f'{self.robot_namespace}/base_link'
        
        # Initialize MoveIt
        moveit_commander.roscpp_initialize([])
        self.move_group = moveit_commander.MoveGroupCommander(
            "arm",
            robot_description=f"{self.robot_namespace}/robot_description",
            ns=self.robot_namespace
        )
        
        # Workspace mapping data
        self.workspace_points = {}
        self.keyboard_corners = {}
        self.key_samples = {}
        self.safety_zones = {}
        
        # Pen tip offset (update this with your calibrated values)
        self.pen_tip_offset = [0.0, 0.0, -0.05]  # Update from calibration
        
        # Current joint state
        self.current_joint_state = None
        self.joint_sub = rospy.Subscriber(f"/{self.robot_namespace}/joint_states", 
                                         JointState, self.joint_state_callback)
        
        # Publishers
        self.status_pub = rospy.Publisher("/workspace_mapping_status", String, queue_size=10)
        self.workspace_pub = rospy.Publisher("/workspace_data", String, queue_size=10)
        
        # TF listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.loginfo("🗺️  Workspace Mapper initialized!")
        rospy.loginfo("Use your teleop to move the robot, then call mapping functions")
        
        self.print_instructions()
        
    def joint_state_callback(self, msg):
        self.current_joint_state = msg
    
    def publish_status(self, message):
        try:
            status_msg = String()
            status_msg.data = message
            self.status_pub.publish(status_msg)
            rospy.loginfo(f"📍 {message}")
        except Exception as e:
            rospy.logerr(f"Status publish error: {e}")
    
    def print_instructions(self):
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("🗺️  WORKSPACE MAPPING INSTRUCTIONS")
        rospy.loginfo("="*60)
        rospy.loginfo("1. Use your teleop to move the robot manually")
        rospy.loginfo("2. Position pen tip at important workspace points")
        rospy.loginfo("3. Call mapping functions via ROS service/topic")
        rospy.loginfo("\n📞 Available Commands (call via rosservice or code):")
        rospy.loginfo("   mapper.map_keyboard_corners()    # Map keyboard boundary")
        rospy.loginfo("   mapper.map_key_samples()         # Sample key positions")
        rospy.loginfo("   mapper.map_safety_zones()        # Define safe areas")
        rospy.loginfo("   mapper.save_workspace()          # Save all data")
        rospy.loginfo("   mapper.analyze_workspace()       # Analyze reachability")
        rospy.loginfo("="*60)
    
    def get_current_pen_tip_position(self):
        """Get current pen tip position in base frame"""
        try:
            current_pose = self.move_group.get_current_pose().pose
            
            # End effector position
            ee_pos = np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z
            ])
            
            # End effector orientation
            ee_quat = [current_pose.orientation.x, current_pose.orientation.y,
                      current_pose.orientation.z, current_pose.orientation.w]
            
            # Transform pen tip offset to base frame
            rotation_matrix = tf_trans.quaternion_matrix(ee_quat)[:3, :3]
            pen_offset_base = rotation_matrix @ np.array(self.pen_tip_offset)
            
            # Pen tip position in base frame
            pen_tip_pos = ee_pos + pen_offset_base
            
            return pen_tip_pos, current_pose
            
        except Exception as e:
            rospy.logerr(f"Error getting pen tip position: {e}")
            return None, None
    
    def map_keyboard_corners(self):
        """Map the four corners of the vertical keyboard"""
        rospy.loginfo("\n🔲 KEYBOARD CORNER MAPPING")
        rospy.loginfo("Position pen tip at each keyboard corner and press Enter")
        
        corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        
        for corner in corners:
            rospy.loginfo(f"\n📍 Position pen tip at {corner.upper()} corner of keyboard")
            input("Press Enter when pen tip is at the corner...")
            
            pen_pos, ee_pose = self.get_current_pen_tip_position()
            if pen_pos is not None:
                self.keyboard_corners[corner] = {
                    'pen_tip_position': pen_pos.tolist(),
                    'end_effector_pose': {
                        'position': [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z],
                        'orientation': [ee_pose.orientation.x, ee_pose.orientation.y, 
                                      ee_pose.orientation.z, ee_pose.orientation.w]
                    },
                    'joint_state': {
                        'names': list(self.current_joint_state.name),
                        'positions': list(self.current_joint_state.position)
                    } if self.current_joint_state else None,
                    'timestamp': rospy.Time.now().to_sec()
                }
                
                rospy.loginfo(f"✅ Recorded {corner}: {pen_pos}")
                self.publish_status(f"Mapped {corner} corner at {pen_pos}")
        
        # Calculate keyboard properties
        self.analyze_keyboard_geometry()
        
        rospy.loginfo("✅ Keyboard corner mapping complete!")
        return self.keyboard_corners
    
    def analyze_keyboard_geometry(self):
        """Analyze keyboard geometry from corner mappings"""
        if len(self.keyboard_corners) != 4:
            rospy.logwarn("Need all 4 corners for geometry analysis")
            return
            
        try:
            # Extract corner positions
            tl = np.array(self.keyboard_corners['top_left']['pen_tip_position'])
            tr = np.array(self.keyboard_corners['top_right']['pen_tip_position'])
            bl = np.array(self.keyboard_corners['bottom_left']['pen_tip_position'])
            br = np.array(self.keyboard_corners['bottom_right']['pen_tip_position'])
            
            # Calculate keyboard dimensions
            width = np.linalg.norm(tr - tl)
            height = np.linalg.norm(bl - tl)
            
            # Calculate keyboard center
            center = (tl + tr + bl + br) / 4
            
            # Calculate keyboard normal (assuming planar surface)
            v1 = tr - tl  # Top edge
            v2 = bl - tl  # Left edge  
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            # Keyboard orientation analysis
            keyboard_info = {
                'width': width,
                'height': height, 
                'center': center.tolist(),
                'normal': normal.tolist(),
                'area': width * height,
                'corners': {
                    'top_left': tl.tolist(),
                    'top_right': tr.tolist(),
                    'bottom_left': bl.tolist(),
                    'bottom_right': br.tolist()
                }
            }
            
            self.workspace_points['keyboard_geometry'] = keyboard_info
            
            rospy.loginfo("\n📐 KEYBOARD GEOMETRY ANALYSIS:")
            rospy.loginfo(f"   Width: {width:.3f} m")
            rospy.loginfo(f"   Height: {height:.3f} m") 
            rospy.loginfo(f"   Area: {width*height:.4f} m²")
            rospy.loginfo(f"   Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
            rospy.loginfo(f"   Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
            
            # Validate geometry
            if width < 0.1 or height < 0.05:
                rospy.logwarn("⚠️  Keyboard dimensions seem small - check corner mapping")
            if width > 0.8 or height > 0.4:
                rospy.logwarn("⚠️  Keyboard dimensions seem large - check corner mapping")
            
        except Exception as e:
            rospy.logerr(f"Geometry analysis error: {e}")
    
    def map_key_samples(self):
        """Map sample key positions across the keyboard"""
        rospy.loginfo("\n⌨️  KEY SAMPLE MAPPING")
        rospy.loginfo("Position pen tip at various keys to sample the workspace")
        
        # Common keys to sample for good coverage
        sample_keys = ['Q', 'P', 'A', 'L', 'Z', 'M', 'SPACE', '1', '0', 'ENTER']
        
        rospy.loginfo(f"Recommended keys to sample: {sample_keys}")
        rospy.loginfo("You can sample any keys you want. Type 'done' when finished.")
        
        while True:
            key_name = input("\nEnter key name (or 'done' to finish): ").strip().upper()
            
            if key_name == 'DONE':
                break
                
            if not key_name:
                continue
                
            rospy.loginfo(f"Position pen tip on the '{key_name}' key")
            input("Press Enter when pen tip is positioned...")
            
            pen_pos, ee_pose = self.get_current_pen_tip_position()
            if pen_pos is not None:
                self.key_samples[key_name] = {
                    'pen_tip_position': pen_pos.tolist(),
                    'end_effector_pose': {
                        'position': [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z],
                        'orientation': [ee_pose.orientation.x, ee_pose.orientation.y,
                                      ee_pose.orientation.z, ee_pose.orientation.w]
                    },
                    'joint_state': {
                        'names': list(self.current_joint_state.name),
                        'positions': list(self.current_joint_state.position)
                    } if self.current_joint_state else None,
                    'timestamp': rospy.Time.now().to_sec()
                }
                
                rospy.loginfo(f"✅ Recorded key '{key_name}': {pen_pos}")
                self.publish_status(f"Mapped key {key_name} at {pen_pos}")
        
        rospy.loginfo(f"✅ Key sampling complete! Mapped {len(self.key_samples)} keys")
        return self.key_samples
    
    def map_safety_zones(self):
        """Map safety zones and constraints"""
        rospy.loginfo("\n🛡️  SAFETY ZONE MAPPING")
        rospy.loginfo("Define safe zones and workspace limits")
        
        safety_points = [
            'home_position',
            'safe_retreat', 
            'workspace_min_bounds',
            'workspace_max_bounds',
            'collision_boundary'
        ]
        
        for point_name in safety_points:
            rospy.loginfo(f"\n📍 Position robot at: {point_name.upper()}")
            
            if point_name == 'home_position':
                rospy.loginfo("   -> Safe starting/ending position")
            elif point_name == 'safe_retreat':
                rospy.loginfo("   -> Emergency retreat position (away from keyboard)")  
            elif point_name == 'workspace_min_bounds':
                rospy.loginfo("   -> Minimum reachable position (closest corner)")
            elif point_name == 'workspace_max_bounds':
                rospy.loginfo("   -> Maximum reachable position (farthest corner)")
            elif point_name == 'collision_boundary':
                rospy.loginfo("   -> Closest safe approach to keyboard surface")
            
            should_map = input("Map this point? (y/n): ").strip().lower()
            if should_map != 'y':
                continue
                
            input("Press Enter when robot is positioned...")
            
            pen_pos, ee_pose = self.get_current_pen_tip_position()
            if pen_pos is not None:
                self.safety_zones[point_name] = {
                    'pen_tip_position': pen_pos.tolist(),
                    'end_effector_pose': {
                        'position': [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z],
                        'orientation': [ee_pose.orientation.x, ee_pose.orientation.y,
                                      ee_pose.orientation.z, ee_pose.orientation.w]
                    },
                    'joint_state': {
                        'names': list(self.current_joint_state.name),
                        'positions': list(self.current_joint_state.position)
                    } if self.current_joint_state else None,
                    'timestamp': rospy.Time.now().to_sec()
                }
                
                rospy.loginfo(f"✅ Recorded safety zone '{point_name}': {pen_pos}")
                self.publish_status(f"Mapped safety zone {point_name}")
        
        rospy.loginfo(f"✅ Safety zone mapping complete! Mapped {len(self.safety_zones)} zones")
        return self.safety_zones
    
    def analyze_workspace(self):
        """Analyze the mapped workspace for reachability and constraints"""
        rospy.loginfo("\n📊 WORKSPACE ANALYSIS")
        
        analysis = {
            'timestamp': rospy.Time.now().to_sec(),
            'total_points_mapped': len(self.keyboard_corners) + len(self.key_samples) + len(self.safety_zones),
            'keyboard_analysis': {},
            'reachability_analysis': {},
            'safety_analysis': {},
            'recommendations': []
        }
        
        try:
            # Keyboard coverage analysis
            if self.keyboard_corners and self.key_samples:
                self.analyze_keyboard_coverage(analysis)
            
            # Reachability analysis
            self.analyze_reachability(analysis)
            
            # Safety analysis  
            self.analyze_safety_constraints(analysis)
            
            # Generate recommendations
            self.generate_workspace_recommendations(analysis)
            
            # Store analysis
            self.workspace_points['analysis'] = analysis
            
            rospy.loginfo("\n📋 ANALYSIS SUMMARY:")
            rospy.loginfo(f"   Total mapped points: {analysis['total_points_mapped']}")
            
            if 'keyboard_coverage' in analysis['keyboard_analysis']:
                coverage = analysis['keyboard_analysis']['keyboard_coverage']
                rospy.loginfo(f"   Keyboard coverage: {coverage:.1f}%")
            
            if 'reachable_keys' in analysis['reachability_analysis']:
                reachable = analysis['reachability_analysis']['reachable_keys']
                total = analysis['reachability_analysis']['total_keys']
                rospy.loginfo(f"   Reachable keys: {reachable}/{total}")
                
            rospy.loginfo("\n💡 RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                rospy.loginfo(f"   • {rec}")
            
        except Exception as e:
            rospy.logerr(f"Workspace analysis error: {e}")
            
        return analysis
    
    def analyze_keyboard_coverage(self, analysis):
        """Analyze how well the key samples cover the keyboard area"""
        try:
            if 'keyboard_geometry' not in self.workspace_points:
                return
                
            keyboard_center = np.array(self.workspace_points['keyboard_geometry']['center'])
            keyboard_width = self.workspace_points['keyboard_geometry']['width']
            keyboard_height = self.workspace_points['keyboard_geometry']['height']
            
            # Check key distribution
            key_positions = []
            for key_data in self.key_samples.values():
                key_positions.append(np.array(key_data['pen_tip_position']))
            
            if not key_positions:
                return
                
            key_positions = np.array(key_positions)
            
            # Calculate coverage metrics
            key_center = np.mean(key_positions, axis=0)
            center_offset = np.linalg.norm(key_center - keyboard_center)
            
            # Coverage area (convex hull approximation)
            key_ranges = np.ptp(key_positions, axis=0)  # Range in each axis
            coverage_area = key_ranges[0] * key_ranges[1]  # Approximate coverage
            keyboard_area = keyboard_width * keyboard_height
            
            coverage_percent = min(100, (coverage_area / keyboard_area) * 100)
            
            analysis['keyboard_analysis'] = {
                'keyboard_coverage': coverage_percent,
                'center_alignment': center_offset,
                'sampled_keys': len(self.key_samples),
                'key_distribution': key_ranges.tolist()
            }
            
        except Exception as e:
            rospy.logerr(f"Coverage analysis error: {e}")
    
    def analyze_reachability(self, analysis):
        """Analyze reachability of different workspace areas"""
        try:
            total_points = len(self.key_samples) + len(self.keyboard_corners)
            reachable_points = 0
            
            # Check joint limits for recorded positions
            if self.current_joint_state:
                # Simplified reachability check based on recorded joint states
                for point_data in list(self.key_samples.values()) + list(self.keyboard_corners.values()):
                    if point_data.get('joint_state'):
                        joint_positions = point_data['joint_state']['positions']
                        # Simple check: if we recorded it, it was reachable
                        reachable_points += 1
            
            # Distance analysis
            distances = []
            if self.safety_zones.get('home_position'):
                home_pos = np.array(self.safety_zones['home_position']['pen_tip_position'])
                
                for key_data in self.key_samples.values():
                    key_pos = np.array(key_data['pen_tip_position'])
                    distance = np.linalg.norm(key_pos - home_pos)
                    distances.append(distance)
            
            analysis['reachability_analysis'] = {
                'reachable_keys': reachable_points,
                'total_keys': total_points,
                'reachability_percentage': (reachable_points / max(1, total_points)) * 100,
                'average_reach_distance': np.mean(distances) if distances else 0,
                'max_reach_distance': np.max(distances) if distances else 0,
                'min_reach_distance': np.min(distances) if distances else 0
            }
            
        except Exception as e:
            rospy.logerr(f"Reachability analysis error: {e}")
    
    def analyze_safety_constraints(self, analysis):
        """Analyze safety constraints and collision risks"""
        try:
            safety_metrics = {
                'safe_zones_mapped': len(self.safety_zones),
                'collision_clearance': 0,
                'workspace_bounds': {},
                'emergency_retreat_available': 'safe_retreat' in self.safety_zones
            }
            
            # Calculate minimum distances to keyboard surface
            if self.safety_zones.get('collision_boundary') and self.key_samples:
                boundary_pos = np.array(self.safety_zones['collision_boundary']['pen_tip_position'])
                
                min_distance = float('inf')
                for key_data in self.key_samples.values():
                    key_pos = np.array(key_data['pen_tip_position'])
                    distance = np.linalg.norm(key_pos - boundary_pos)
                    min_distance = min(min_distance, distance)
                
                safety_metrics['collision_clearance'] = min_distance
            
            analysis['safety_analysis'] = safety_metrics
            
        except Exception as e:
            rospy.logerr(f"Safety analysis error: {e}")
    
    def generate_workspace_recommendations(self, analysis):
        """Generate recommendations based on workspace analysis"""
        recommendations = []
        
        try:
            # Coverage recommendations
            if 'keyboard_analysis' in analysis and 'keyboard_coverage' in analysis['keyboard_analysis']:
                coverage = analysis['keyboard_analysis']['keyboard_coverage']
                if coverage < 70:
                    recommendations.append(f"Low keyboard coverage ({coverage:.1f}%) - sample more keys across keyboard area")
                elif coverage > 95:
                    recommendations.append("Excellent keyboard coverage!")
            
            # Reachability recommendations
            if 'reachability_analysis' in analysis:
                reachability = analysis['reachability_analysis']['reachability_percentage']
                if reachability < 90:
                    recommendations.append(f"Some areas may be unreachable ({reachability:.1f}%) - check robot positioning")
                
                max_distance = analysis['reachability_analysis']['max_reach_distance']
                if max_distance > 0.8:  # 80cm reach seems excessive
                    recommendations.append(f"Maximum reach distance is high ({max_distance:.2f}m) - consider robot repositioning")
            
            # Safety recommendations
            if 'safety_analysis' in analysis:
                if not analysis['safety_analysis']['emergency_retreat_available']:
                    recommendations.append("No emergency retreat position mapped - add safe retreat zone")
                
                if analysis['safety_analysis']['safe_zones_mapped'] < 3:
                    recommendations.append("Few safety zones mapped - consider mapping more safety positions")
            
            # General recommendations
            total_points = analysis['total_points_mapped']
            if total_points < 10:
                recommendations.append(f"Limited mapping data ({total_points} points) - collect more samples for better analysis")
            elif total_points > 50:
                recommendations.append("Comprehensive mapping data collected!")
            
            if not recommendations:
                recommendations.append("Workspace mapping looks good - ready for typing operations!")
            
            analysis['recommendations'] = recommendations
            
        except Exception as e:
            rospy.logerr(f"Recommendations generation error: {e}")
            analysis['recommendations'] = ["Error generating recommendations"]
    
    def save_workspace(self, filename=None):
        """Save all workspace mapping data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workspace_mapping_{timestamp}.json"
        
        try:
            # Compile all data
            workspace_data = {
                'metadata': {
                    'robot_namespace': self.robot_namespace,
                    'pen_tip_offset': self.pen_tip_offset,
                    'mapping_timestamp': rospy.Time.now().to_sec(),
                    'mapping_date': datetime.now().isoformat()
                },
                'keyboard_corners': self.keyboard_corners,
                'key_samples': self.key_samples,
                'safety_zones': self.safety_zones,
                'workspace_analysis': self.workspace_points.get('analysis', {}),
                'keyboard_geometry': self.workspace_points.get('keyboard_geometry', {})
            }
            
            # Save to file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(current_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(workspace_data, f, indent=2)
            
            rospy.loginfo(f"✅ Workspace data saved to: {filepath}")
            
            # Publish workspace data  
            self.workspace_pub.publish(String(data=json.dumps(workspace_data)))
            
            return filepath
            
        except Exception as e:
            rospy.logerr(f"Error saving workspace: {e}")
            return None
    
    def load_workspace(self, filename):
        """Load workspace mapping data from file"""
        try:
            with open(filename, 'r') as f:
                workspace_data = json.load(f)
            
            self.keyboard_corners = workspace_data.get('keyboard_corners', {})
            self.key_samples = workspace_data.get('key_samples', {})
            self.safety_zones = workspace_data.get('safety_zones', {})
            self.workspace_points['analysis'] = workspace_data.get('workspace_analysis', {})
            self.workspace_points['keyboard_geometry'] = workspace_data.get('keyboard_geometry', {})
            
            # Update pen offset if different
            if 'metadata' in workspace_data:
                loaded_offset = workspace_data['metadata'].get('pen_tip_offset')
                if loaded_offset and loaded_offset != self.pen_tip_offset:
                    rospy.logwarn(f"Loaded pen offset {loaded_offset} differs from current {self.pen_tip_offset}")
            
            rospy.loginfo(f"✅ Workspace data loaded from: {filename}")
            rospy.loginfo(f"   Corners: {len(self.keyboard_corners)}")
            rospy.loginfo(f"   Key samples: {len(self.key_samples)}")
            rospy.loginfo(f"   Safety zones: {len(self.safety_zones)}")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Error loading workspace: {e}")
            return False
    
    def print_mapping_summary(self):
        """Print summary of current mapping data"""
        rospy.loginfo("\n" + "="*50)
        rospy.loginfo("📋 WORKSPACE MAPPING SUMMARY")
        rospy.loginfo("="*50)
        rospy.loginfo(f"Keyboard corners: {len(self.keyboard_corners)}/4")
        for corner in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            status = "✅" if corner in self.keyboard_corners else "❌"
            rospy.loginfo(f"  {status} {corner}")
        
        rospy.loginfo(f"\nKey samples: {len(self.key_samples)}")
        if self.key_samples:
            rospy.loginfo(f"  Keys: {list(self.key_samples.keys())}")
        
        rospy.loginfo(f"\nSafety zones: {len(self.safety_zones)}")
        if self.safety_zones:
            rospy.loginfo(f"  Zones: {list(self.safety_zones.keys())}")
        
        if 'keyboard_geometry' in self.workspace_points:
            geom = self.workspace_points['keyboard_geometry']
            rospy.loginfo(f"\nKeyboard geometry:")
            rospy.loginfo(f"  Size: {geom.get('width', 0):.3f} × {geom.get('height', 0):.3f} m")
            rospy.loginfo(f"  Center: {geom.get('center', [0,0,0])}")
        
        rospy.loginfo("="*50)


def main():
    """Interactive workspace mapping session"""
    try:
        mapper = WorkspaceMapper()
        
        rospy.loginfo("🚀 Starting interactive workspace mapping...")
        
        while not rospy.is_shutdown():
            rospy.loginfo("\n📋 WORKSPACE MAPPING MENU:")
            rospy.loginfo("1. Map keyboard corners")
            rospy.loginfo("2. Map key samples") 
            rospy.loginfo("3. Map safety zones")
            rospy.loginfo("4. Analyze workspace")
            rospy.loginfo("5. Save workspace data")
            rospy.loginfo("6. Load workspace data")
            rospy.loginfo("7. Print mapping summary")
            rospy.loginfo("8. Exit")
            
            try:
                choice = input("\nEnter choice (1-8): ").strip()
                
                if choice == '1':
                    mapper.map_keyboard_corners()
                elif choice == '2':
                    mapper.map_key_samples()
                elif choice == '3':
                    mapper.map_safety_zones()
                elif choice == '4':
                    mapper.analyze_workspace()
                elif choice == '5':
                    filename = input("Enter filename (or press Enter for auto): ").strip()
                    mapper.save_workspace(filename if filename else None)
                elif choice == '6':
                    filename = input("Enter filename to load: ").strip()
                    if filename:
                        mapper.load_workspace(filename)
                elif choice == '7':
                    mapper.print_mapping_summary()
                elif choice == '8':
                    rospy.loginfo("👋 Exiting workspace mapper...")
                    break
                else:
                    rospy.logwarn("Invalid choice, please try again")
                    
            except KeyboardInterrupt:
                rospy.loginfo("👋 User interrupted")
                break
            except Exception as e:
                rospy.logerr(f"Menu error: {e}")
        
    except Exception as e:
        rospy.logfatal(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()