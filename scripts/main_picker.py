#!/usr/bin/env python3
"""
Main Picker - Interactive program with STRICT plate boundaries
Ensures robot only moves within the defined plate area
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String
import yaml
import os
from ament_index_python.packages import get_package_share_directory


class MainPicker(Node):
    def __init__(self):
        super().__init__('main_picker')
        
        # Load config
        self.config = self.load_config()
        
        # Latest detected ball
        self.detected_position = None
        self.detected_color = None
        
        # Subscribe to detections
        self.create_subscription(Point, '/detected_ball_position',
                                self.detection_callback, 10)
        self.create_subscription(String, '/detected_ball_color',
                                self.color_callback, 10)
        
        self.get_logger().info('Main Picker ready!')

    def load_config(self):
        """Load settings"""
        try:
            pkg_share = get_package_share_directory('o_detector')
            config_path = os.path.join(pkg_share, 'config', 'robot_config.yaml')
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                self.get_logger().info(f'✅ Config loaded from {config_path}')
                return cfg
        except Exception as e:
            self.get_logger().error(f'❌ Config load failed: {e}')
            return {}

    def detection_callback(self, msg):
        """Store detected ball position"""
        self.detected_position = [msg.x, msg.y, msg.z]

    def color_callback(self, msg):
        """Store detected color"""
        self.detected_color = msg.data

    def is_on_plate(self, x, y):
        """Check if position is within plate boundaries"""
        plate = self.config['plate']
        margin = plate.get('safety_margin', 0.05)
        
        x_min = plate['center_x'] - plate['width']/2 - margin
        x_max = plate['center_x'] + plate['width']/2 + margin
        y_min = plate['center_y'] - plate['height']/2 - margin
        y_max = plate['center_y'] + plate['height']/2 + margin
        
        on_plate = (x_min <= x <= x_max and y_min <= y <= y_max)
        
        if not on_plate:
            self.get_logger().warn(f'⚠️  Position ({x:.3f}, {y:.3f}) is OFF the plate!')
            self.get_logger().warn(f'    Plate bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]')
        
        return on_plate

    def automatic_mode(self):
        """Automatic mode - wait for detection and pick"""
        print("\n🤖 AUTOMATIC MODE")
        print("=" * 50)
        print("Place a colored ball on the plate...")
        print("Robot will automatically detect and pick it!")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        # Import and create robot controller
        from o_detector.robot_controller import RobotController
        robot = RobotController()
        
        while rclpy.ok():
            # Spin both nodes
            rclpy.spin_once(self, timeout_sec=0.1)
            rclpy.spin_once(robot, timeout_sec=0.1)
            
            # Check if ball detected
            if self.detected_position is not None:
                x, y, z = self.detected_position
                color = self.detected_color or "unknown"
                
                print(f"\n✓ Detected {color} ball at ({x:.3f}, {y:.3f}, {z:.3f})")
                
                # CRITICAL: Check if on plate
                if not self.is_on_plate(x, y):
                    print("❌ Ball is OFF the plate! Skipping...")
                    self.detected_position = None
                    self.detected_color = None
                    continue
                
                print("Starting pick sequence...")
                
                # Pick it!
                success = robot.pick_ball(x, y, z)
                
                if success:
                    print("✓ Pick successful!")
                else:
                    print("✗ Pick failed!")
                
                # Reset detection
                self.detected_position = None
                self.detected_color = None
                
                # Wait before next detection
                import time
                time.sleep(2.0)

    def manual_mode(self):
        """Manual mode - ask for coordinates"""
        print("\n🎮 MANUAL MODE")
        print("=" * 50)
        print("Enter coordinates to pick ball")
        print("=" * 50)
        
        # Show plate boundaries
        plate = self.config['plate']
        print(f"\n📏 Plate boundaries:")
        print(f"   X: {plate['center_x'] - plate['width']/2:.3f} to {plate['center_x'] + plate['width']/2:.3f}")
        print(f"   Y: {plate['center_y'] - plate['height']/2:.3f} to {plate['center_y'] + plate['height']/2:.3f}")
        print(f"   Z: ~{plate['center_z']:.3f} (plate surface)")
        
        # Import and create robot controller
        from o_detector.robot_controller import RobotController
        robot = RobotController()
        
        while rclpy.ok():
            print("\n📍 Enter pick coordinates:")
            print("Format: X Y Z (in meters)")
            print(f"Example: {plate['center_x']:.5f} {plate['center_y']:.5f} {plate['center_z']:.5f}")
            print("Or type 'quit' to exit")
            
            try:
                user_input = input("\nCoordinates: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting...")
                    break
                
                # Parse coordinates
                parts = user_input.split()
                if len(parts) != 3:
                    print("❌ Error: Need 3 numbers (X Y Z)")
                    continue
                
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                
                # Check if on plate
                if not self.is_on_plate(x, y):
                    print("❌ Coordinates are OFF the plate!")
                    print("   Do you want to proceed anyway? (yes/no)")
                    confirm = input().strip().lower()
                    if confirm != 'yes':
                        continue
                
                print(f"\n➜ Picking at ({x:.3f}, {y:.3f}, {z:.3f})...")
                
                # Pick it!
                success = robot.pick_ball(x, y, z)
                
                if success:
                    print("✓ Pick successful!")
                else:
                    print("✗ Pick failed!")
                
            except ValueError:
                print("❌ Error: Invalid numbers! Use format: 0.004 -0.948 -0.071")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    def run(self):
        """Main program loop"""
        print("\n" + "="*60)
        print("  🤖 UR10e BALL PICKER - Interactive Mode")
        print("="*60)
        
        # Show plate info
        plate = self.config['plate']
        print(f"\n📋 Plate Configuration:")
        print(f"   Center: ({plate['center_x']:.3f}, {plate['center_y']:.3f}, {plate['center_z']:.3f})")
        print(f"   Size: {plate['width']}m × {plate['height']}m")
        print(f"   Safety margin: {plate.get('safety_margin', 0.05)}m")
        
        print("\n" + "="*60)
        print("  Choose Mode:")
        print("="*60)
        print("  [A] Automatic - Camera detects balls automatically")
        print("  [M] Manual    - You enter coordinates")
        print("="*60)
        
        while True:
            choice = input("\nEnter choice (A or M): ").strip().upper()
            
            if choice == 'A':
                self.automatic_mode()
                break
            elif choice == 'M':
                self.manual_mode()
                break
            else:
                print("❌ Invalid choice! Please enter A or M")


def main():
    rclpy.init()
    
    try:
        node = MainPicker()
        node.run()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
