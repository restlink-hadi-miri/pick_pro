#!/usr/bin/env python3
"""
Ball Detector with Proper Depth Integration and TF Transforms
Following Max's guidance for professional implementation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, TransformStamped, PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os
from ament_index_python.packages import get_package_share_directory
from tf2_ros import StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')
        
        # Load configuration
        self.config = self.load_config()
        
        # OpenCV bridge
        self.bridge = CvBridge()
        
        # Camera data
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.camera_ready = False
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Publish static camera transform
        self.publish_static_camera_transform()
        
        # Subscribe to camera topics
        self.create_subscription(
            Image,
            self.config['camera']['color_topic'],
            self.color_callback,
            10
        )
        self.create_subscription(
            Image,
            self.config['camera']['depth_topic'],
            self.depth_callback,
            10
        )
        self.create_subscription(
            CameraInfo,
            self.config['camera']['info_topic'],
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.ball_pub = self.create_publisher(Point, '/detected_ball_position', 10)
        self.color_pub = self.create_publisher(String, '/detected_ball_color', 10)
        
        # Detection timer (5 Hz)
        self.create_timer(0.2, self.detect_balls)
        
        self.get_logger().info('🎯 Ball Detector started with DEPTH and TF2!')
        self.get_logger().info(f'📷 Camera at: ({self.config["camera_calibration"]["translation_xyz_m"]["x"]:.3f}, '
                              f'{self.config["camera_calibration"]["translation_xyz_m"]["y"]:.3f}, '
                              f'{self.config["camera_calibration"]["translation_xyz_m"]["z"]:.3f})')

    def load_config(self):
        """Load configuration from YAML"""
        try:
            pkg_share = get_package_share_directory('o_detector')
            config_path = os.path.join(pkg_share, 'config', 'robot_config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.get_logger().info(f'✅ Config loaded from {config_path}')
                return config
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load config: {e}')
            return {}

    def publish_static_camera_transform(self):
        """
        Publish static transform from base_link to camera_color_optical_frame
        Uses EXACT calibration data from hand-eye calibration
        """
        cam_cal = self.config['camera_calibration']
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = cam_cal['parent_frame']
        t.child_frame_id = cam_cal['camera_frame']
        
        # Translation (from calibration)
        t.transform.translation.x = cam_cal['translation_xyz_m']['x']
        t.transform.translation.y = cam_cal['translation_xyz_m']['y']
        t.transform.translation.z = cam_cal['translation_xyz_m']['z']
        
        # Rotation (quaternion from calibration)
        t.transform.rotation.x = cam_cal['rotation_quaternion_xyzw']['x']
        t.transform.rotation.y = cam_cal['rotation_quaternion_xyzw']['y']
        t.transform.rotation.z = cam_cal['rotation_quaternion_xyzw']['z']
        t.transform.rotation.w = cam_cal['rotation_quaternion_xyzw']['w']
        
        # Broadcast static transform
        self.static_tf_broadcaster.sendTransform(t)
        
        self.get_logger().info('📡 Camera transform published from CALIBRATION data!')

    def camera_info_callback(self, msg):
        """Store camera intrinsics"""
        if not self.camera_ready:
            self.camera_info = msg
            self.camera_ready = True
            self.get_logger().info(
                f'📸 Camera ready: {msg.width}x{msg.height}, '
                f'fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}'
            )

    def color_callback(self, msg):
        """Store color image"""
        self.color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_callback(self, msg):
        """Store depth image (uint16 in millimeters)"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def detect_balls(self):
        """Main detection loop"""
        if self.color_image is None or self.depth_image is None:
            return
        
        if not self.camera_ready:
            return
        
        # Try each color
        for color_name in ['red', 'green', 'blue']:
            ball_pos = self.find_color(color_name)
            
            if ball_pos is not None:
                # Publish position in base_link frame
                point_msg = Point()
                point_msg.x = ball_pos[0]
                point_msg.y = ball_pos[1]
                point_msg.z = ball_pos[2]
                self.ball_pub.publish(point_msg)
                
                # Publish color
                color_msg = String()
                color_msg.data = color_name
                self.color_pub.publish(color_msg)
                
                self.get_logger().info(
                    f'🎯 {color_name.upper()} ball at '
                    f'({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f}) m'
                )
                
                return  # Detect one at a time

    def find_color(self, color_name):
        """
        Complete detection pipeline:
        1. Color detection in image
        2. Get 3D position in camera frame (using depth)
        3. Account for ball geometry
        4. Transform to base_link frame
        5. Validate safety
        """
        # Step 1: Color detection
        hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)
        
        color_cfg = self.config['ball']['colors'][color_name]
        
        # Create mask
        if color_name == 'red':
            mask1 = cv2.inRange(hsv, np.array(color_cfg['lower1']), 
                               np.array(color_cfg['upper1']))
            mask2 = cv2.inRange(hsv, np.array(color_cfg['lower2']), 
                               np.array(color_cfg['upper2']))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array(color_cfg['lower']), 
                              np.array(color_cfg['upper']))
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        min_area = self.config['camera']['min_ball_area']
        max_area = self.config['camera']['max_ball_area']
        
        best_contour = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area and area > best_area:
                best_area = area
                best_contour = contour
        
        if best_contour is None:
            return None
        
        # Get center pixel
        M = cv2.moments(best_contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Step 2 & 3: Get 3D position in camera frame (with ball geometry)
        pos_camera = self.pixel_to_3d_camera_frame(cx, cy, best_contour)
        
        if pos_camera is None:
            return None
        
        # Step 4: Transform to base_link frame
        pos_base_link = self.transform_to_base_link(pos_camera)
        
        if pos_base_link is None:
            return None
        
        # Step 5: Safety check
        if not self.is_safe_position(pos_base_link):
            self.get_logger().warn('⚠️  Position outside safe bounds')
            return None
        
        return pos_base_link

    def pixel_to_3d_camera_frame(self, cx, cy, contour):
        """
        Convert pixel + depth to 3D in camera optical frame
        
        RealSense optical frame:
        - +X: right
        - +Y: down
        - +Z: out of lens (forward)
        
        Accounts for ball geometry (detected center = top of sphere)
        """
        # Get depth at center pixel (mm)
        depth_mm = self.depth_image[cy, cx]
        depth_m = depth_mm / 1000.0
        
        # Validate depth
        if depth_m < self.config['camera']['min_depth'] or \
           depth_m > self.config['camera']['max_depth'] or \
           depth_m == 0:
            return None
        
        # Use median depth from center region (noise reduction)
        mask = np.zeros(self.depth_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        x, y, w, h = cv2.boundingRect(contour)
        center_region = mask[y + h//4 : y + 3*h//4, x + w//4 : x + 3*w//4]
        depth_region = self.depth_image[y + h//4 : y + 3*h//4, x + w//4 : x + 3*w//4]
        
        valid_depths = depth_region[center_region > 0]
        if len(valid_depths) > 5:
            depth_m = np.median(valid_depths) / 1000.0
        
        # Camera intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        ppx = self.camera_info.k[2]
        ppy = self.camera_info.k[5]
        
        # Convert pixel to 3D in camera optical frame
        # This gives us the TOP surface of the ball
        x_cam = (cx - ppx) * depth_m / fx
        y_cam = (cy - ppy) * depth_m / fy
        z_cam = depth_m
        
        # CRITICAL: Adjust for ball geometry
        # The detected point is the ball TOP (closest to camera)
        # Ball center is ball_radius deeper along viewing direction
        ball_radius = self.config['ball']['diameter'] / 2.0
        
        # Since +Z is viewing direction, ball center is at z + radius
        z_cam_center = z_cam + ball_radius
        
        # Recalculate X, Y for adjusted depth (keeps pixel projection consistent)
        x_cam_center = (cx - ppx) * z_cam_center / fx
        y_cam_center = (cy - ppy) * z_cam_center / fy
        
        return [x_cam_center, y_cam_center, z_cam_center]

    def transform_to_base_link(self, pos_camera):
        """
        Transform point from camera_color_optical_frame to base_link
        Uses TF2 with calibrated transform
        """
        try:
            # Create stamped point in camera frame
            point_camera = PointStamped()
            point_camera.header.frame_id = self.config['camera_calibration']['camera_frame']
            point_camera.header.stamp = self.get_clock().now().to_msg()
            point_camera.point.x = pos_camera[0]
            point_camera.point.y = pos_camera[1]
            point_camera.point.z = pos_camera[2]
            
            # Transform to base_link
            point_base = self.tf_buffer.transform(
                point_camera,
                self.config['camera_calibration']['parent_frame'],
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            return [
                point_base.point.x,
                point_base.point.y,
                point_base.point.z
            ]
            
        except Exception as e:
            self.get_logger().warn(f'❌ Transform failed: {e}')
            return None

    def is_safe_position(self, pos):
        """Validate position is within safe workspace"""
        plate = self.config['plate']
        margin = plate['safety_margin']
        
        x_min = plate['center_x'] - plate['width']/2 - margin
        x_max = plate['center_x'] + plate['width']/2 + margin
        y_min = plate['center_y'] - plate['height']/2 - margin
        y_max = plate['center_y'] + plate['height']/2 + margin
        
        return (x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max)


def main():
    rclpy.init()
    node = BallDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
