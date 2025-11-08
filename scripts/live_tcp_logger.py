#!/usr/bin/env python3
"""
Live TCP Logger - Waits properly by spinning while checking for TF
Updates every 1 second
"""
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener


class LiveTCPLogger(Node):
    def __init__(self):
        super().__init__('live_tcp_logger', 
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True)
        
        # Parameters
        self.parent_frame = self._get_param_str('parent_frame', 'base')
        self.child_frame = self._get_param_str('child_frame', 'tool0')
        self.update_interval = self._get_param_float('update_interval', 1.0)
        self.tcp_offset_m = self._get_param_float('tcp_offset_m', 0.0)
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Counter
        self._update_count = 0
        
        # Wait for TF with proper spinning
        self.get_logger().info('⏳ Waiting for TF buffer to populate...')
        self.get_logger().info(f'   Looking for: {self.parent_frame} → {self.child_frame}')
        
        # Wait up to 10 seconds, spinning while waiting
        found = False
        for i in range(100):  # 10 seconds (100 x 0.1s)
            rclpy.spin_once(self, timeout_sec=0.1)  # CRITICAL: Spin to receive TF!
            
            try:
                self.tf_buffer.lookup_transform(
                    self.parent_frame,
                    self.child_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.1)
                )
                found = True
                self.get_logger().info(f'✅ Found transform after {(i+1)*0.1:.1f} seconds!')
                break
            except:
                if i % 10 == 0 and i > 0:  # Print every second
                    self.get_logger().info(f'   Waiting... {i/10:.0f}s')
        
        if not found:
            self.get_logger().error('❌ Timeout! Transform not found after 10 seconds.')
            self.get_logger().error('   Check: ros2 run tf2_ros tf2_echo base tool0')
            return
        
        # Create timer
        self.timer = self.create_timer(self.update_interval, self._update)
        
        # Success message
        print()
        print('='*90)
        print('✅ LIVE TCP LOGGER - ACTIVE')
        print('='*90)
        print(f'📍 Tracking: {self.parent_frame} → {self.child_frame}')
        print(f'⏱️  Updates: every {self.update_interval:.1f} second(s)')
        if self.tcp_offset_m != 0.0:
            print(f'📏 TCP offset: {self.tcp_offset_m*1000:.1f} mm')
        print('='*90)
        print('🤖 Move the robot arm in FREEDRIVE mode!')
        print('⏹️  Press Ctrl+C to stop')
        print('='*90)
        print()

    def _get_param_str(self, name, default):
        try:
            if self.has_parameter(name):
                val = self.get_parameter(name).value
                return str(val) if val is not None else default
            self.declare_parameter(name, default)
            return default
        except:
            return default

    def _get_param_float(self, name, default):
        try:
            if self.has_parameter(name):
                return float(self.get_parameter(name).value)
            self.declare_parameter(name, float(default))
            return float(default)
        except:
            return float(default)

    def _update(self):
        """Get and display TCP position"""
        try:
            # Get transform
            transform = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            
            # Extract position
            t = transform.transform.translation
            q = transform.transform.rotation
            
            x, y, z = t.x, t.y, t.z
            
            # Apply TCP offset if specified
            if self.tcp_offset_m != 0.0:
                roll, pitch, yaw = self._quat_to_rpy(q.x, q.y, q.z, q.w)
                zx, zy, zz = self._get_tool_z_axis(roll, pitch, yaw)
                x += self.tcp_offset_m * zx
                y += self.tcp_offset_m * zy
                z += self.tcp_offset_m * zz
            
            # Convert to mm
            x_mm, y_mm, z_mm = x * 1000.0, y * 1000.0, z * 1000.0
            
            # Get orientation
            roll, pitch, yaw = self._quat_to_rpy(q.x, q.y, q.z, q.w)
            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            
            # Increment counter
            self._update_count += 1
            
            # Print header every 20 lines
            if self._update_count % 20 == 1:
                print()
                print('='*115)
                print('  #  |     X (meters)  |     Y (meters)  |     Z (meters)  |     X (mm)    |     Y (mm)    |     Z (mm)   ')
                print('='*115)
            
            # Print position
            print(f'{self._update_count:4d} | {x:9.5f} m     | {y:9.5f} m     | {z:9.5f} m     | {x_mm:8.2f} mm  | {y_mm:8.2f} mm  | {z_mm:8.2f} mm')
            
            # Print orientation every 20 updates
            if self._update_count % 20 == 0:
                print(f'      Orientation: RX={roll:6.3f} rad ({roll_deg:7.2f}°), '
                      f'RY={pitch:6.3f} rad ({pitch_deg:7.2f}°), '
                      f'RZ={yaw:6.3f} rad ({yaw_deg:7.2f}°)')
                print()
            
        except Exception as e:
            self.get_logger().warn(f'⚠️  Transform lookup failed: {str(e)[:60]}')

    @staticmethod
    def _quat_to_rpy(x, y, z, w):
        """Convert quaternion to roll-pitch-yaw"""
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    @staticmethod
    def _get_tool_z_axis(roll, pitch, yaw):
        """Get tool Z-axis direction"""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        zx = sy * sp * cr + cy * sr
        zy = -cy * sp * cr + sy * sr
        zz = cp * cr
        
        return zx, zy, zz


def main():
    rclpy.init()
    
    try:
        node = LiveTCPLogger()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n✅ Live TCP Logger stopped by user')
    except Exception as e:
        print(f'\n❌ Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
