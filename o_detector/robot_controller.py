#!/usr/bin/env python3
"""
UR10e Robot Controller - WITH DETAILED DEBUG LOGGING
"""

import math
import os
import time
import yaml

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from action_msgs.msg import GoalStatus

from ament_index_python.packages import get_package_share_directory
from tf2_ros import Buffer, TransformListener

MOVEIT_SUCCESS = 1


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.config = self._load_config()
        self.joint_state = None
        self._last_ik_solution = None
        self.executing = False

        # TF + joint states
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        # MoveIt IK
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for MoveIt...')
        if not self.ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().fatal('MoveIt IK service /compute_ik not available.')
            raise SystemExit(1)

        # Choose controller
        self.controller_name = self._choose_controller()
        self.traj_client = ActionClient(self, FollowJointTrajectory, self.controller_name)
        self.get_logger().info(f'Waiting for trajectory controller: {self.controller_name}')
        if not self.traj_client.wait_for_server(timeout_sec=20.0):
            self.get_logger().fatal(f'No action server at {self.controller_name}')
            raise SystemExit(2)
        
        self.get_logger().info('✅ Robot Controller ready!')
        self.get_logger().info('='*60)
        plate = self.config.get('plate', {})
        self.get_logger().info(f"Plate center: ({plate.get('center_x',0):.3f}, "
                               f"{plate.get('center_y',0):.3f}, {plate.get('center_z',0):.3f})")
        self.get_logger().info('='*60)

    def _load_config(self):
        pkg = get_package_share_directory('o_detector')
        cfg_path = os.path.join(pkg, 'config', 'robot_config.yaml')
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)

    def _joint_cb(self, msg: JointState):
        if self.joint_state is None:
            self.get_logger().info(f'✅ Joint states connected ({len(msg.name)} joints)')
        self.joint_state = msg

    # ==================== PUBLIC API ====================
    
    def pick_ball(self, x, y, z):
        """Execute pick sequence"""
        self.get_logger().info('='*60)
        self.get_logger().info(f'🎯 PICK SEQUENCE START')
        self.get_logger().info(f'   Target: ({x:.4f}, {y:.4f}, {z:.4f})')
        self.get_logger().info('='*60)
        
        # Check workspace
        if not self._in_workspace(x, y, z):
            self.get_logger().error('❌ Target outside workspace!')
            return False
        
        self.get_logger().info('✅ Target is in workspace')
        
        # Move to target
        self.get_logger().info('📍 Step 1: Moving to pick position...')
        if not self._move_to_pose(x, y, z):
            self.get_logger().error('❌ Failed to reach pick position')
            return False
        
        self.get_logger().info('✅ Reached pick position')
        
        # Close gripper (skipping for now to test motion)
        # self._gripper(False)
        self.get_logger().info('🤏 [Gripper close skipped for testing]')
        
        # Lift
        lift_z = z + float(self.config['motion']['lift_height'])
        self.get_logger().info(f'📍 Step 2: Lifting to Z={lift_z:.4f}...')
        if not self._move_to_pose(x, y, lift_z):
            self.get_logger().error('❌ Failed to lift')
            return False
        
        self.get_logger().info('='*60)
        self.get_logger().info('✅ PICK SEQUENCE COMPLETE!')
        self.get_logger().info('='*60)
        return True

    # ==================== INTERNAL ====================
    
    def _in_workspace(self, x, y, z):
        """Check if position is in workspace"""
        ws = self.config['workspace']
        
        x_ok = ws['x_min'] <= x <= ws['x_max']
        y_ok = ws['y_min'] <= y <= ws['y_max']
        z_ok = ws['z_min'] <= z <= ws['z_max']
        
        if not x_ok:
            self.get_logger().error(f'❌ X={x:.3f} outside [{ws["x_min"]:.3f}, {ws["x_max"]:.3f}]')
        if not y_ok:
            self.get_logger().error(f'❌ Y={y:.3f} outside [{ws["y_min"]:.3f}, {ws["y_max"]:.3f}]')
        if not z_ok:
            self.get_logger().error(f'❌ Z={z:.3f} outside [{ws["z_min"]:.3f}, {ws["z_max"]:.3f}]')
        
        return x_ok and y_ok and z_ok

    def _move_to_pose(self, x_tcp, y_tcp, z_tcp):
        """Move to TCP position with flexible orientation"""
        self.get_logger().info(f'  🎯 Planning move to TCP: ({x_tcp:.4f}, {y_tcp:.4f}, {z_tcp:.4f})')
        
        # Get base orientation
        base = self.config['motion']['tool_orientation']
        
        # Try multiple orientations
        attempts = [
            (base['roll'], base['pitch'], base['yaw']),
            (base['roll'], base['pitch'] - 0.15, base['yaw']),
            (base['roll'], base['pitch'] + 0.15, base['yaw']),
            (base['roll'] + 0.10, base['pitch'], base['yaw']),
            (base['roll'] - 0.10, base['pitch'], base['yaw']),
        ]
        
        for i, (r, p, y) in enumerate(attempts, 1):
            self.get_logger().info(f'  🔄 IK attempt {i}/{len(attempts)}...')
            
            # Convert TCP to tool0 (flange)
            x_fl, y_fl, z_fl = self._tcp_to_flange(x_tcp, y_tcp, z_tcp, r, p, y)
            self.get_logger().info(f'     Flange target: ({x_fl:.4f}, {y_fl:.4f}, {z_fl:.4f})')
            
            # Create pose
            pose = PoseStamped()
            pose.header.frame_id = self.config['moveit']['planning_frame']
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x_fl
            pose.pose.position.y = y_fl
            pose.pose.position.z = z_fl
            
            qx, qy, qz, qw = self._rpy_to_quat(r, p, y)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            
            # Compute IK
            joints, error_code = self._compute_ik(pose)
            
            if joints is None:
                self.get_logger().warn(f'     ❌ IK failed (error code: {error_code})')
                continue
            
            self.get_logger().info(f'     ✅ IK solved!')
            self.get_logger().info(f'     Using RPY: ({r:.3f}, {p:.3f}, {y:.3f})')
            
            # Execute
            if self._execute(joints):
                self.get_logger().info(f'  ✅ Move completed successfully')
                return True
            else:
                self.get_logger().error(f'     ❌ Execution failed')
                return False
        
        self.get_logger().error(f'❌ ALL {len(attempts)} IK ATTEMPTS FAILED!')
        return False

    def _compute_ik(self, pose: PoseStamped):
        """Compute IK"""
        self.get_logger().info('       Calling MoveIt IK service...')
        
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.config['moveit']['group_name']
        req.ik_request.ik_link_name = self.config['moveit']['end_effector']
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout = Duration(seconds=float(self.config['moveit']['ik_timeout'])).to_msg()

        if self.joint_state:
            st = RobotState()
            st.joint_state = self.joint_state
            req.ik_request.robot_state = st

        fut = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        
        if not fut.done():
            self.get_logger().error('       ❌ IK service timeout!')
            return None, None
        
        res = fut.result()
        if not res:
            self.get_logger().error('       ❌ No IK result received')
            return None, None
        
        if res.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().warn(f'       ❌ IK error code: {res.error_code.val}')
            return None, res.error_code.val

        names = list(res.solution.joint_state.name)
        pos = list(res.solution.joint_state.position)
        self._last_ik_solution = (names, pos)
        
        self.get_logger().info(f'       ✅ IK solution: {len(pos)} joint values')
        return pos, MOVEIT_SUCCESS

    def _execute(self, joint_positions):
        """Execute trajectory"""
        if not self.joint_state:
            self.get_logger().error('     ❌ No joint states available!')
            return False

        self.get_logger().info('     📤 Preparing trajectory...')
        
        # UR joint order
        ctrl_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # Reorder joints
        if self._last_ik_solution:
            ik_names, ik_pos = self._last_ik_solution
            pos_by_name = dict(zip(ik_names, ik_pos))
        else:
            pos_by_name = dict(zip(self.joint_state.name, joint_positions))

        try:
            ordered = [float(pos_by_name[n]) for n in ctrl_names]
        except Exception as e:
            self.get_logger().error(f'     ❌ Joint mapping failed: {e}')
            return False

        # Create trajectory
        pt = JointTrajectoryPoint()
        pt.positions = ordered
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(seconds=float(self.config['motion']['move_time'])).to_msg()

        traj = JointTrajectory()
        traj.joint_names = ctrl_names
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Send goal
        self.get_logger().info(f'     📤 Sending trajectory goal...')
        send_fut = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        
        if not send_fut.done():
            self.get_logger().error('     ❌ Goal send timeout!')
            return False
        
        gh = send_fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error('     ❌ Goal REJECTED by controller!')
            self.get_logger().error('     Check: 1) Robot in remote control mode')
            self.get_logger().error('             2) External control program running')
            return False

        self.get_logger().info('     ✅ Goal ACCEPTED, robot moving...')
        self.executing = True
        
        # Wait for completion
        res_fut = gh.get_result_async()
        wait_time = float(self.config['motion']['move_time']) + 5.0
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=wait_time)
        self.executing = False

        if not res_fut.done():
            self.get_logger().error('     ❌ Execution timeout!')
            return False
        
        res = res_fut.result()
        if not res:
            self.get_logger().error('     ❌ No execution result')
            return False

        success = (res.status == GoalStatus.STATUS_SUCCEEDED)
        if success:
            self.get_logger().info('     ✅ Execution SUCCESS')
        else:
            self.get_logger().error(f'     ❌ Execution FAILED (status: {res.status})')
        
        return success

    # ==================== HELPERS ====================
    
    def _choose_controller(self):
        """Auto-detect controller"""
        candidates = [
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            '/joint_trajectory_controller/follow_joint_trajectory',
        ]
        
        for name in candidates:
            client = ActionClient(self, FollowJointTrajectory, name)
            if client.wait_for_server(timeout_sec=1.0):
                self.get_logger().info(f'✅ Connected to: {name}')
                return name
        
        self.get_logger().warn(f'No controller responded, using: {candidates[0]}')
        return candidates[0]

    def _tcp_to_flange(self, x, y, z, r, p, yaw):
        """Convert TCP to tool0 using tcp_offset"""
        offset = float(self.config['tool']['tcp_offset_m'])
        if offset <= 0.0:
            return x, y, z
        zx, zy, zz = self._tool_z_axis(r, p, yaw)
        return x - offset * zx, y - offset * zy, z - offset * zz

    @staticmethod
    def _rpy_to_quat(r, p, y):
        cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
        cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
        cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    @staticmethod
    def _tool_z_axis(r, p, y):
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        zx = sy * sp * cr + cy * sr
        zy = -cy * sp * cr + sy * sr
        zz = cp * cr
        return zx, zy, zz

    def _gripper(self, open_: bool):
        from std_msgs.msg import Float64
        pub = self.create_publisher(Float64, '/gripper_command', 10)
        msg = Float64()
        msg.data = self.config['gripper']['open_value' if open_ else 'close_value']
        pub.publish(msg)
        time.sleep(float(self.config['gripper']['wait_time']))


def main():
    rclpy.init()
    try:
        node = RobotController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
