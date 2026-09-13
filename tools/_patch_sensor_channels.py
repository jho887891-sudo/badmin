from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/src/badminton_brain/types.py')
s = p.read_text()
if 'odom_twist' in s:
    print('already patched'); raise SystemExit
old = """    frame: str = COURT_FRAME
    base_pose: Any = None
    joint_pos: Any = None
    joint_vel: Any = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'RobotSensorState')
        self.base_pose = check_batched('RobotSensorState.base_pose', self.base_pose, 7)
        self.joint_pos = check_batched('RobotSensorState.joint_pos', self.joint_pos, 6)
        self.joint_vel = check_batched('RobotSensorState.joint_vel', self.joint_vel, 6)
        _check_batch_dim([self.base_pose, self.joint_pos, self.joint_vel], 'RobotSensorState')"""
new = """    frame: str = COURT_FRAME
    base_pose: Any = None
    joint_pos: Any = None
    joint_vel: Any = None
    timestamp: float = 0.0
    # Optional proprioception channels (coordinator ruling 2026-09-13): the estimation
    # layer needs odometry and a yaw-rate source, but they must stay optional so that
    # existing callers keep working unchanged.
    odom_twist: Any = None
    imu_yaw_rate: Any = None

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'RobotSensorState')
        self.base_pose = check_batched('RobotSensorState.base_pose', self.base_pose, 7)
        self.joint_pos = check_batched('RobotSensorState.joint_pos', self.joint_pos, 6)
        self.joint_vel = check_batched('RobotSensorState.joint_vel', self.joint_vel, 6)
        if self.odom_twist is not None:
            self.odom_twist = check_batched('RobotSensorState.odom_twist', self.odom_twist, 3)
        if self.imu_yaw_rate is not None:
            self.imu_yaw_rate = check_per_env_scalar('RobotSensorState.imu_yaw_rate', self.imu_yaw_rate)
        _check_batch_dim([self.base_pose, self.joint_pos, self.joint_vel,
                          self.odom_twist, self.imu_yaw_rate], 'RobotSensorState')"""
if old not in s:
    raise SystemExit('RobotSensorState block not found verbatim')
s = s.replace(old, new)
p.write_text(s)
import ast; ast.parse(s)
print('types.py extended with odom_twist / imu_yaw_rate')
