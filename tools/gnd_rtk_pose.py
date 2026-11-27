"""RTK/IMU trajectory reader for GND preview; coordinate conventions must be checked."""
from pathlib import Path
import glob
import numpy as np


def load_body_poses(clip, frame_times, icp_poses, max_time_gap=.05,
                    max_xy_error=1.0, max_yaw_error_deg=3.0):
    """Interpolate time-stamped IMU poses and verify them against LiDAR odometry.

    Vehicle.pose is interpreted as timestamp, xyz, xyzw quaternion (world from
    IMU). imu-to-body calibration is required. Agreement with ICP is a coarse
    consistency test and does not prove pixel reprojection accuracy.
    """
    from scipy.spatial.transform import Rotation, Slerp
    import yaml
    pose_file = Path(clip) / 'rtkimu_qj_7683' / 'Vehicle.pose'
    ext_files = glob.glob(str(Path(clip) / 'parameter_yaml' / '*' / 'extrinsic' / 'imu-1_to_body-*.yaml'))
    if not pose_file.is_file() or len(ext_files) != 1:
        raise ValueError('RTK pose or unique IMU-to-body calibration unavailable')
    data = np.loadtxt(str(pose_file), delimiter=',', ndmin=2)
    times = np.asarray(frame_times, dtype=np.float64)
    if data.shape[1] != 8 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError('invalid RTK pose fields')
    t = data[:, 0]
    if np.any(np.diff(t) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError('RTK or frame timestamps are not strictly increasing')
    if times[0] < t[0] or times[-1] > t[-1]:
        raise ValueError('RTK trajectory does not cover all camera frames')
    idx = np.searchsorted(t, times, side='right').clip(1, len(t)-1)
    if np.max(t[idx]-t[idx-1]) > max_time_gap:
        raise ValueError('RTK timestamp gap exceeds limit')
    q = data[:, 4:8]; qnorm = np.linalg.norm(q, axis=1)
    if np.any(np.abs(qnorm - 1) > .02):
        raise ValueError('invalid RTK quaternion norm')
    R = Slerp(t, Rotation.from_quat(q / qnorm[:, None]))(times).as_matrix()
    pos = np.column_stack([np.interp(times, t, data[:, k]) for k in (1, 2, 3)])
    with open(ext_files[0], encoding='utf-8') as f:
        t_body_imu = np.asarray(yaml.safe_load(f)['transformation'], dtype=np.float64)
    if t_body_imu.shape != (4, 4) or not np.isfinite(t_body_imu).all():
        raise ValueError('invalid IMU calibration')
    r = t_body_imu[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=.001) or not np.isclose(np.linalg.det(r), 1, atol=.001):
        raise ValueError('invalid IMU calibration rotation')
    poses = np.tile(np.eye(4), (len(times), 1, 1)); poses[:, :3, :3] = R; poses[:, :3, 3] = pos
    poses = poses @ np.linalg.inv(t_body_imu)
    poses = np.linalg.inv(poses[0]) @ poses
    icp = np.asarray(icp_poses, dtype=np.float64)
    if icp.shape != poses.shape or not np.isfinite(icp).all():
        raise ValueError('ICP trajectory does not match frame count')
    xy = np.linalg.norm(poses[:, :2, 3] - icp[:, :2, 3], axis=1)
    delta = np.transpose(icp[:, :3, :3], (0, 2, 1)) @ poses[:, :3, :3]
    angles = np.degrees(np.abs(np.arctan2(delta[:, 1, 0], delta[:, 0, 0])))
    if np.max(xy) > max_xy_error or np.max(angles) > max_yaw_error_deg:
        raise ValueError('RTK and ICP trajectories disagree')
    return poses, {'source': 'rtk_imu_to_body_verified_against_icp',
                   'max_xy_error_m': float(np.max(xy)),
                   'median_xy_error_m': float(np.median(xy)),
                   'max_yaw_error_deg': float(np.max(angles)),
                   'pixel_alignment_verified': False}
