"""
Example: write 3D multi-camera pose estimates to NWB using MultiCameraPoseEstimation.

This demonstrates a DANNCE-style workflow:
  - Multiple synchronized cameras record the same animal.
  - Camera intrinsic and extrinsic calibration parameters are stored in a CameraCalibration object.
  - Each camera has its own CameraView that links the device and its source video.
  - A 3D triangulation algorithm produces world-space (x, y, z) coordinates for
    each body-part landmark, stored as PoseEstimationSeries inside MultiCameraPoseEstimation.

Run with:
    python write_multicamera_pose_estimates.py
"""

import datetime

import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject
from pynwb.image import ImageSeries

from ndx_pose import (
    CameraCalibration,
    CameraView,
    MultiCameraPoseEstimation,
    PoseEstimationSeries,
    Skeleton,
    Skeletons,
)

# ---------------------------------------------------------------------------
# 1. Create the NWBFile
# ---------------------------------------------------------------------------
nwbfile = NWBFile(
    session_description="Multi-camera 3D pose estimation of a freely moving mouse.",
    identifier="multicamera_pose_example",
    session_start_time=datetime.datetime.now(datetime.timezone.utc),
)

subject = Subject(subject_id="mouse001", species="Mus musculus", sex="M")
nwbfile.subject = subject

# ---------------------------------------------------------------------------
# 2. Add camera devices (must be in NWBFile before linking)
# ---------------------------------------------------------------------------
camera1 = nwbfile.create_device(
    name="camera1",
    description="Basler acA1300 — lateral view, left side",
    manufacturer="Basler",
)
camera2 = nwbfile.create_device(
    name="camera2",
    description="Basler acA1300 — lateral view, right side",
    manufacturer="Basler",
)
camera3 = nwbfile.create_device(
    name="camera3",
    description="Basler acA1300 — top-down view",
    manufacturer="Basler",
)
devices = [camera1, camera2, camera3]
n_cameras = len(devices)

# ---------------------------------------------------------------------------
# 3. Store source videos in acquisition (one ImageSeries per camera)
#    CameraView will link to these rather than embedding them.
# ---------------------------------------------------------------------------
frame_rate = 30.0  # Hz
source_videos = []
for cam in devices:
    video = ImageSeries(
        name=cam.name,
        description=f"Source video from {cam.name}.",
        unit="NA",
        format="external",
        external_file=[f"path/to/{cam.name}.mp4"],
        dimension=[1280, 1024],
        starting_frame=[0],
        rate=frame_rate,
    )
    nwbfile.add_acquisition(video)
    source_videos.append(video)

# ---------------------------------------------------------------------------
# 4. Camera calibration
#    Row order of every dataset must match the order of `devices`.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)

intrinsic = np.tile(np.eye(3, dtype="float32"), (n_cameras, 1, 1))
intrinsic[:, 0, 0] = 800.0   # fx
intrinsic[:, 1, 1] = 800.0   # fy
intrinsic[:, 0, 2] = 640.0   # cx
intrinsic[:, 1, 2] = 512.0   # cy

calibration = CameraCalibration(
    intrinsic_matrix=intrinsic,
    rotation_matrix=np.tile(np.eye(3, dtype="float32"), (n_cameras, 1, 1)),
    translation_vector=rng.standard_normal((n_cameras, 3)).astype("float32"),
    distortion_coefficients=np.zeros((n_cameras, 5), dtype="float32"),
    devices=devices,
)

# ---------------------------------------------------------------------------
# 5. Build one CameraView per camera
# ---------------------------------------------------------------------------
camera_views = [
    CameraView(
        name=cam.name,
        device=cam,
        source_video=vid,
    )
    for cam, vid in zip(devices, source_videos)
]

# ---------------------------------------------------------------------------
# 6. Define the skeleton
# ---------------------------------------------------------------------------
skeleton = Skeleton(
    name="mouse001_skeleton",
    nodes=["nose", "left_ear", "right_ear", "spine_mid", "tail_base"],
    edges=np.array([[0, 3], [1, 3], [2, 3], [3, 4]], dtype="uint8"),
    subject=subject,
)
skeletons = Skeletons(skeletons=[skeleton])

# ---------------------------------------------------------------------------
# 7. Simulate 3D pose estimation output
#    DANNCE produces (n_frames, 3) arrays in millimetres (world space).
# ---------------------------------------------------------------------------
n_frames = 1000
timestamps = np.arange(n_frames) / frame_rate  # seconds

reference_frame = "(0, 0, 0) is the geometric centre of the DANNCE camera rig."
confidence_definition = "Maximum probability from the 3D volumetric heatmap."

pose_estimation_series = []
for node in skeleton.nodes:
    data = rng.standard_normal((n_frames, 3)).astype("float32") * 10  # mm
    confidence = rng.random(n_frames).astype("float32")
    pes = PoseEstimationSeries(
        name=node,
        description=f"3D position of {node} in world coordinates.",
        data=data,
        unit="millimeters",
        reference_frame=reference_frame,
        timestamps=timestamps if node == skeleton.nodes[0] else pose_estimation_series[0],
        confidence=confidence,
        confidence_definition=confidence_definition,
    )
    pose_estimation_series.append(pes)

# ---------------------------------------------------------------------------
# 8. Build MultiCameraPoseEstimation
# ---------------------------------------------------------------------------
mcpe = MultiCameraPoseEstimation(
    name="MultiCameraPoseEstimation",
    pose_estimation_series=pose_estimation_series,
    camera_views=camera_views,
    calibration=calibration,
    description="3D keypoint coordinates of mouse001 estimated by DANNCE.",
    scorer="DANNCE",
    source_software="DANNCE",
    source_software_version="2.0.0",
    skeleton=skeleton,
)

# ---------------------------------------------------------------------------
# 9. Add everything to a behaviour processing module
# ---------------------------------------------------------------------------
behavior_pm = nwbfile.create_processing_module(
    name="behavior",
    description="Processed behavioral data — 3D pose estimation.",
)
behavior_pm.add(skeletons)
behavior_pm.add(mcpe)

# ---------------------------------------------------------------------------
# 10. Write to disk and read back
# ---------------------------------------------------------------------------
path = "test_multicamera_pose.nwb"

with NWBHDF5IO(path, mode="w") as io:
    io.write(nwbfile)

print(f"Wrote {path}")

with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
    read_nwbfile = io.read()
    read_mcpe = read_nwbfile.processing["behavior"]["MultiCameraPoseEstimation"]

    print(read_mcpe)
    print(f"  source_software : {read_mcpe.source_software} {read_mcpe.source_software_version}")
    print(f"  body parts      : {list(read_mcpe.pose_estimation_series.keys())}")
    print(f"  camera views    : {list(read_mcpe.camera_views.keys())}")
    print(f"  skeleton nodes  : {list(read_mcpe.skeleton.nodes)}")

    cal = read_mcpe.calibration
    print(f"  calibration     : intrinsic_matrix shape = {cal.intrinsic_matrix.shape}")
    print(f"  calibration devices : {[d.name for d in cal.devices]}")

    cv1 = read_mcpe.camera_views["camera1"]
    print(f"  camera1 device  : {cv1.device.name}")
    print(f"  camera1 video   : {cv1.source_video.external_file[0]}")
