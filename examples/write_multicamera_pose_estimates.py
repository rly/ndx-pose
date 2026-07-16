"""
Example: write 3D multi-camera pose estimates to NWB using MultiCameraPoseEstimation.

This demonstrates a (s)DANNCE-style workflow where only the triangulated 3D pose is
available, with no per-camera 2D estimates:
  - Multiple synchronized cameras record the same animal. Each camera is a CalibratedCamera
    (a Device extended with intrinsic/extrinsic calibration parameters), added once to the
    NWBFile so it can be shared by reference (e.g., across subjects in a multi-subject
    sDANNCE session) without duplicating the rig.
  - Each camera is still represented by its own PoseEstimation object, which links to that
    camera device and its source video. Since DANNCE/sDANNCE does not output per-camera 2D
    keypoints, these PoseEstimation objects hold no PoseEstimationSeries children (this is
    valid: PoseEstimationSeries is optional, zero-or-more, in PoseEstimation) - they exist to
    record device/video provenance for each camera view.
  - A 3D triangulation algorithm produces world-space (x, y, z) coordinates for each body-part
    landmark, stored as PoseEstimationSeries directly inside MultiCameraPoseEstimation.

For a workflow where 2D per-camera estimates ARE available (e.g., Anipose running
DeepLabCut per camera before triangulating), simply pass `pose_estimation_series` to each
per-camera PoseEstimation as well.

Run with:
    python write_multicamera_pose_estimates.py
"""

import datetime

import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject
from pynwb.image import ImageSeries

from ndx_pose import (
    CalibratedCamera,
    MultiCameraPoseEstimation,
    PoseEstimation,
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
# 2. Add calibrated camera devices (must be in NWBFile before linking).
#    Each CalibratedCamera carries its own intrinsic/extrinsic calibration, so there is no
#    row-order matching against a separate calibration object, and the same camera can be
#    linked from multiple PoseEstimation/MultiCameraPoseEstimation objects (e.g., one per
#    subject in a multi-subject session) without duplicating the rig.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)

camera_names = ["camera1", "camera2", "camera3"]
camera_descriptions = [
    "Basler acA1300 - lateral view, left side",
    "Basler acA1300 - lateral view, right side",
    "Basler acA1300 - top-down view",
]
cameras = []
for name, description in zip(camera_names, camera_descriptions):
    intrinsic_matrix = np.eye(3, dtype="float32")
    intrinsic_matrix[0, 0] = 800.0  # fx
    intrinsic_matrix[1, 1] = 800.0  # fy
    intrinsic_matrix[0, 2] = 640.0  # cx
    intrinsic_matrix[1, 2] = 512.0  # cy
    camera = CalibratedCamera(
        name=name,
        description=description,
        manufacturer="Basler",
        intrinsic_matrix=intrinsic_matrix,
        rotation_matrix=np.eye(3, dtype="float32"),
        translation_vector=rng.standard_normal(3).astype("float32"),
        distortion_coefficients=np.zeros(5, dtype="float32"),
    )
    nwbfile.add_device(camera)
    cameras.append(camera)

# ---------------------------------------------------------------------------
# 3. Store source videos in acquisition (one ImageSeries per camera).
#    Each per-camera PoseEstimation will link to one of these.
# ---------------------------------------------------------------------------
frame_rate = 30.0  # Hz
n_frames = 1000
source_videos = []
for camera in cameras:
    video = ImageSeries(
        name=f"{camera.name}_video",
        description=f"Source video from {camera.name}.",
        unit="NA",
        format="external",
        external_file=[f"path/to/{camera.name}.mp4"],
        dimension=[1280, 1024],
        starting_frame=[0],
        rate=frame_rate,
    )
    nwbfile.add_acquisition(video)
    source_videos.append(video)

# ---------------------------------------------------------------------------
# 4. Define the skeleton
# ---------------------------------------------------------------------------
skeleton = Skeleton(
    name="mouse001_skeleton",
    nodes=["nose", "left_ear", "right_ear", "spine_mid", "tail_base"],
    edges=np.array([[0, 3], [1, 3], [2, 3], [3, 4]], dtype="uint8"),
    subject=subject,
)
skeletons = Skeletons(skeletons=[skeleton])

# ---------------------------------------------------------------------------
# 5. Build one PoseEstimation per camera to record that camera's device/video provenance.
#    sDANNCE does not produce per-camera 2D keypoints, so these have no PoseEstimationSeries
#    children - only the device and source_video links are set. This is valid because
#    PoseEstimationSeries is optional (zero or more) in PoseEstimation.
# ---------------------------------------------------------------------------
timestamps = np.arange(n_frames) / frame_rate  # seconds

pose_estimations = [
    PoseEstimation(
        name=f"PoseEstimation_{camera.name}",
        description=f"Camera view {camera.name}; no 2D keypoints, only used for 3D triangulation.",
        source_software="sDANNCE",
        source_software_version="1.0.0",
        device=camera,
        source_video=video,
    )
    for camera, video in zip(cameras, source_videos)
]

# ---------------------------------------------------------------------------
# 6. Simulate 3D pose estimation output.
#    sDANNCE triangulates directly from the raw multi-camera video (no intermediate
#    per-camera 2D keypoints) into (n_frames, 3) arrays in millimetres (world space).
# ---------------------------------------------------------------------------
reference_frame = "(0, 0, 0) is the geometric centre of the sDANNCE camera rig."
confidence_definition = "Maximum probability from the 3D volumetric heatmap."

pose_estimation_series_3d = []
for node in skeleton.nodes:
    data = rng.standard_normal((n_frames, 3)).astype("float32") * 10  # mm
    confidence = rng.random(n_frames).astype("float32")
    pes = PoseEstimationSeries(
        name=node,
        description=f"3D position of {node} in world coordinates.",
        data=data,
        unit="millimeters",
        reference_frame=reference_frame,
        timestamps=timestamps if node == skeleton.nodes[0] else pose_estimation_series_3d[0],
        confidence=confidence,
        confidence_definition=confidence_definition,
    )
    pose_estimation_series_3d.append(pes)

# ---------------------------------------------------------------------------
# 7. Build MultiCameraPoseEstimation, combining the 3D estimates and the per-camera
#    PoseEstimation objects.
# ---------------------------------------------------------------------------
mcpe = MultiCameraPoseEstimation(
    name="MultiCameraPoseEstimation",
    pose_estimation_series=pose_estimation_series_3d,
    pose_estimations=pose_estimations,
    description="3D keypoint coordinates of mouse001 estimated by sDANNCE.",
    scorer="sDANNCE",
    source_software="sDANNCE",
    source_software_version="1.0.0",
    skeleton=skeleton,
)

# ---------------------------------------------------------------------------
# 8. Add everything to a behaviour processing module
# ---------------------------------------------------------------------------
behavior_pm = nwbfile.create_processing_module(
    name="behavior",
    description="Processed behavioral data - 3D pose estimation.",
)
behavior_pm.add(skeletons)
behavior_pm.add(mcpe)

# ---------------------------------------------------------------------------
# 9. Write to disk and read back
# ---------------------------------------------------------------------------
path = "test_multicamera_pose.nwb"

with NWBHDF5IO(path, mode="w") as io:
    io.write(nwbfile)

print(f"Wrote {path}")

with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
    read_nwbfile = io.read()
    read_mcpe = read_nwbfile.processing["behavior"]["MultiCameraPoseEstimation"]

    print(read_mcpe)
    print(f"  source_software  : {read_mcpe.source_software} {read_mcpe.source_software_version}")
    print(f"  3D body parts    : {list(read_mcpe.pose_estimation_series.keys())}")
    print(f"  camera views     : {list(read_mcpe.pose_estimations.keys())}")
    print(f"  skeleton nodes   : {list(read_mcpe.skeleton.nodes)}")

    pe1 = read_mcpe.pose_estimations["PoseEstimation_camera1"]
    camera1 = pe1.device
    print(f"  camera1 device        : {camera1.name}")
    print(f"  camera1 intrinsic     : {camera1.intrinsic_matrix[:]}")
    print(f"  camera1 video         : {pe1.source_video.external_file[0]}")
    print(f"  camera1 2D keypoints  : {len(pe1.pose_estimation_series)} (none for sDANNCE)")
