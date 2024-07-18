# Run this with ndx-pose 0.1.*
# In ndx-pose 0.2.0, the "nodes" and "edges" constructor arguments of PoseEstimation were deprecated.
# A warning should be raised if they are used when not reading them from a file.
# In ndx-pose 0.2.0, PoseEstimation was changed to raise a DeprecationWarning if the number of original videos,
# labeled videos, or dimensions does not equal the number of camera devices when creating the object not from a file.

import datetime
from ndx_pose import PoseEstimationSeries, PoseEstimation
import numpy as np
from pathlib import Path
from pynwb import NWBFile, NWBHDF5IO


def generate_pose_estimation_nodes_edges():
    """Generate an NWB file with a PoseEstimation object with deprecated fields "nodes" and "edges"."""
    nwbfile = NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    data = np.random.rand(100, 3)  # num_frames x (x, y, z)
    timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
    confidence = np.random.rand(100)  # a confidence value for every frame
    front_left_paw = PoseEstimationSeries(
        name="front_left_paw",
        description="Marker placed around fingers of front left paw.",
        data=data,
        unit="pixels",
        reference_frame="(0,0,0) corresponds to ...",
        timestamps=timestamps,
        confidence=confidence,
        confidence_definition="Softmax output of the deep neural network.",
    )

    data = np.random.rand(100, 2)  # num_frames x (x, y)
    timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
    confidence = np.random.rand(100)  # a confidence value for every frame
    front_right_paw = PoseEstimationSeries(
        name="front_right_paw",
        description="Marker placed around fingers of front right paw.",
        data=data,
        unit="pixels",
        reference_frame="(0,0,0) corresponds to ...",
        timestamps=front_left_paw,  # link to timestamps of front_left_paw
        confidence=confidence,
        confidence_definition="Softmax output of the deep neural network.",
    )

    pose_estimation_series = [front_left_paw, front_right_paw]

    pe = PoseEstimation(
        pose_estimation_series=pose_estimation_series,
        description="Estimated positions of front paws using DeepLabCut.",
        scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
        source_software="DeepLabCut",
        source_software_version="2.2b8",
        nodes=["front_left_paw", "front_right_paw"],
        edges=np.array([[0, 1]], dtype="uint8"),
    )

    behavior_pm = nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
    behavior_pm.add(pe)

    path = Path(__file__).parent / "0.1.1_poseestimation_nodes_edges.nwb"
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)


def generate_pose_estimation_no_cameras():
    """Generate an NWB file with a PoseEstimation object with video metadata and no cameras."""
    nwbfile = NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )
    pe = PoseEstimation(
        pose_estimation_series=[],
        description="Estimated positions of front paws using DeepLabCut.",
        original_videos=["camera1.mp4", "camera2.mp4"],
        labeled_videos=["camera1_labeled.mp4", "camera2_labeled.mp4"],
        dimensions=np.array([[640, 480], [1024, 768]], dtype="uint8"),
        scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
        source_software="DeepLabCut",
        source_software_version="2.2b8",
    )

    behavior_pm = nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
    behavior_pm.add(pe)

    path = Path(__file__).parent / "0.1.1_poseestimation_no_cameras.nwb"
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)


if __name__ == "__main__":
    generate_pose_estimation_nodes_edges()
    generate_pose_estimation_no_cameras()
