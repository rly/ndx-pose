# Run this with ndx-pose 0.3.*
# In ndx-pose 0.4.0, PoseEstimation was scoped to a single camera view: the "devices" constructor argument
# (a list) was deprecated in favor of the singular "device" argument, and the schema replaced the unnamed,
# zero-or-more Device link with a link named "device".
# A PoseEstimation object written by an earlier version that links one camera still reads. One that links
# more than one camera raises, because there is no single "device" to assign it to.

import datetime
from ndx_pose import PoseEstimationSeries, PoseEstimation, Skeleton, Skeletons
import numpy as np
from pathlib import Path
from pynwb import NWBFile, NWBHDF5IO


def generate_pose_estimation(num_cameras, path_name):
    """Generate an NWB file with a PoseEstimation object linked to num_cameras camera devices."""
    rng = np.random.default_rng(0)
    nwbfile = NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    devices = [nwbfile.create_device(name="camera%d" % (i + 1)) for i in range(num_cameras)]

    skeleton = Skeleton(
        name="subject",
        nodes=["front_left_paw", "front_right_paw"],
        edges=np.array([[0, 1]], dtype="uint8"),
    )

    timestamps = np.linspace(0, 10, num=10)
    front_left_paw = PoseEstimationSeries(
        name="front_left_paw",
        description="Marker placed around fingers of front left paw.",
        data=rng.random((10, 2)),
        unit="pixels",
        reference_frame="(0,0) corresponds to the top left corner of the video.",
        timestamps=timestamps,
        confidence=rng.random(10),
        confidence_definition="Softmax output of the deep neural network.",
    )
    front_right_paw = PoseEstimationSeries(
        name="front_right_paw",
        description="Marker placed around fingers of front right paw.",
        data=rng.random((10, 2)),
        unit="pixels",
        reference_frame="(0,0) corresponds to the top left corner of the video.",
        timestamps=front_left_paw,  # link to timestamps of front_left_paw
        confidence=rng.random(10),
        confidence_definition="Softmax output of the deep neural network.",
    )

    pe = PoseEstimation(
        pose_estimation_series=[front_left_paw, front_right_paw],
        description="Estimated positions of front paws using DeepLabCut.",
        original_videos=["camera%d.mp4" % (i + 1) for i in range(num_cameras)],
        labeled_videos=["camera%d_labeled.mp4" % (i + 1) for i in range(num_cameras)],
        dimensions=np.array([[640, 480]] * num_cameras, dtype="uint16"),
        devices=devices,
        scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
        source_software="DeepLabCut",
        source_software_version="2.2b8",
        skeleton=skeleton,
    )

    behavior_pm = nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
    behavior_pm.add(pe)
    behavior_pm.add(Skeletons(skeletons=[skeleton]))

    path = Path(__file__).parent / path_name
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)


if __name__ == "__main__":
    generate_pose_estimation(1, "0.3.0_poseestimation_one_camera.nwb")
    generate_pose_estimation(2, "0.3.0_poseestimation_two_cameras.nwb")
