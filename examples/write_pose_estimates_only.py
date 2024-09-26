"""
Example script that demonstrates how to write a PoseEstimation and Skeletons object to an NWB file.


With one camera, one video, one skeleton, and three body parts per skeleton.
"""

import datetime
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from ndx_pose import (
    PoseEstimationSeries,
    PoseEstimation,
    Skeleton,
    Skeletons,
)

# initialize an NWBFile object
nwbfile = NWBFile(
    session_description="session_description",
    identifier="identifier",
    session_start_time=datetime.datetime.now(datetime.timezone.utc),
)

# add a subject to the NWB file
subject = Subject(subject_id="subject1", species="Mus musculus")
nwbfile.subject = subject

# create a skeleton that define the relationship between the markers. also link this skeleton to the subject.
skeleton = Skeleton(
    name="subject1_skeleton",
    nodes=["front_left_paw", "body", "front_right_paw"],
    # define edges between nodes using the indices of the nodes in the node list.
    # this array represents an edge between front left paw and body, and an edge between body and front right paw.
    edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
    subject=subject,
)

# store the skeleton into a Skeletons container object.
# (this is more useful if you have multiple skeletons in your training data)
skeletons = Skeletons(skeletons=[skeleton])

# create a device for the camera
camera1 = nwbfile.create_device(
    name="camera1",
    description="camera for recording behavior",
    manufacturer="my manufacturer",
)

# a PoseEstimationSeries represents the estimated position of a single marker.
# in this example, we have three PoseEstimationSeries: one for the body and one for each front paw.
data = np.random.rand(100, 2)  # num_frames x (x, y) but can be (x, y, z)
timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
confidence = np.random.rand(100)  # a confidence value for every frame
reference_frame = "(0,0,0) corresponds to ..."
confidence_definition = "Softmax output of the deep neural network."

front_left_paw = PoseEstimationSeries(
    name="front_left_paw",
    description="Marker placed around fingers of front left paw.",
    data=data,
    unit="pixels",
    reference_frame=reference_frame,
    timestamps=timestamps,
    confidence=confidence,
    confidence_definition=confidence_definition,
)

data = np.random.rand(100, 2)  # num_frames x (x, y) but can be (x, y, z)
confidence = np.random.rand(100)  # a confidence value for every frame
body = PoseEstimationSeries(
    name="body",
    description="Marker placed on center of body.",
    data=data,
    unit="pixels",
    reference_frame=reference_frame,
    timestamps=front_left_paw,  # link to timestamps of front_left_paw so we don't have to duplicate them
    confidence=confidence,
    confidence_definition=confidence_definition,
)

data = np.random.rand(100, 2)  # num_frames x (x, y) but can be num_frames x (x, y, z)
confidence = np.random.rand(100)  # a confidence value for every frame
front_right_paw = PoseEstimationSeries(
    name="front_right_paw",
    description="Marker placed around fingers of front right paw.",
    data=data,
    unit="pixels",
    reference_frame=reference_frame,
    timestamps=front_left_paw,  # link to timestamps of front_left_paw so we don't have to duplicate them
    confidence=confidence,
    confidence_definition=confidence_definition,
)

# store all PoseEstimationSeries in a list
pose_estimation_series = [front_left_paw, body, front_right_paw]

# create a PoseEstimation object that represents the estimated positions of each node, references
# the original video and labeled video files, and provides metadata on how these estimates were generated.
# multiple videos and cameras can be referenced.
pose_estimation = PoseEstimation(
    name="PoseEstimation",
    pose_estimation_series=pose_estimation_series,
    description="Estimated positions of front paws of subject1 using DeepLabCut.",
    original_videos=["path/to/camera1.mp4"],
    labeled_videos=["path/to/camera1_labeled.mp4"],
    dimensions=np.array([[640, 480]], dtype="uint16"),  # pixel dimensions of the video
    devices=[camera1],
    scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
    source_software="DeepLabCut",
    source_software_version="2.3.8",
    skeleton=skeleton,  # link to the skeleton object
)

# create a "behavior" processing module to store the PoseEstimation and Skeletons objects
behavior_pm = nwbfile.create_processing_module(
    name="behavior",
    description="processed behavioral data",
)
behavior_pm.add(skeletons)
behavior_pm.add(pose_estimation)

# write the NWBFile to disk
path = "test_pose.nwb"
with NWBHDF5IO(path, mode="w") as io:
    io.write(nwbfile)

# read the NWBFile from disk and print out the PoseEstimation and Skeleton objects
# as well as the first training frame
with NWBHDF5IO(path, mode="r") as io:
    read_nwbfile = io.read()
    print(read_nwbfile.processing["behavior"]["PoseEstimation"])
    print(read_nwbfile.processing["behavior"]["Skeletons"]["subject1_skeleton"])
