import datetime
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.image import ImageSeries
from ndx_pose import (
    Skeleton,
    SkeletonInstance,
    TrainingFrame,
    PoseTraining,
    Skeletons,
    TrainingFrames,
    SourceVideos,
    SkeletonInstances,
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

# in this example, we have two subjects in the training data and therefore two skeletons.
# each skeleton defines the relationship between the markers.
# Skeleton names must be unique because the Skeleton objects will be added to a Skeletons container object
# which requires unique names.
skeleton1 = Skeleton(
    name="subject1_skeleton",
    nodes=["front_left_paw", "body", "front_right_paw"],
    # edge between front left paw and body, edge between body and front right paw.
    # the values are the indices of the nodes in the nodes list.
    edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
)
skeleton2 = Skeleton(
    name="subject2_skeleton",
    nodes=["front_left_paw", "body", "front_right_paw"],
    # edge between front left paw and body, edge between body and front right paw.
    # the values are the indices of the nodes in the nodes list.
    edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
)

skeletons = Skeletons(skeletons=[skeleton1, skeleton2])

# create a device for the camera
camera1 = nwbfile.create_device(
    name="camera1",
    description="camera for recording behavior",
    manufacturer="my manufacturer",
)

# next, we specify the ground truth data that was used to train the pose estimation model.
# this includes the training video and the ground truth annotations for each frame.

# create an ImageSeries that represents the raw video that was used to train the pose estimation model.
# the video can be stored as an MP4 file that is linked to from this ImageSeries object.
# if there are multiple videos, the names must be unique because they will be added to a SourceVideos
# container object which requires unique names.
training_video1 = ImageSeries(
    name="source_video",
    description="Training video used to train the pose estimation model.",
    unit="NA",
    format="external",
    external_file=["path/to/camera1.mp4"],
    dimension=[640, 480],
    starting_frame=[0],
    rate=30.0,
)

# initial locations ((x, y) coordinates) of each node in the skeleton.
# the order of the nodes is defined by the skeleton.
node_locations_sk1 = np.array(
    [
        [10.0, 10.0],  # front_left_paw
        [20.0, 20.0],  # body
        [30.0, 10.0],  # front_right_paw
    ]
)
node_locations_sk2 = np.array(
    [
        [40.0, 40.0],  # front_left_paw
        [50.0, 50.0],  # body
        [60.0, 60.0],  # front_right_paw
    ]
)

# in this example, frame indices 0, 5, 10, ..., 500 from the training video were used for training.
# each training frame has two skeleton instances, one for each skeleton.
training_frames_list = []
for i in range(0, 500, 5):
    skeleton_instances_list = []
    # add some noise to the node locations from the previous frame
    node_locations_sk1 = node_locations_sk1 + np.random.rand(3, 2)
    instance_sk1 = SkeletonInstance(
        name="skeleton1_instance",
        id=np.uint(i),
        node_locations=node_locations_sk1,
        node_visibility=[
            True,  # front_left_paw
            True,  # body
            True,  # front_right_paw
        ],
        skeleton=skeleton1,  # link to the skeleton
    )
    skeleton_instances_list.append(instance_sk1)

    # add some noise to the node locations from the previous frame
    node_locations_sk2 = node_locations_sk2 + np.random.rand(3, 2)
    instance_sk2 = SkeletonInstance(
        name="skeleton2_instance",
        id=np.uint(i),
        node_locations=node_locations_sk2,
        node_visibility=[
            True,  # front_left_paw
            True,  # body
            True,  # front_right_paw
        ],
        skeleton=skeleton2,  # link to the skeleton
    )
    skeleton_instances_list.append(instance_sk2)

    # store the skeleton instances in a SkeletonInstances object
    skeleton_instances = SkeletonInstances(skeleton_instances=skeleton_instances_list)

    # TrainingFrame names must be unique because the TrainingFrame objects will be added to a
    # TrainingFrames container object which requires unique names.
    # the source video frame index is the index of the frame in the source video, which is useful
    # for linking the training frames to the source video.
    training_frame = TrainingFrame(
        name=f"frame_{i}",
        annotator="Bilbo Baggins",
        skeleton_instances=skeleton_instances,
        source_video=training_video1,
        source_video_frame_index=np.uint(i),
    )
    training_frames_list.append(training_frame)

# store the training frames and source videos in their corresponding container objects
training_frames = TrainingFrames(training_frames=training_frames_list)
source_videos = SourceVideos(image_series=[training_video1])

# store the skeletons group, training frames group, and source videos group in a PoseTraining object
pose_training = PoseTraining(
    training_frames=training_frames,
    source_videos=source_videos,
)

# create a "behavior" processing module to store the PoseEstimation and PoseTraining objects
behavior_pm = nwbfile.create_processing_module(
    name="behavior",
    description="processed behavioral data",
)
behavior_pm.add(skeletons)
behavior_pm.add(pose_training)

# write the NWBFile to disk
path = "test_pose.nwb"
with NWBHDF5IO(path, mode="w") as io:
    io.write(nwbfile)

# read the NWBFile from disk and print out the PoseEstimation and PoseTraining objects
# as well as the first training frame
with NWBHDF5IO(path, mode="r") as io:
    read_nwbfile = io.read()
    print(read_nwbfile.processing["behavior"]["Skeletons"]["subject1_skeleton"])
    read_pt = read_nwbfile.processing["behavior"]["PoseTraining"]
    print(read_pt.training_frames["frame_10"].skeleton_instances["skeleton2_instance"].node_locations[:])
