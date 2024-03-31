# ndx-pose Extension for NWB

[![PyPI version](https://badge.fury.io/py/ndx-pose.svg)](https://badge.fury.io/py/ndx-pose)

ndx-pose is a standardized format for storing pose estimation data in NWB, such as from
[DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) and [SLEAP](https://sleap.ai/).
Please post an issue or PR to suggest or add support for another pose estimation tool.

This extension consists of several new neurodata types:
- `Skeleton` which stores the relationship between the body parts (nodes and edges).
- `Skeletons` which stores multiple `Skeleton` objects.
- `PoseEstimationSeries` which stores the estimated positions (x, y) or (x, y, z) of a body part over time as well as
the confidence/likelihood of the estimated positions.
- `PoseEstimation` which stores the estimated position data (`PoseEstimationSeries`) for multiple body parts,
computed from the same video(s) with the same tool/algorithm.
- `SkeletonInstance` which stores the estimated positions and visibility of the body parts for a single frame.
- `TrainingFrame` which stores the ground truth data for a single frame. It contains `SkeletonInstance` objects and
references a frame of a source video (`ImageSeries`). The source videos can be stored internally as data arrays or
externally as files referenced by relative file path.
- `TrainingFrames` which stores multiple `TrainingFrame` objects.
- `SourceVideos` which stores multiple `ImageSeries` objects representing source videos used in training.
- `PoseTraining` which stores the ground truth data (`TrainingFrames`) and source videos (`SourceVideos`)
used to train the pose estimation model.

It is recommended to place the `Skeletons`, `PoseEstimation`, and `PoseTraining` objects in an NWB processing module
named "behavior", as shown below.

## Installation

`pip install ndx-pose`

## Usage

### Example storing pose estimates (keypoints)
With one camera, one video, one skeleton, and three body parts per skeleton.

```python
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
    dimensions=np.array(
        [[640, 480]], dtype="uint16"
    ),  # pixel dimensions of the video
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
```

### Example storing pose estimates and training data (keypoints)
With one camera, one video, two skeletons (but only one pose estimate), three body parts per skeleton,
50 training frames with two skeleton instances per frame, and one source video.

```python
import datetime
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.image import ImageSeries
from ndx_pose import (
    PoseEstimationSeries,
    PoseEstimation,
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

# a PoseEstimationSeries represents the estimated position of a single marker.
# in this example, we have three PoseEstimationSeries: one for the body and one for each front paw.
# a single NWB file contains pose estimation data for a single subject. if you have pose estimates for
# multiple subjects, store them in separate files.
data = np.random.rand(100, 2)  # num_frames x (x, y) but can be num_frames x (x, y, z)
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

data = np.random.rand(100, 2)  # num_frames x (x, y) but can be (x, y, z)
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
    dimensions=np.array(
        [[640, 480]], dtype="uint16"
    ),  # pixel dimensions of the video
    devices=[camera1],
    scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
    source_software="DeepLabCut",
    source_software_version="2.3.8",
    skeleton=skeleton1,  # link to the skeleton
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
    skeleton_instances = SkeletonInstances(
        skeleton_instances=skeleton_instances_list
    )

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
behavior_pm.add(pose_estimation)
behavior_pm.add(pose_training)

# write the NWBFile to disk
path = "test_pose.nwb"
with NWBHDF5IO(path, mode="w") as io:
    io.write(nwbfile)

# read the NWBFile from disk and print out the PoseEstimation and PoseTraining objects
# as well as the first training frame
with NWBHDF5IO(path, mode="r") as io:
    read_nwbfile = io.read()
    print(read_nwbfile.processing["behavior"]["PoseEstimation"])
    print(read_nwbfile.processing["behavior"]["Skeletons"]["subject1_skeleton"])
    read_pt = read_nwbfile.processing["behavior"]["PoseTraining"]
    print(read_pt.training_frames["frame_10"].skeleton_instances["skeleton2_instance"].node_locations[:])
```


## Discussion

1. Currently, the spec `PoseEstimationSeries.confidence__definition` is mapped to the Python attribute
`PoseEstimationSeries.confidence_definition` and the spec
`PoseEstimation.source_software__version` is mapped to the Python attribute
`PoseEstimation.source_software_version`, only in the Python API.
Note that the Matlab API uses a different format for accessing these fields. Should we maintain this mapping?

- Pros:
    - Stays consistent with version 0.1.0-0.1.1
    - When ndx-pose is integrated into the NWB core, the con below will not be relevant, and we will probably
      want to use the single underscore version because it's slightly more readable and consistent with the
      rest of PyNWB, so it would be better to start using that version now.
- Cons:
    - When reading data in Python, the code is different depending on whether the Python classes from ndx-pose
      are used or the classes are generated from the schema directly.
    - PyNWB may eventually get rid of custom I/O mapping.

2. If a user annotates 500 frames of a video, there will be 500 groups. Should the node locations and visibilities
instead be stored in a set of `PoseEstimationSeries`-like `TimeSeries` objects? We can add in the API the ability to
extract a `SkeletonInstance` from a set of those objects and create a set of those objects from a set of
`TrainingFrame` objects. This would also make for a more consistent storage pattern for keypoint data.

## Handling pose estimates for multiple subjects

NWB files are designed to store data from a single subject and have only one root-level `Subject` object.
As a result, ndx-pose was designed to store pose estimates from a single subject.
Pose estimates data from different subjects should be stored in separate NWB files.

Training images can involve multiple skeletons, however. These training images may be the same across subjects,
and therefore the same across NWB files. These training images should be duplicated between files, until
multi-subject support is added to NWB and ndx-pose. See https://github.com/rly/ndx-pose/pull/3

## Resources

Utilities to convert DLC output to/from NWB: https://github.com/DeepLabCut/DLC2NWB
- For multi-animal projects, one NWB file is created per animal. The NWB file contains only a `PoseEstimation` object
  under `/processing/behavior`. That `PoseEstimation` object contains `PoseEstimationSeries` objects, one for each
  body part, and general metadata about the pose estimation process, skeleton, and videos. The
  `PoseEstimationSeries` objects contain the estimated positions for that body part for a particular animal.

Utilities to convert SLEAP pose tracking data to/from NWB: https://github.com/talmolab/sleap-io
- Used by SLEAP (sleap.io.dataset.Labels.export_nwb)
- See also https://github.com/talmolab/sleap/blob/develop/sleap/io/format/ndx_pose.py

Keypoint MoSeq: https://github.com/dattalab/keypoint-moseq
- Supports read of `PoseEstimation` objects from NWB files.

NeuroConv: https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/conversion_example_gallery.html#behavior
- NeuroConv supports converting data from DeepLabCut (using `dlc2nwb` described above),
  SLEAP (using `sleap_io` described above), FicTrac, and LightningPose to NWB. It supports appending pose estimation data to an existing NWB file.

Ethome: Tools for machine learning of animal behavior: https://github.com/benlansdell/ethome
- Supports read of `PoseEstimation` objects from NWB files.

Related work:
- https://github.com/ndx-complex-behavior
- https://github.com/datajoint/element-deeplabcut

Several NWB datasets use ndx-pose 0.1.1:
- [A detailed behavioral, videographic, and neural dataset on object recognition in mice](https://dandiarchive.org/dandiset/000231)
- [IBL Brain Wide Map](https://dandiarchive.org/dandiset/000409)

Several [open-source conversion scripts on GitHub](https://github.com/search?q=ndx-pose&type=code&p=1)
also use ndx-pose.

## Contributors
- @rly
- @bendichter
- @AlexEMG
- @roomrys
- @CBroz1
- @h-mayorquin
- @talmo
- @eberrigan

This extension was created using [ndx-template](https://github.com/nwb-extensions/ndx-template).
