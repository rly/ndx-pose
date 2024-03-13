# ndx-pose Extension for NWB

[![PyPI version](https://badge.fury.io/py/ndx-pose.svg)](https://badge.fury.io/py/ndx-pose)

ndx-pose is a standardized format for storing pose estimation data in NWB. It was developed initially to store the
output of [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) in NWB, but is also designed to store the output
of general pose estimation tools. Please post an issue or PR to suggest or add support for your favorite pose
estimation tool.

This extension consists of several new neurodata types:
- `PoseEstimationSeries` which stores the estimated positions (x, y) or (x, y, z) of a body part over time as well as
the confidence/likelihood of the estimated positions.
- `PoseEstimation` which stores the estimated position data (`PoseEstimationSeries`) for multiple body parts,
computed from the same video(s) with the same tool/algorithm.
- `Skeleton` which stores the relationship between the body parts (nodes and edges).
- `SkeletonInstance` which stores the estimated positions and visibility of the body parts for a single frame.
- `TrainingFrame` which stores the ground truth data for a single frame. It references a `SkeletonInstance` and
frame of a source video (`ImageSeries`). The source videos can be stored internally as data arrays or externally as
files referenced by relative file path.
- `Skeletons` which stores multiple `Skeleton` objects.
- `TrainingFrames` which stores multiple `TrainingFrame` objects.
- `SourceVideos` which stores multiple `ImageSeries` objects representing source videos used in training.
- `PoseTraining` which stores the skeletons (`Skeletons`), ground truth data (`TrainingFrames`), and
source videos (`SourceVideos`) used to train the pose estimation model.

If training data are not available, then the `PoseTraining` object should contain only the `Skeletons` object
which contains the `Skeleton` object(s) used to generate the pose estimates.

## Installation

`pip install ndx-pose`

## Usage

### Example storing pose estimates (keypoints) and training data
With one camera, one video, three body parts, and 50 training frames.

```python
import datetime
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
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
)
from pynwb.image import ImageSeries

# initialize an NWBFile object
nwbfile = NWBFile(
    session_description="session_description",
    identifier="identifier",
    session_start_time=datetime.datetime.now(datetime.timezone.utc),
)

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

# note the double underscore in "confidence__definition" because this is a property of the "confidence" field
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

# create a skeleton that defines the relationship between the markers
skeleton = Skeleton(
    name="subject1",
    nodes=["front_left_paw", "body", "front_right_paw"],
    # edge between front left paw and body, edge between body and front right paw.
    # the values are the indices of the nodes in the nodes list.
    edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
)

# create a PoseEstimation object that represents the estimated positions of the front paws and body
# from DLC and references the original video simultaneously recorded from one camera and the labeled
# video that was generated by DLC. multiple videos and cameras can be referenced.
pose_estimation = PoseEstimation(
    pose_estimation_series=pose_estimation_series,
    description="Estimated positions of front paws using DeepLabCut.",
    original_videos=["path/to/camera1.mp4"],
    labeled_videos=["path/to/camera1_labeled.mp4"],
    dimensions=np.array([[640, 480]], dtype="uint16"),  # pixel dimensions of the video
    devices=[camera1],
    scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
    source_software="DeepLabCut",
    source_software_version="2.3.8",
    skeleton=skeleton,
)

# next, we specify the ground truth data that was used to train the pose estimation model.
# this includes the training video and the ground truth annotations for each frame.
# this is optional. if you don't have ground truth data, you can skip this step.

# create an ImageSeries that represents the raw video that was used to train the pose estimation model
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

# create 50 ground truth instances of the skeleton at slightly random positions.
# in this example, each node is visible on every frame.
# the mapping of index in node_locations and node_visibilities to label is defined by the skeleton.

# the node locations are the (x, y) coordinates of each node in the skeleton.
# the order of the nodes is defined by the skeleton.
node_locations = np.array(
    [
        [10.0, 10.0],  # front_left_paw
        [20.0, 20.0],  # body
        [30.0, 10.0],  # front_right_paw
    ]
)

skeleton_instances = []
for i in range(50):
    # add some noise to the node locations from the location on the previous frame
    node_locations = node_locations + np.random.rand(3, 2)
    instance = SkeletonInstance(
        id=np.uint(i),
        node_locations=node_locations,
        node_visibility=[
            True,  # front_left_paw
            True,  # body
            True,  # front_right_paw
        ],
        skeleton=skeleton,  # link to the skeleton
    )
    skeleton_instances.append(instance)

# create 50 training frames using the training video and the skeleton instances.
# the skeleton instances start with video frame 0 and end with video frame 49.
training_frames_list = []
for i in range(50):
    # names must be unique within a PoseTraining object (we will add them to a PoseTraining object below)
    training_frame = TrainingFrame(
        name="frame_{}".format(i),
        annotator="Bilbo Baggins",
        skeleton_instance=skeleton_instances[i],
        source_video=training_video1,
        source_video_frame_index=np.uint(i),
    )
    training_frames_list.append(training_frame)

# store the skeletons, training frames, and source videos in their corresponding grouping objects
skeletons = Skeletons(skeletons=[skeleton])
training_frames = TrainingFrames(training_frames=training_frames_list)
source_videos = SourceVideos(image_series=[training_video1])

# store the skeletons group, training frames group, and source videos group in a PoseTraining object
pose_training = PoseTraining(
    skeletons=skeletons,
    training_frames=training_frames,
    source_videos=source_videos,
)

# create a "behavior" processing module to store the PoseEstimation and PoseTraining objects
behavior_pm = nwbfile.create_processing_module(
    name="behavior",
    description="processed behavioral data",
)
behavior_pm.add(pose_estimation)
behavior_pm.add(pose_training)

# write the NWBFile to disk
path = "test_pose.nwb"
with NWBHDF5IO(path, mode="w") as io:
    io.write(nwbfile)

# read the NWBFile from disk and print out the PoseEstimation and PoseTraining objects
# as well as the first training frame
with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
    read_nwbfile = io.read()
    read_pe = read_nwbfile.processing["behavior"]["PoseEstimation"]
    print(read_pe)
    read_pt = read_nwbfile.processing["behavior"]["PoseTraining"]
    print(read_pt)
    print(read_pt.training_frames["frame_0"])
```


## Discussion

1. Should we map `PoseEstimationSeries.confidence__definition` -> `PoseEstimationSeries.confidence_definition` and
`PoseEstimation.source_software__version` -> `PoseEstimation.source_software_version` in the Python API?
Note that the Matlab API uses a different format for accessing these fields.
- Pros:
    - Stays consistent with version 0.1.0-0.1.1
    - When ndx-pose is integrated into the NWB core, the con below will not be relevant, and we will probably
      want to use the single underscore version because it's slightly more readable and consistent with the
      rest of PyNWB, so it would be better to start using that version now.
- Cons:
    - When reading data in Python, the code is different depending on whether the Python classes from ndx-pose
      are used or the classes are generated from the schema directly.

2. Should we create a typed group that would be contained within a `PoseTraining` object to contain `Skeleton` objects,
e.g., a `Skeletons` group? This would be similar to the `Position` group containing `SpatialSeries` objects and a `BehavioralTimeSeries` group containing `TimeSeries` objects, except that unlike `SpatialSeries` and `TimeSeries`,
`Skeleton` is not really a multi-purpose neurodata type. This makes the most sense for `SourceVideos`, because
`ImageSeries` (i.e., a video), is pretty generic, and `SourceVideos` would tag the video as a source video for training
and the auto-generated functions and variables in the parent `PoseTraining` object would use
"source_videos" instead of "image_series".

Similarly, should we create typed groups that would be contained within a `PoseTraining` object to contain
`TrainingFrame` objects and `ImageSeries` objects? This type would be purely organizational.
`NWBFile` has an untyped `acquisition` group and an untyped `processing` group for organization and tagging.
A `ProcessingModule` is a typed group that exists solely for organization and tagging with custom names and
descriptions.

3. Currently, multiple `Skeleton` objects are allowed. Should they be allowed?
If so, how do `Skeleton` objects relate to `Subject` objects?
NWB files are designed to contain data from a single subject.
Initially, ndx-pose was designed to store pose estimates from a single subject.
Data from multiple subjects would be stored in separate NWB files.
See https://github.com/rly/ndx-pose/pull/3

4. If a user annotates 500 frames of a video, there will be 500 groups. Should the node locations and visibilities
instead be stored in a set of PoseEstimationSeries-like TimeSeries objects? We can add in the API the ability to
extract a SkeletonInstance from a set of those objects and create a set of those objects from a set of TrainingFrames.
This would also make for a more consistent storage pattern for keypoint data.

## Resources

Utilities to convert DLC output to/from NWB: https://github.com/DeepLabCut/DLC2NWB
- For multi-animal projects, one NWB file is created per animal. The NWB file contains only a `PoseEstimation` object
  under `/processing/behavior`. That `PoseEstimation` object contains `PoseEstimationSeries` objects, one for each
  body part, and general metadata about the pose estimation process, skeleton, and videos. The
  `PoseEstimationSeries` objects contain the estimated positions for that body part for a particular animal.

Utilties to convert SLEAP pose tracking data to/from NWB: https://github.com/talmolab/sleap-io
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
