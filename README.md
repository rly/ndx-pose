# ndx-pose Extension for NWB

[![PyPI version](https://badge.fury.io/py/ndx-pose.svg)](https://badge.fury.io/py/ndx-pose)

ndx-pose is a standardized format for storing pose estimation data in NWB. It was developed initially to store the
output of [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) in NWB, but is also designed to store the output
of general pose estimation tools. Please post an issue or PR to suggest or add support for your favorite pose 
estimation tool.

This extension consists of two new neurodata types:
- `PoseEstimationSeries` which stores the estimated positions (x, y) or (x, y, z) of a body part over time as well as
the confidence/likelihood of the estimated positions.
- `PoseEstimation` which stores the estimated position data (`PoseEstimationSeries`) for multiple body parts,
computed from the same video(s) with the same tool/algorithm.

## Installation

`pip install ndx-pose`

## Usage
```python
import datetime
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation

nwbfile = NWBFile(
    session_description='session_description',
    identifier='identifier',
    session_start_time=datetime.datetime.now(datetime.timezone.utc)
)

camera1 = nwbfile.create_device(
    name='camera1',
    description='left camera',
    manufacturer='my manufacturer'
)
camera2 = nwbfile.create_device(
    name='camera2',
    description='right camera',
    manufacturer='my manufacturer'
)

data = np.random.rand(100, 3)  # num_frames x (x, y, z)
timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
confidence = np.random.rand(100)  # a confidence value for every frame
front_left_paw = PoseEstimationSeries(
    name='front_left_paw',
    description='Marker placed around fingers of front left paw.',
    data=data,
    unit='pixels',
    reference_frame='(0,0,0) corresponds to ...',
    timestamps=timestamps,
    confidence=confidence,
    confidence_definition='Softmax output of the deep neural network.',
)

data = np.random.rand(100, 2)  # num_frames x (x, y)
timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
confidence = np.random.rand(100)  # a confidence value for every frame
front_right_paw = PoseEstimationSeries(
    name='front_right_paw',
    description='Marker placed around fingers of front right paw.',
    data=data,
    unit='pixels',
    reference_frame='(0,0,0) corresponds to ...',
    timestamps=front_left_paw,  # link to timestamps of front_left_paw
    confidence=confidence,
    confidence_definition='Softmax output of the deep neural network.',
)

pose_estimation_series = [front_left_paw, front_right_paw]

pe = PoseEstimation(
    pose_estimation_series=pose_estimation_series,
    description='Estimated positions of front paws using DeepLabCut.',
    original_videos=['camera1.mp4', 'camera2.mp4'],
    labeled_videos=['camera1_labeled.mp4', 'camera2_labeled.mp4'],
    dimensions=np.array([[640, 480], [1024, 768]], dtype='uint8'),
    scorer='DLC_resnet50_openfieldOct30shuffle1_1600',
    source_software='DeepLabCut',
    source_software_version='2.2b8',
    nodes=['front_left_paw', 'front_right_paw'],
    edges=np.array([[0, 1]], dtype='uint8'),
    # devices=[camera1, camera2],  # this is not yet supported
)

behavior_pm = nwbfile.create_processing_module(
    name='behavior',
    description='processed behavioral data'
)
behavior_pm.add(pe)

path = 'test_pose.nwb'
with NWBHDF5IO(path, mode='w') as io:
    io.write(nwbfile)

with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    read_nwbfile = io.read()
    read_pe = read_nwbfile.processing['behavior']['PoseEstimation']
    print(read_pe)
```


## Contributors
- @rly
- @bendichter
- @AlexEMG

This extension was created using [ndx-template](https://github.com/nwb-extensions/ndx-template).
