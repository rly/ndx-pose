# ndx-pose Extension for NWB

[![PyPI version](https://badge.fury.io/py/ndx-pose.svg)](https://badge.fury.io/py/ndx-pose)

ndx-pose is a standardized format for storing pose estimation data in NWB, such as from
[DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), [SLEAP](https://sleap.ai/), and
[DANNCE](https://github.com/spoonsso/dannce).
Please post an issue or PR to suggest or add support for another pose estimation tool.

## Data types overview

This extension consists of several new neurodata types. They are divided into two main categories:

1. **Pose estimation data**: This includes the estimated positions of body parts (keypoints) over time, along with
   metadata about the pose estimation process.
2. **Training data**: This includes ground truth data for training pose estimation models, such as labeled images and
   video frames.

## Pose estimation data types

- `Skeleton` which stores the relationship between the body parts (nodes and edges).
- `Skeletons` which is a container that stores multiple `Skeleton` objects.
- `PoseEstimationSeries` which stores the estimated positions (x, y) or (x, y, z) of a body part over time as well as
  the confidence/likelihood of the estimated positions.
- `PoseEstimation` which stores the estimated position data (`PoseEstimationSeries`) for multiple body parts,
  computed from a single camera view with the same tool/algorithm, and links to the `Device` (camera) used.

### Multi-camera 3D pose estimation types

For multi-camera setups that produce 3D world-space coordinates (e.g. DANNCE, Anipose):

- `CalibratedCamera` which extends `Device` with intrinsic and extrinsic calibration parameters (intrinsic matrix,
  rotation matrix, translation vector, distortion coefficients) for that single camera. Because it is a `Device`,
  it is added once to the NWBFile and can be linked to by reference from multiple `PoseEstimation`/
  `MultiCameraPoseEstimation` objects (e.g., one per subject in a multi-subject session), so the camera rig and
  its calibration are never duplicated.
- `MultiCameraPoseEstimation` which stores 3D world-space `PoseEstimationSeries`, one `PoseEstimation` per camera
  view (holding that camera's 2D pixel-space estimates and its device link), and an optional link to a `Skeleton`.

## Training data types

- `SkeletonInstance` which stores the estimated positions and visibility of the body parts for a single frame.
- `TrainingFrame` which stores the ground truth data for a single frame. It contains `SkeletonInstance` objects and
references a frame of a source video (`ImageSeries`). The source videos can be stored internally as data arrays or
externally as files referenced by relative file path.
- `TrainingFrames` which is a container that stores multiple `TrainingFrame` objects.
- `SourceVideos` which is a container that stores multiple `ImageSeries` objects representing source videos used in training.
- `PoseTraining` which is a container that stores the ground truth data (`TrainingFrames`) and source videos (`SourceVideos`)
used to train the pose estimation model.

It is recommended to place the `Skeletons`, `PoseEstimation`, `MultiCameraPoseEstimation`, and `PoseTraining` objects
in an NWB processing module named "behavior", as shown below.

## Installation

```bash
pip install "ndx-pose"
```

### Development installation

Development dependencies are defined as [PEP 735](https://peps.python.org/pep-0735/) dependency groups in
`pyproject.toml`. Installing them requires pip 25.1 or later. To set up an editable install with the development
tools (tests, docs, and linters), run:

```bash
git clone https://github.com/rly/ndx-pose.git
cd ndx-pose
pip install -e . --group dev
```

The `test`, `docs`, and `min-reqs` groups can be installed individually with `pip install -e . --group <name>`.

## Usage examples

1. [Example writing pose estimates (keypoints) to an NWB file](examples/write_pose_estimates_only.py).

2. [Example writing training data to an NWB file](examples/write_pose_training.py).

3. [Example writing 3D multi-camera pose estimates to an NWB file](examples/write_multicamera_pose_estimates.py).

## Handling pose estimates for multiple subjects

NWB files are designed to store data from a single subject and have only one root-level `Subject` object.
As a result, ndx-pose was designed to store pose estimates from a single subject.
Pose estimates data from different subjects should be stored in separate NWB files.

Training images can involve multiple skeletons, however. These training images may be the same across subjects,
and therefore the same across NWB files. These training images should be duplicated between files.

## Resources

[Utilities to convert DLC output to/from NWB](https://github.com/DeepLabCut/DLC2NWB)

- For multi-animal projects, one NWB file is created per animal. The NWB file contains only a `PoseEstimation` object
  under `/processing/behavior`. That `PoseEstimation` object contains `PoseEstimationSeries` objects, one for each
  body part, and general metadata about the pose estimation process, skeleton, and videos. The
  `PoseEstimationSeries` objects contain the estimated positions for that body part for a particular animal.

[Utilities to convert SLEAP pose tracking data to/from NWB](https://github.com/talmolab/sleap-io)

- Used by SLEAP (`sleap.io.dataset.Labels.export_nwb`)
- See also [sleap/io/format/ndx_pose.py](https://github.com/talmolab/sleap/blob/develop/sleap/io/format/ndx_pose.py)

[Keypoint MoSeq](https://github.com/dattalab/keypoint-moseq)

- Supports read of `PoseEstimation` objects from NWB files.

[NeuroConv](https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/conversion_example_gallery.html#behavior)

- NeuroConv supports converting data from DeepLabCut, SLEAP (using `sleap_io` described above), and LightningPose to NWB. It also supports appending pose estimation data to an existing NWB file.

[Ethome: Tools for machine learning of animal behavior](https://github.com/benlansdell/ethome)

- Supports read of `PoseEstimation` objects from NWB files.

Related work:

- [ndx-complex-behavior](https://github.com/ndx-complex-behavior)
- [datajoint/element-deeplabcut](https://github.com/datajoint/element-deeplabcut)

Several NWB datasets use ndx-pose 0.1.1:

- [A detailed behavioral, videographic, and neural dataset on object recognition in mice](https://dandiarchive.org/dandiset/000231)
- [IBL Brain Wide Map](https://dandiarchive.org/dandiset/000409)

Several [open-source conversion scripts on GitHub](https://github.com/search?q=ndx-pose&type=code&p=1)
also use ndx-pose.

## Diagram of pose estimation types

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffffff', "primaryBorderColor': '#144E73', 'lineColor': '#D96F32'}}}%%

classDiagram
    direction LR
    namespace ndx-pose {
        class PoseEstimationSeries{
            <<SpatialSeries>>
            name : str
            description : str
            timestamps : array[float; dims [frame]]
            data : array[float; dims [frame, [x, y]] or [frame, [x, y, z]]]
            confidence : array[float; dims [frame]]
            reference_frame: str
        }

        class PoseEstimation {
            <<NWBDataInterface>>
            name : str
            description : str, optional
            original_videos : array[str; dims [file]], optional, deprecated
            labeled_videos : array[str; dims [file]], optional, deprecated
            dimensions : array[uint, dims [file, [width, height]]], optional, deprecated
            scorer : str, optional
            source_software : str, optional
            source_software__version : str, optional
            PoseEstimationSeries
            Skeleton, link, optional
            device : Device, link, optional
            source_video : ImageSeries, link, optional
            labeled_video : ImageSeries, link, optional
        }

        class CalibratedCamera {
            <<Device>>
            intrinsic_matrix : array[float; dims [3, 3]]
            rotation_matrix : array[float; dims [3, 3]], optional
            translation_vector : array[float; dims [3]], optional
            distortion_coefficients : array[float; dims [N]], optional
        }

        class MultiCameraPoseEstimation {
            <<NWBDataInterface>>
            description : str, optional
            scorer : str, optional
            source_software : str, optional
            source_software_version : str, optional
            PoseEstimationSeries (3D world-space)
            PoseEstimation (one per camera view)
            Skeleton, link, optional
        }

        class Skeletons {
            <<NWBDataInterface>>
            Skeleton
        }

        class Skeleton {
            <<NWBDataInterface>>
            name : str
            nodes : array[str; dims [body part]]
            edges : array[uint; dims [edge, [node, node]]]
            subject: link (to pynwb.Subject), optional
        }

    }

    class Device
    class ImageSeries

    PoseEstimation --o PoseEstimationSeries : contains 0 or more
    PoseEstimation --> Skeleton : links to
    PoseEstimation --> Device : links to (device)
    PoseEstimation --> ImageSeries : links to (source_video)

    CalibratedCamera --|> Device : extends

    MultiCameraPoseEstimation --o PoseEstimationSeries : contains 0 or more
    MultiCameraPoseEstimation --o PoseEstimation : contains 0 or more
    MultiCameraPoseEstimation --> Skeleton : links to

    Skeletons --o Skeleton : contains 0 or more
```

## Diagram of all types

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffffff', "primaryBorderColor': '#144E73', 'lineColor': '#D96F32'}}}%%

classDiagram
    direction LR
    namespace ndx-pose {
        class PoseEstimationSeries{
            <<SpatialSeries>>
            name : str
            description : str
            timestamps : array[float; dims [frame]]
            data : array[float; dims [frame, [x, y]] or [frame, [x, y, z]]]
            confidence : array[float; dims [frame]]
            reference_frame: str
        }

        class PoseEstimation {
            <<NWBDataInterface>>
            name : str
            description : str, optional
            original_videos : array[str; dims [file]], optional, deprecated
            labeled_videos : array[str; dims [file]], optional, deprecated
            dimensions : array[uint, dims [file, [width, height]]], optional, deprecated
            scorer : str, optional
            source_software : str, optional
            source_software__version : str, optional
            PoseEstimationSeries
            Skeleton, link, optional
            device : Device, link, optional
            source_video : ImageSeries, link, optional
            labeled_video : ImageSeries, link, optional
        }

        class CalibratedCamera {
            <<Device>>
            intrinsic_matrix : array[float; dims [3, 3]]
            rotation_matrix : array[float; dims [3, 3]], optional
            translation_vector : array[float; dims [3]], optional
            distortion_coefficients : array[float; dims [N]], optional
        }

        class MultiCameraPoseEstimation {
            <<NWBDataInterface>>
            description : str, optional
            scorer : str, optional
            source_software : str, optional
            source_software_version : str, optional
            PoseEstimationSeries (3D world-space)
            PoseEstimation (one per camera view)
            Skeleton, link, optional
        }

        class Skeleton {
            <<NWBDataInterface>>
            name : str
            nodes : array[str; dims [body part]]
            edges : array[uint; dims [edge, [node, node]]]
        }

        class TrainingFrame {
            <<NWBDataInterface>>
            name : str
            annotator : str, optional
            source_video_frame_index : uint, optional
            skeleton_instances : SkeletonInstances
            source_video : ImageSeries, link, optional
            source_frame : Image, link, optional
        }

        class SkeletonInstance {
            <<NWBDataInterface>>
            id: uint, optional
            node_locations : array[float; dims [body part, [x, y]] or [body part, [x, y, z]]]
            node_visibility : array[bool; dims [body part]], optional
            Skeleton, link
        }

        class TrainingFrames {
            <<NWBDataInterface>>
            TrainingFrame
        }

        class SkeletonInstances {
            <<NWBDataInterface>>
            SkeletonInstance
        }

        class SourceVideos {
            <<NWBDataInterface>>
            ImageSeries
        }

        class Skeletons {
            <<NWBDataInterface>>
            Skeleton
        }

        class PoseTraining {
            <<NWBDataInterface>>>
            training_frames : TrainingFrames, optional
            source_videos : SourceVideos, optional
        }

    }

    class Device
    class ImageSeries
    class Image

    PoseEstimation --o PoseEstimationSeries : contains 0 or more
    PoseEstimation --> Skeleton : links to
    PoseEstimation --> Device : links to (device)
    PoseEstimation --> ImageSeries : links to (source_video)

    CalibratedCamera --|> Device : extends

    MultiCameraPoseEstimation --o PoseEstimationSeries : contains 0 or more
    MultiCameraPoseEstimation --o PoseEstimation : contains 0 or more
    MultiCameraPoseEstimation --> Skeleton : links to

    PoseTraining --o TrainingFrames : contains
    PoseTraining --o SourceVideos : contains

    TrainingFrames --o TrainingFrame : contains 0 or more

    TrainingFrame --o SkeletonInstances : contains
    TrainingFrame --> ImageSeries : links to
    TrainingFrame --> Image : links to

    SkeletonInstances --o SkeletonInstance : contains 0 or more
    SkeletonInstance --o Skeleton : links to

    SourceVideos --o ImageSeries : contains 0 or more

    Skeletons --o Skeleton : contains 0 or more
```

## Contributors

- @rly
- @bendichter
- @AlexEMG
- @roomrys
- @CBroz1
- @h-mayorquin
- @talmo
- @eberrigan
- @pauladkisson
- @alessandratrapani

This extension was created using [ndx-template](https://github.com/nwb-extensions/ndx-template).
