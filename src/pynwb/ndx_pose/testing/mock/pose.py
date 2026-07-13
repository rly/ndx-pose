from typing import Optional, Any, Union, List

import numpy as np
from pynwb import NWBFile
from pynwb.device import Device
from pynwb.image import ImageSeries, Image, RGBImage
from pynwb.testing.mock.utils import name_generator
from pynwb.testing.mock.device import mock_Device

from ...pose import (
    CalibratedCamera,
    MultiCameraPoseEstimation,
    PoseEstimationSeries,
    Skeleton,
    PoseEstimation,
    SkeletonInstance,
    SkeletonInstances,
    TrainingFrame,
    Skeletons,
    SourceVideos,
)


def mock_PoseEstimationSeries(
    *,
    name: Optional[str] = None,
    description: Optional[str] = "A description.",
    data=None,
    unit: Optional[str] = "pixels",
    resolution: float = -1.0,
    conversion: float = 1.0,
    offset: float = 0.0,
    timestamps=None,
    starting_time: Optional[float] = None,
    rate: Optional[float] = None,
    reference_frame: Optional[str] = "(0,0,0) corresponds to ...",
    confidence=None,
    confidence_definition: Optional[str] = "Softmax output of the deep neural network.",
):
    if data is None:
        data = np.arange(30, dtype=np.float64).reshape((10, 3))
    if timestamps is not None:
        rate = None
    if timestamps is None and rate is None:
        timestamps = np.linspace(0, 10, num=len(data))  # a timestamp for every frame
    if confidence is None:
        confidence = np.linspace(0, 1, num=len(data))  # confidence value for every frame
    pes = PoseEstimationSeries(
        name=name or name_generator("PoseEstimationSeries"),
        data=data,
        unit=unit,
        resolution=resolution,
        conversion=conversion,
        offset=offset,
        timestamps=timestamps,
        starting_time=starting_time,
        rate=rate,
        reference_frame=reference_frame,
        description=description,
        confidence=confidence,
        confidence_definition=confidence_definition,
    )

    return pes


def mock_Skeleton(
    *,
    name: Optional[str] = None,
    nodes: Optional[Any] = None,
    edges: Optional[Any] = None,
):
    if nodes is None and edges is None:
        nodes = ["node1", "node2", "node3"]
        edges = np.array([(0, 1), (1, 2)], dtype="uint8")
    skeleton = Skeleton(
        name=name or name_generator("Skeleton"),
        nodes=nodes,
        edges=edges,
    )

    return skeleton


def mock_PoseEstimation(
    *,
    nwbfile: NWBFile,
    name: Optional[str] = None,
    pose_estimation_series: Optional[list] = None,
    skeleton: Optional[Skeleton] = None,
    device: Optional[Device] = None,
    description: Optional[str] = "Estimated positions of front paws using DeepLabCut.",
    original_videos: Optional[list] = None,
    labeled_videos: Optional[list] = None,
    dimensions: Optional[np.ndarray] = None,
    scorer: Optional[str] = "DLC_resnet50_openfieldOct30shuffle1_1600",
    source_software: Optional[str] = "DeepLabCut",
    source_software_version: Optional[str] = "2.2b8",
    source_video: Optional[ImageSeries] = None,
    labeled_video: Optional[ImageSeries] = None,
    add_to_nwbfile: bool = True,
):
    """Create a mock PoseEstimation object, representing pose estimates from a single camera view.

    NWBFile should be provided so that the device and skeleton can be added to the NWBFile.
    """
    skeleton = skeleton or mock_Skeleton()
    pose_estimation_series = pose_estimation_series or [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]
    pe = PoseEstimation(
        name=name or "PoseEstimation",
        pose_estimation_series=pose_estimation_series,
        description=description,
        original_videos=original_videos,
        labeled_videos=labeled_videos,
        dimensions=dimensions,
        device=device or mock_Device(nwbfile=nwbfile),
        scorer=scorer,
        source_software=source_software,
        source_software_version=source_software_version,
        skeleton=skeleton,
        source_video=source_video,
        labeled_video=labeled_video,
    )

    if add_to_nwbfile:
        skeletons = Skeletons(skeletons=[skeleton])
        if "behavior" not in nwbfile.processing:
            behavior_pm = nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        else:
            behavior_pm = nwbfile.processing["behavior"]
        behavior_pm.add(pe)
        behavior_pm.add(skeletons)

    return pe


def mock_CalibratedCamera(
    *,
    nwbfile: NWBFile,
    name: Optional[str] = None,
) -> "CalibratedCamera":
    """Create a mock CalibratedCamera with identity-like calibration matrices, added to the NWBFile."""
    rng = np.random.default_rng(0)
    intrinsic_matrix = np.eye(3, dtype="float32")
    intrinsic_matrix[0, 0] = 800.0  # fx
    intrinsic_matrix[1, 1] = 800.0  # fy
    intrinsic_matrix[0, 2] = 320.0  # cx
    intrinsic_matrix[1, 2] = 240.0  # cy
    camera = CalibratedCamera(
        name=name or name_generator("CalibratedCamera"),
        intrinsic_matrix=intrinsic_matrix,
        rotation_matrix=np.eye(3, dtype="float32"),
        translation_vector=rng.standard_normal(3).astype("float32"),
        distortion_coefficients=np.zeros(5, dtype="float32"),
    )
    nwbfile.add_device(camera)
    return camera


def mock_MultiCameraPoseEstimation(
    *,
    nwbfile: NWBFile,
    pose_estimation_series: Optional[list] = None,
    pose_estimations: Optional[list] = None,
    skeleton: Optional[Skeleton] = None,
    description: Optional[str] = "3D pose estimated from multiple synchronized cameras.",
    scorer: Optional[str] = "DANNCE",
    source_software: Optional[str] = "DANNCE",
    source_software_version: Optional[str] = "2.0.0",
) -> "MultiCameraPoseEstimation":
    """Create a mock MultiCameraPoseEstimation.  NWBFile must be provided."""
    skeleton = skeleton or mock_Skeleton()

    pose_estimation_series = pose_estimation_series or [
        mock_PoseEstimationSeries(
            name=node,
            data=np.arange(30, dtype=np.float64).reshape((10, 3)),
            unit="millimeters",
            reference_frame="(0,0,0) is the midpoint of the camera rig.",
        )
        for node in skeleton.nodes
    ]

    if pose_estimations is None:
        pose_estimations = [
            mock_PoseEstimation(
                nwbfile=nwbfile,
                name=f"PoseEstimation_camera{i + 1}",
                skeleton=skeleton,
                device=mock_CalibratedCamera(nwbfile=nwbfile, name=f"camera{i + 1}"),
                description=f"2D pose estimates from camera{i + 1}.",
                add_to_nwbfile=False,
            )
            for i in range(2)
        ]

    mcpe = MultiCameraPoseEstimation(
        pose_estimation_series=pose_estimation_series,
        pose_estimations=pose_estimations,
        description=description,
        scorer=scorer,
        source_software=source_software,
        source_software_version=source_software_version,
        skeleton=skeleton,
    )

    skeletons = Skeletons(skeletons=[skeleton])
    if "behavior" not in nwbfile.processing:
        behavior_pm = nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
    else:
        behavior_pm = nwbfile.processing["behavior"]
    behavior_pm.add(mcpe)
    behavior_pm.add(skeletons)

    return mcpe


def mock_SkeletonInstance(
    *,
    name: Optional[str] = None,
    id: Optional[np.uint] = np.uint(10),
    node_locations: Optional[Any] = None,
    node_visibility: Optional[list] = None,
    skeleton: Optional[Skeleton] = None,
):
    if node_locations is None and node_visibility is None:
        num_nodes = 3
    elif node_locations is not None:
        num_nodes = len(node_locations)
    else:
        num_nodes = len(node_visibility)

    if skeleton is None:
        skeleton = mock_Skeleton(
            nodes=["node" + str(i) for i in range(num_nodes)],
            edges=np.array([[0, 1]], dtype="uint8"),
        )
    if node_locations is None:
        node_locations = np.arange(num_nodes * 2, dtype=np.float64).reshape((num_nodes, 2))

    if name is None:
        name = skeleton.name + "_instance_" + str(id)
    if node_visibility is None:
        node_visibility = np.ones(num_nodes, dtype="bool")

    skeleton_instance = SkeletonInstance(
        name=name,
        id=id,
        node_locations=node_locations,
        node_visibility=node_visibility,
        skeleton=skeleton,
    )

    return skeleton_instance


def mock_SkeletonInstances(skeleton_instances: Union[SkeletonInstance, List[SkeletonInstance]] = None):
    if skeleton_instances is None:
        skeleton_instances = [mock_SkeletonInstance()]
    if not isinstance(skeleton_instances, list):
        skeleton_instances = [skeleton_instances]
    return SkeletonInstances(
        skeleton_instances=skeleton_instances,
    )


def mock_source_video(
    *,
    name: Optional[str] = None,
):
    source_video = ImageSeries(
        name=name or name_generator("ImageSeries"),
        description="Training video used to train the pose estimation model.",
        unit="NA",
        format="external",
        external_file=["path/to/camera1.mp4"],
        dimension=[640, 480],
        starting_frame=[0],
        rate=30.0,
    )
    return source_video


def mock_source_frame(
    *,
    name: Optional[str] = None,
):
    return RGBImage(name=name or name_generator("RGBImage"), data=np.random.rand(640, 480, 3).astype("uint8"))


def mock_TrainingFrame(
    *,
    name: Optional[str] = None,
    annotator: Optional[str] = "Awesome Possum",
    skeleton_instances: Optional[SkeletonInstances] = None,
    source_video: Optional[ImageSeries] = None,
    source_frame: Optional[Image] = None,
    source_video_frame_index: Optional[np.uint] = np.uint(10),
):
    training_frame = TrainingFrame(
        name=name or name_generator("TrainingFrame"),
        annotator=annotator,
        skeleton_instances=skeleton_instances or mock_SkeletonInstances(),
        source_video=source_video or (mock_source_video() if source_frame is None else None),
        source_frame=source_frame,
        source_video_frame_index=source_video_frame_index,
    )
    return training_frame
