from typing import Optional, Any, Union, List

import numpy as np
from pynwb import NWBFile
from pynwb.image import ImageSeries, Image, RGBImage
from pynwb.testing.mock.utils import name_generator
from pynwb.testing.mock.device import mock_Device

from ...pose import (
    CameraCalibration,
    CameraView,
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
    pose_estimation_series: Optional[list] = None,
    skeleton: Optional[Skeleton] = None,
    devices: Optional[list] = None,
    description: Optional[str] = "Estimated positions of front paws using DeepLabCut.",
    original_videos: Optional[list] = None,
    labeled_videos: Optional[list] = None,
    dimensions: Optional[np.ndarray] = None,
    scorer: Optional[str] = "DLC_resnet50_openfieldOct30shuffle1_1600",
    source_software: Optional[str] = "DeepLabCut",
    source_software_version: Optional[str] = "2.2b8",
    source_video: Optional[ImageSeries] = None,
    labeled_video: Optional[ImageSeries] = None,
):
    """Create a mock PoseEstimation object.

    NWBFile should be provided so that the skeleton can be added to the NWBFile in a PoseTraining object.
    """
    skeleton = skeleton or mock_Skeleton()
    pose_estimation_series = pose_estimation_series or [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]
    pe = PoseEstimation(
        pose_estimation_series=pose_estimation_series,
        description=description,
        original_videos=original_videos or ["camera1.mp4"],
        labeled_videos=labeled_videos or ["camera1_labeled.mp4"],
        dimensions=dimensions or np.array([[640, 480]], dtype="uint16"),
        devices=devices or [mock_Device(nwbfile=nwbfile)],
        scorer=scorer,
        source_software=source_software,
        source_software_version=source_software_version,
        skeleton=skeleton,
        source_video=source_video,
        labeled_video=labeled_video,
    )

    if nwbfile is not None:
        skeletons = Skeletons(skeletons=[skeleton])

        if "behavior" not in nwbfile.processing:
            behavior_pm = nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        else:
            behavior_pm = nwbfile.processing["behavior"]
        behavior_pm.add(pe)
        behavior_pm.add(skeletons)

    return pe


def mock_CameraCalibration(
    *,
    nwbfile: NWBFile,
    devices: Optional[list] = None,
    n_cameras: int = 2,
) -> "CameraCalibration":
    """Create a mock CameraCalibration with identity-like matrices."""
    rng = np.random.default_rng(0)
    if devices is None:
        devices = [mock_Device(nwbfile=nwbfile, name=f"camera{i + 1}") for i in range(n_cameras)]
    n = len(devices)
    intrinsic = np.tile(np.eye(3, dtype="float32"), (n, 1, 1))
    intrinsic[:, 0, 0] = 800.0  # fx
    intrinsic[:, 1, 1] = 800.0  # fy
    intrinsic[:, 0, 2] = 320.0  # cx
    intrinsic[:, 1, 2] = 240.0  # cy
    return CameraCalibration(
        intrinsic_matrix=intrinsic,
        rotation_matrix=np.tile(np.eye(3, dtype="float32"), (n, 1, 1)),
        translation_vector=rng.standard_normal((n, 3)).astype("float32"),
        distortion_coefficients=np.zeros((n, 5), dtype="float32"),
        devices=devices,
    )


def mock_CameraView(
    *,
    nwbfile: NWBFile,
    name: Optional[str] = None,
    device=None,
    source_video: Optional[ImageSeries] = None,
    with_2d_estimates: bool = False,
    skeleton: Optional[Skeleton] = None,
) -> "CameraView":
    """Create a mock CameraView, optionally adding 2D pixel-space estimates."""
    if device is None:
        device = mock_Device(nwbfile=nwbfile, name=name or "camera1")
    if source_video is None:
        cam_name = device.name
        source_video = ImageSeries(
            name=cam_name,
            description=f"Source video from {cam_name}.",
            unit="NA",
            format="external",
            external_file=[f"{cam_name}.mp4"],
            rate=30.0,
        )
        nwbfile.add_acquisition(source_video)

    pes_2d = None
    if with_2d_estimates and skeleton is not None:
        pes_2d = [
            mock_PoseEstimationSeries(
                name=node,
                data=np.arange(20, dtype=np.float64).reshape((10, 2)),
                unit="pixels",
                reference_frame="top-left corner of the frame is (0, 0).",
            )
            for node in skeleton.nodes
        ]

    return CameraView(
        name=name or device.name,
        device=device,
        source_video=source_video,
        pose_estimation_series=pes_2d,
    )


def mock_MultiCameraPoseEstimation(
    *,
    nwbfile: NWBFile,
    pose_estimation_series: Optional[list] = None,
    camera_views: Optional[list] = None,
    skeleton: Optional[Skeleton] = None,
    calibration=None,
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

    if camera_views is None:
        device1 = mock_Device(nwbfile=nwbfile, name="camera1")
        device2 = mock_Device(nwbfile=nwbfile, name="camera2")
        camera_views = [
            mock_CameraView(nwbfile=nwbfile, name="camera1", device=device1),
            mock_CameraView(nwbfile=nwbfile, name="camera2", device=device2),
        ]

    mcpe = MultiCameraPoseEstimation(
        pose_estimation_series=pose_estimation_series,
        camera_views=camera_views,
        description=description,
        scorer=scorer,
        source_software=source_software,
        source_software_version=source_software_version,
        skeleton=skeleton,
        calibration=calibration,
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
    return RGBImage(name=name, data=np.random.rand(640, 480, 3).astype("uint8"))


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
