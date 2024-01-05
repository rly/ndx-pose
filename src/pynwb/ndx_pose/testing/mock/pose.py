from typing import Optional, Any

import numpy as np
from pynwb.image import ImageSeries
from pynwb.testing.mock.utils import name_generator

from ...pose import PoseEstimationSeries, Skeleton, SkeletonInstance, TrainingFrame


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
        confidence = np.linspace(
            0, 1, num=len(data)
        )  # confidence value for every frame
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


def mock_SkeletonInstance(
    *,
    id: Optional[np.uint] = np.uint(10),
    node_locations: Optional[Any] = None,
    node_visibility: list = None,
    skeleton: Skeleton = None,
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
        node_locations = np.arange(num_nodes * 2).reshape((num_nodes, 2))
    if node_visibility is None:
        node_visibility = np.ones(num_nodes, dtype="bool")
    skeleton_instance = SkeletonInstance(
        id=id,
        node_locations=node_locations,
        node_visibility=node_visibility,
        skeleton=skeleton,
    )
    return skeleton_instance


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


def mock_TrainingFrame(
    *,
    name: Optional[str] = None,
    annotator: Optional[str] = "Awesome Possum",
    skeleton_instance: SkeletonInstance = None,
    source_video: ImageSeries = None,
    source_video_frame_index: np.uint = np.uint(10),
):
    training_frame = TrainingFrame(
        name=name or name_generator("TrainingFrame"),
        annotator=annotator,
        skeleton_instance=skeleton_instance or mock_SkeletonInstance(),
        source_video=source_video or mock_source_video(),
        source_video_frame_index=source_video_frame_index,
    )
    return training_frame
