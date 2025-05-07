import warnings
from hdmf.utils import docval, popargs, get_docval, AllowPositional
from pynwb import register_class, TimeSeries, get_class
from pynwb.behavior import SpatialSeries
from pynwb.core import MultiContainerInterface

# TODO validate Skeleton nodes and edges correspondence, convert edges to uint
# TODO validate that all Skeleton nodes are used in edges
Skeleton = get_class("Skeleton", "ndx-pose")
Skeletons = get_class("Skeletons", "ndx-pose")
SkeletonInstance = get_class("SkeletonInstance", "ndx-pose")
SkeletonInstances = get_class("SkeletonInstances", "ndx-pose")
TrainingFrame = get_class("TrainingFrame", "ndx-pose")
TrainingFrames = get_class("TrainingFrames", "ndx-pose")
SourceVideos = get_class("SourceVideos", "ndx-pose")
PoseTraining = get_class("PoseTraining", "ndx-pose")


@register_class("PoseEstimationSeries", "ndx-pose")
class PoseEstimationSeries(SpatialSeries):
    """Estimated position (x, y) or (x, y, z) of a body part over time."""

    __nwbfields__ = ("confidence", "confidence_definition")

    # NOTE: custom mapper in ndx_pose.io.pose maps:
    # 'confidence' dataset -> 'definition' attribute in spec to 'confidence_definition' field in Python class
    # if not for the custom mapper, this class could be auto-generated from the spec

    @docval(
        {
            "name": "name",
            "type": str,
            "doc": "Name of this PoseEstimationSeries, usually the name of a body part.",
        },
        {
            "name": "data",
            "type": ("array_data", "data", TimeSeries),
            "shape": ((None, 2), (None, 3)),
            "doc": "Estimated position (x, y) or (x, y, z).",
        },
        {
            "name": "reference_frame",
            "type": str,
            "doc": "Description defining what the zero-position (0, 0) or (0, 0, 0) is.",
        },
        {
            "name": "confidence",
            "type": ("array_data", "data"),
            "shape": (None,),
            "doc": "Confidence or likelihood of the estimated positions, scaled to be between 0 and 1.",
            "default": None,
        },
        {
            "name": "unit",
            "type": str,
            "doc": (
                "Base unit of measurement for working with the data. The default value "
                "is 'pixels'. Actual stored values are not necessarily stored in these units. "
                "To access the data in these units, multiply 'data' by 'conversion'."
            ),
            "default": "pixels",
        },
        {
            "name": "confidence_definition",
            "type": str,
            "doc": "Description of how the confidence was computed, e.g., 'Softmax output of the deep neural network'.",
            "default": None,
        },
        *get_docval(
            TimeSeries.__init__,
            "conversion",
            "resolution",
            "offset",
            "timestamps",
            "starting_time",
            "rate",
            "comments",
            "description",
            "control",
            "control_description",
        ),
        allow_positional=AllowPositional.ERROR,
    )
    def __init__(self, **kwargs):
        """Construct a new PoseEstimationSeries representing pose estimates for a particular body part."""
        confidence, confidence_definition = popargs("confidence", "confidence_definition", kwargs)
        super().__init__(**kwargs)
        self.confidence = confidence
        self.confidence_definition = confidence_definition


@register_class("PoseEstimation", "ndx-pose")
# NOTE: NWB MultiContainerInterface extends NWBDataInterface and HDMF MultiContainerInterface
class PoseEstimation(MultiContainerInterface):
    """Estimated position data for multiple body parts, computed from the same video with the same tool/algorithm.
    The timestamps of each child PoseEstimationSeries type should be the same.
    """

    __clsconf__ = [
        {
            "add": "add_pose_estimation_series",
            "get": "get_pose_estimation_series",
            "create": "create_pose_estimation_series",
            "type": PoseEstimationSeries,
            "attr": "pose_estimation_series",
        },
        # NOTE: devices is a list of **linked** Device objects. Because they are linked, we do not set up
        # MultiContainerInterface-generated functions for devices.
    ]

    __nwbfields__ = (
        "description",
        "original_videos",
        "labeled_videos",
        "dimensions",
        "devices",
        "scorer",
        "source_software",
        "source_software_version",
        "nodes",
        "edges",
        "skeleton",  # <-- this is a link to a Skeleton object
    )

    # custom mapper in ndx_pose.io.pose maps:
    # 'source_software' dataset -> 'version' attribute to 'source_software_version' field
    # if not for the custom mapper and custom validation, this class could be auto-generated from the spec

    @docval(  # all fields are optional
        {
            "name": "pose_estimation_series",
            "type": ("array_data", "data"),
            "doc": "Estimated position data for each body part.",
            "default": None,
        },
        {
            "name": "name",
            "type": str,
            "doc": "Description of the pose estimation procedure and output.",
            "default": "PoseEstimation",
        },
        {
            "name": "description",
            "type": str,
            "doc": "Description of the pose estimation procedure and output.",
            "default": None,
        },
        {
            "name": "original_videos",
            "type": ("array_data", "data"),
            "shape": (None,),
            "doc": "Paths to the original video files. The number of files should equal the number of camera devices.",
            "default": None,
        },
        {
            "name": "labeled_videos",
            "type": ("array_data", "data"),
            "shape": (None,),
            "doc": "Paths to the labeled video files. The number of files should equal the number of camera devices.",
            "default": None,
        },
        {
            "name": "dimensions",
            "type": ("array_data", "data"),
            "shape": ((None, 2)),
            "doc": (
                "Dimensions of each labeled video file. The number of dimension pairs should equal the number of "
                "camera devices."
            ),
            "default": None,
        },
        {
            "name": "devices",
            "type": ("array_data", "data"),
            "doc": "Cameras used to record the videos.",
            "default": None,
        },
        {
            "name": "scorer",
            "type": str,
            "doc": "Name of the scorer / algorithm used.",
            "default": None,
        },
        {
            "name": "source_software",
            "type": str,
            "doc": "Name of the software tool used. Specifying the version attribute is strongly encouraged.",
            "default": None,
        },
        {
            "name": "source_software_version",
            "type": str,
            "doc": "Version string of the software tool used.",
            "default": None,
        },
        {
            "name": "skeleton",
            "type": Skeleton,
            "doc": (
                "Layout of body part locations and connections. The Skeleton object should be placed in a "
                "Skeletons object which resides in the NWBFile at the same level as the PoseEstimation object. "
                "The Skeleton object should be linked here."
            ),
            "default": None,
        },
        {
            "name": "nodes",
            "type": ("array_data", "data"),
            "doc": (
                "DEPRECATED. Please use the 'skeleton' argument instead. "
                "Array of body part names corresponding to the names of the PoseEstimationSeries objects within "
                "this container."
            ),
            "default": None,
        },
        {
            "name": "edges",
            "type": ("array_data", "data"),
            "doc": (
                "DEPRECATED. Please use the 'skeleton' argument instead. "
                "Array of pairs of indices corresponding to edges between nodes. Index values correspond to row "
                "indices of the 'nodes' field. Index values use 0-indexing."
            ),
            "default": None,
        },
        allow_positional=AllowPositional.ERROR,
    )
    def __init__(self, **kwargs):
        nodes, edges, skeleton = popargs("nodes", "edges", "skeleton", kwargs)
        if nodes is not None or edges is not None:
            if skeleton is not None:
                raise ValueError("Cannot specify 'skeleton' with 'nodes' or 'edges'.")
            # TODO: this Skeleton is normally a link to a Skeleton elsewhere in the file (e.g., in a Skeletons object)
            # Here, the Skeleton is constructed from the nodes and edges and exists only in this PoseEstimation object
            # and not as a child; this can have unintended consequences if the file is rewritten with the latest
            # schema. This is a limitation of the current implementation, and will be addressed in a future release.
            skeleton = Skeleton(name="subject", nodes=nodes, edges=edges)
            # warn on new, no warning on construction from existing file
            if not self._in_construct_mode:
                msg = (
                    "The 'nodes' and 'edges' constructor arguments are deprecated. Please use the 'skeleton' "
                    "argument instead. These will be removed in a future release."
                )
                warnings.warn(msg, DeprecationWarning)

        # devices must be added to the NWBFile before being linked to from a PoseEstimation object.
        # otherwise, they will be added as children of the PoseEstimation object.
        devices = popargs("devices", kwargs)
        if devices is not None:
            for device in devices:
                if device.parent is None:
                    raise ValueError(
                        "All devices linked to from a PoseEstimation object must be added to the NWBFile first."
                    )

        # validate that if original videos, labeled videos, or dimensions are provided, then an equal number of
        # camera devices must be provided.
        # specification of cameras was not allowed in ndx-pose 0.1.*.
        # warn on new, no warning on construction from existing file
        # TODO in a future release, change DeprecationWarning to a ValueError
        original_videos, labeled_videos, dimensions = popargs("original_videos", "labeled_videos", "dimensions", kwargs)
        if original_videos is not None and (devices is None or len(original_videos) != len(devices)):
            if not self._in_construct_mode:
                msg = (
                    "The number of original videos must equal the number of camera devices. This will become an error "
                    "in a future release."
                )
                warnings.warn(msg, DeprecationWarning)
        if labeled_videos is not None and (devices is None or len(labeled_videos) != len(devices)):
            if not self._in_construct_mode:
                msg = (
                    "The number of labeled videos must equal the number of camera devices. This will become an error "
                    "in a future release."
                )
                warnings.warn(msg, DeprecationWarning)
        if dimensions is not None and (devices is None or len(dimensions) != len(devices)):
            if not self._in_construct_mode:
                msg = (
                    "The number of dimensions must equal the number of camera devices. This will become an error in a "
                    "future release."
                )
                warnings.warn(msg, DeprecationWarning)

        pose_estimation_series, description = popargs("pose_estimation_series", "description", kwargs)
        scorer = popargs("scorer", kwargs)
        source_software, source_software_version = popargs("source_software", "source_software_version", kwargs)
        super().__init__(**kwargs)

        self.pose_estimation_series = pose_estimation_series
        self.description = description
        self.original_videos = original_videos
        self.labeled_videos = labeled_videos
        self.dimensions = dimensions
        self.devices = devices
        self.scorer = scorer
        self.source_software = source_software
        self.source_software_version = source_software_version
        self.skeleton = skeleton

        # TODO include calibration images for 3D estimates?
        # TODO validate that the nodes correspond to the names of the pose estimation series objects

    @property
    def nodes(self):
        return self.skeleton.nodes

    @nodes.setter
    def nodes(self, value):
        raise ValueError(
            "Setting PoseEstimation.nodes is deprecated. Please use PoseEstimation.skeleton.nodes instead."
        )

    @property
    def edges(self):
        return self.skeleton.edges

    @edges.setter
    def edges(self, value):
        raise ValueError(
            "Setting PoseEstimation.edges is deprecated. Please use PoseEstimation.skeleton.edges instead."
        )
