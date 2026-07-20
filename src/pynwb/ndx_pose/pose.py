import warnings
from hdmf.utils import docval, popargs, get_docval, AllowPositional
from pynwb import register_class, TimeSeries, get_class
from pynwb.behavior import SpatialSeries
from pynwb.core import MultiContainerInterface
from pynwb.device import Device
from pynwb.image import ImageSeries

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
        "device",
        "scorer",
        "source_software",
        "source_software_version",
        "nodes",
        "edges",
        "skeleton",  # <-- this is a link to a Skeleton object
        "source_video",  # <-- this is a link to an ImageSeries object
        "labeled_video",  # <-- this is a link to an ImageSeries object
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
            "doc": "Name of this PoseEstimation object.",
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
            "doc": (
                "Paths to the original video files. The number of files should equal the number of camera devices. "
                "Note: these string paths might be fragile unless relative paths are used and care is taken to "
                "keep them consistent. Consider using 'source_video' instead for a formal link to an ImageSeries."
            ),
            "default": None,
        },
        {
            "name": "labeled_videos",
            "type": ("array_data", "data"),
            "shape": (None,),
            "doc": (
                "Paths to the labeled video files. The number of files should equal the number of camera devices. "
                "Note: these string paths might be fragile unless relative paths are used and care is taken to "
                "keep them consistent. Consider using 'labeled_video' instead for a formal link to an ImageSeries."
            ),
            "default": None,
        },
        {
            "name": "dimensions",
            "type": ("array_data", "data"),
            "shape": (None, 2),
            "doc": (
                "Dimensions of each labeled video file. The number of dimension pairs should equal the number of "
                "camera devices."
            ),
            "default": None,
        },
        {
            "name": "device",
            "type": (Device, list, tuple),
            "doc": (
                "The camera device used to record the video for this pose estimation. Must be added to the "
                "NWBFile before being linked here. Use a CalibratedCamera instead of a plain Device when "
                "intrinsic/extrinsic calibration coordinates are available. NOTE: when reading a file written "
                "with ndx-pose < 0.4.0 that links more than one Device to a PoseEstimation object, HDMF resolves "
                "all of those links by type and passes them here as a list, which is handled the same way as "
                "the deprecated 'devices' argument below."
            ),
            "default": None,
        },
        {
            "name": "devices",
            "type": ("array_data", "data"),
            "doc": (
                "DEPRECATED. Please use the 'device' argument instead. PoseEstimation now represents pose "
                "estimates from a single camera view; for multi-camera setups, add one PoseEstimation per "
                "camera view to a MultiCameraPoseEstimation object."
            ),
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
            "name": "source_video",
            "type": ImageSeries,
            "doc": (
                "Link to an ImageSeries containing the source video used for pose estimation. "
                "The ImageSeries should be stored in the NWBFile (e.g., in acquisition) and linked here. "
                "When available, this field should be preferred over 'original_videos' as it provides "
                "a formal reference rather than a file path string."
            ),
            "default": None,
        },
        {
            "name": "labeled_video",
            "type": ImageSeries,
            "doc": (
                "Link to an ImageSeries containing the labeled video (with pose estimation overlays) "
                "produced from the source video. The ImageSeries should be stored in the NWBFile "
                "(e.g., in acquisition) and linked here. When available, this field should be preferred "
                "over 'labeled_videos' as it provides a formal reference rather than a file path string."
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
        nodes, edges, skeleton, source_video, labeled_video = popargs(
            "nodes", "edges", "skeleton", "source_video", "labeled_video", kwargs
        )
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

        # device must be added to the NWBFile before being linked to from a PoseEstimation object.
        # otherwise, it will be added as a child of the PoseEstimation object.
        device, devices = popargs("device", "devices", kwargs)
        if isinstance(device, (list, tuple)):
            # When reading a file written with ndx-pose < 0.4.0 that links more than one Device to a
            # PoseEstimation object, HDMF resolves all of the Device-typed links by type (regardless of the
            # current spec's quantity) and passes them here as a list instead of routing them through the
            # 'devices' constructor arg. Treat this exactly like the deprecated 'devices' argument.
            if devices is not None:
                raise ValueError("Cannot specify both 'device' and 'devices'. Please use 'device' only.")
            devices = device
            device = None
        if devices is not None:
            if device is not None:
                raise ValueError("Cannot specify both 'device' and 'devices'. Please use 'device' only.")
            if len(devices) > 1:
                raise ValueError(
                    "PoseEstimation now represents pose estimates from a single camera view and supports only one "
                    "device, but multiple Device objects are linked from this PoseEstimation. This is likely "
                    "because the file was written with ndx-pose < 0.4.0, when a PoseEstimation object could link "
                    "to multiple cameras. Reading files with more than one camera linked to a single "
                    "PoseEstimation object is not supported; each camera view now needs its own PoseEstimation "
                    "object inside a MultiCameraPoseEstimation object."
                )
            # warn on new, no warning on construction from existing file
            if not self._in_construct_mode:
                msg = (
                    "The 'devices' constructor argument is deprecated. Please use the 'device' argument instead. "
                    "This will be removed in a future release."
                )
                warnings.warn(msg, DeprecationWarning)
            device = devices[0] if len(devices) == 1 else None
        if device is not None and device.parent is None:
            raise ValueError("The device linked from a PoseEstimation object must be added to the NWBFile first.")

        original_videos, labeled_videos, dimensions = popargs("original_videos", "labeled_videos", "dimensions", kwargs)

        pose_estimation_series, description = popargs("pose_estimation_series", "description", kwargs)
        scorer = popargs("scorer", kwargs)
        source_software, source_software_version = popargs("source_software", "source_software_version", kwargs)
        if source_software_version is not None and source_software is None:
            raise ValueError(
                "'source_software_version' was specified without 'source_software'. The version is stored as an "
                "attribute on the 'source_software' dataset, so 'source_software' must be provided as well."
            )
        super().__init__(**kwargs)

        self.pose_estimation_series = pose_estimation_series
        self.description = description
        self.original_videos = original_videos
        self.labeled_videos = labeled_videos
        self.dimensions = dimensions
        self.device = device
        self.scorer = scorer
        self.source_software = source_software
        self.source_software_version = source_software_version
        self.skeleton = skeleton
        self.source_video = source_video
        self.labeled_video = labeled_video

        # TODO include calibration images for 3D estimates?
        # TODO validate that the nodes correspond to the names of the pose estimation series objects

    @property
    def nodes(self):
        if self.skeleton is None:
            raise ValueError(
                "This PoseEstimation object has no Skeleton, so it has no nodes. Provide a 'skeleton' argument "
                "to access nodes via PoseEstimation.skeleton.nodes."
            )
        return self.skeleton.nodes

    @nodes.setter
    def nodes(self, value):
        raise ValueError(
            "Setting PoseEstimation.nodes is deprecated. Please use PoseEstimation.skeleton.nodes instead."
        )

    @property
    def edges(self):
        if self.skeleton is None:
            raise ValueError(
                "This PoseEstimation object has no Skeleton, so it has no edges. Provide a 'skeleton' argument "
                "to access edges via PoseEstimation.skeleton.edges."
            )
        return self.skeleton.edges

    @edges.setter
    def edges(self, value):
        raise ValueError(
            "Setting PoseEstimation.edges is deprecated. Please use PoseEstimation.skeleton.edges instead."
        )

    @property
    def devices(self):
        warnings.warn(
            "PoseEstimation.devices is deprecated. Please use PoseEstimation.device instead.",
            DeprecationWarning,
        )
        return [self.device] if self.device is not None else []

    @devices.setter
    def devices(self, value):
        raise ValueError(
            "Setting PoseEstimation.devices is deprecated. Please use PoseEstimation.device instead."
        )


@register_class("CalibratedCamera", "ndx-pose")
class CalibratedCamera(Device):
    """A Device representing a single camera, extended with its calibration parameters.

    Because it is a Device, a CalibratedCamera is added once to the NWBFile and can be linked to by
    reference from multiple PoseEstimation and MultiCameraPoseEstimation objects (e.g., one per subject
    in a multi-subject recording session), so the camera rig and its calibration are never duplicated.
    """

    __nwbfields__ = (
        "intrinsic_matrix",
        "rotation_matrix",
        "translation_vector",
        "distortion_coefficients",
    )

    @docval(
        {
            "name": "name",
            "type": str,
            "doc": "Name of this CalibratedCamera, typically the camera identifier (e.g. 'camera1').",
        },
        {
            "name": "intrinsic_matrix",
            "type": ("array_data", "data"),
            "shape": (3, 3),
            "doc": "Intrinsic camera matrix K, encoding focal length and principal point. Shape (3, 3).",
        },
        {
            "name": "rotation_matrix",
            "type": ("array_data", "data"),
            "shape": (3, 3),
            "doc": "Rotation matrix R mapping world coordinates to this camera's coordinate frame. Shape (3, 3).",
            "default": None,
        },
        {
            "name": "translation_vector",
            "type": ("array_data", "data"),
            "shape": (3,),
            "doc": "Translation vector t mapping world coordinates to this camera's coordinate frame. Shape (3,).",
            "default": None,
        },
        {
            "name": "distortion_coefficients",
            "type": ("array_data", "data"),
            "shape": (None,),
            "doc": "Lens distortion coefficients for this camera.",
            "default": None,
        },
        *get_docval(Device.__init__, "description", "manufacturer"),
        allow_positional=AllowPositional.ERROR,
    )
    def __init__(self, **kwargs):
        intrinsic_matrix, rotation_matrix, translation_vector, distortion_coefficients = popargs(
            "intrinsic_matrix", "rotation_matrix", "translation_vector", "distortion_coefficients", kwargs
        )
        super().__init__(**kwargs)

        self.intrinsic_matrix = intrinsic_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.distortion_coefficients = distortion_coefficients


@register_class("MultiCameraPoseEstimation", "ndx-pose")
class MultiCameraPoseEstimation(MultiContainerInterface):
    """3D pose estimation from multiple synchronized cameras.

    Unlike PoseEstimation (single-camera, pixel-space), this type stores keypoints in a shared 3D
    world-space reference frame. Per-camera 2D data (device link, source video, optional 2D estimates)
    is organised through PoseEstimation children, one per camera view.
    """

    __clsconf__ = [
        {
            "add": "add_pose_estimation_series",
            "get": "get_pose_estimation_series",
            "create": "create_pose_estimation_series",
            "type": PoseEstimationSeries,
            "attr": "pose_estimation_series",
        },
        {
            "add": "add_pose_estimation",
            "get": "get_pose_estimation",
            "create": "create_pose_estimation",
            "type": PoseEstimation,
            "attr": "pose_estimations",
        },
    ]

    __nwbfields__ = (
        "description",
        "scorer",
        "source_software",
        "source_software_version",
        "skeleton",
    )

    @docval(
        {
            "name": "name",
            "type": str,
            "doc": "Name of this MultiCameraPoseEstimation container.",
            "default": "MultiCameraPoseEstimation",
        },
        {
            "name": "pose_estimation_series",
            "type": ("array_data", "data"),
            "doc": "3D pose estimates (x, y, z) for each body part in world-space coordinates.",
            "default": None,
        },
        {
            "name": "pose_estimations",
            "type": ("array_data", "data"),
            "doc": (
                "PoseEstimation objects, one per camera view. Each PoseEstimation links to the camera Device "
                "(ideally a CalibratedCamera) used to record that view."
            ),
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "Description of the pose estimation procedure and output.",
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
            "doc": "Name of the software tool used. Specifying the version is strongly encouraged.",
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
                "Skeletons object which resides in the NWBFile at the same level as this container."
            ),
            "default": None,
        },
        allow_positional=AllowPositional.ERROR,
    )
    def __init__(self, **kwargs):
        skeleton = popargs("skeleton", kwargs)
        pose_estimation_series, pose_estimations = popargs("pose_estimation_series", "pose_estimations", kwargs)
        description, scorer = popargs("description", "scorer", kwargs)
        source_software, source_software_version = popargs("source_software", "source_software_version", kwargs)
        super().__init__(**kwargs)

        self.pose_estimation_series = pose_estimation_series
        self.pose_estimations = pose_estimations
        self.description = description
        self.scorer = scorer
        self.source_software = source_software
        self.source_software_version = source_software_version
        self.skeleton = skeleton
