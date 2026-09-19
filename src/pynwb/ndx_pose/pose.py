import warnings
from hdmf.utils import docval, popargs, get_docval, get_data_shape, AllowPositional
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

def _shapes_agree(shape_a, shape_b):
    """Return whether two shapes match, treating a dimension of unknown length as a match.

    hdmf's get_data_shape reports None for a dimension whose length it cannot determine without
    consuming an iterator, so an unknown dimension is not treated as a mismatch.
    """
    if len(shape_a) != len(shape_b):
        return False
    return all(a is None or b is None or a == b for a, b in zip(shape_a, shape_b))


def _max_value(values):
    """Return the largest value in an in-memory array, or None when it is not one.

    Only an array that is already in memory is scanned. Finding the maximum of an h5py
    dataset or a chunk iterator would read the whole array, which for a long recording is
    the largest thing in the file. numpy arrays expose 'max'; those lazy objects do not.
    """
    if isinstance(values, (list, tuple)):
        maxima = [max(row) for row in values if len(row)]
        return max(maxima) if maxima else None
    if not hasattr(values, "max"):
        return None
    try:
        return values.max()
    except ValueError:  # an empty array has no maximum
        return None


@register_class("ContourSeries", "ndx-pose")
class ContourSeries(TimeSeries):
    """Polygon contours outlining a segmented instance over time.

    Each frame holds a fixed number of contour slots. ``vertex_count`` gives the number of valid
    vertices in each slot, so trailing slots and trailing vertices in ``data`` are unused padding
    and carry no meaning. More than one contour may be needed to describe an instance on a frame:
    an outer boundary plus one or more holes, or a body that an occluder splits into disjoint parts.
    ``is_external`` tells the two apart, without which the polygons cannot be rendered or measured,
    so it is required. ``contour_group``, when known, records which component each contour belongs
    to, so a hole stays attached to the part of a split instance that contains it.

    Vertex positions are in the frame of reference described by ``reference_frame``. This type extends
    TimeSeries rather than SpatialSeries, whose ``data`` may have at most two dimensions, so it declares
    ``reference_frame`` itself instead of inheriting it.

    Store this inside a PoseEstimation object to associate the contours with the pose estimates and
    subject for the same instance.
    """

    __nwbfields__ = ("reference_frame", "vertex_count", "is_external", "contour_group")

    @docval(
        {
            "name": "name",
            "type": str,
            "doc": "Name of this ContourSeries.",
        },
        {
            "name": "data",
            "type": ("array_data", "data", TimeSeries),
            "shape": (None, None, None, 2),
            "doc": (
                "Contour vertex positions (x, y), with shape "
                "(num_frames, num_contours, num_vertices, 2). Only the first 'vertex_count' "
                "vertices of each contour slot hold a position; the rest are padding."
            ),
        },
        {
            "name": "reference_frame",
            "type": str,
            "doc": (
                "Description defining what the zero-position (0, 0) of the vertex coordinates is and "
                "which way each axis increases. Give the same frame of reference as the "
                "PoseEstimationSeries objects describing the same instance."
            ),
        },
        {
            "name": "vertex_count",
            "type": ("array_data", "data"),
            "shape": (None, None),
            "doc": (
                "Number of valid vertices in each contour slot, with shape "
                "(num_frames, num_contours). 0 means the slot holds no contour on that frame. "
                "No count may exceed the number of vertices each slot of 'data' holds."
            ),
        },
        {
            "name": "is_external",
            "type": ("array_data", "data"),
            "shape": (None, None),
            "doc": (
                "True where the contour slot is an external boundary, i.e. an outer edge of the "
                "instance, and False where it is an internal boundary, i.e. a hole. Has no meaning "
                "where 'vertex_count' is 0. A producer that retrieves only outer boundaries sets "
                "this True throughout."
            ),
        },
        {
            "name": "contour_group",
            "type": ("array_data", "data"),
            "shape": (None, None),
            "doc": (
                "Index grouping contours into connected components within a frame. Contours "
                "describing the same component share a value, and a hole carries the value of the "
                "component that contains it. Has no meaning where 'vertex_count' is 0. Omit when "
                "the component structure is not known."
            ),
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
        """Construct a new ContourSeries representing the outline of a segmented instance over time."""
        reference_frame, vertex_count, is_external, contour_group = popargs(
            "reference_frame", "vertex_count", "is_external", "contour_group", kwargs
        )

        # Passing a TimeSeries as 'data' to link another series resolves to that series' own
        # array before this runs, so the shape here is the one the file will hold either way.
        # get_data_shape still reports None for a shape it cannot determine without consuming
        # an iterator, so skip the cross-check in that case rather than guess at it.
        data_shape = get_data_shape(kwargs["data"])
        count_shape = get_data_shape(vertex_count)
        # Compare only the dimensions both shapes report; the h5py datasets read back from a
        # file always report concrete lengths.
        if data_shape is not None and not _shapes_agree(data_shape[:2], count_shape):
            raise ValueError(
                "ContourSeries 'vertex_count' shape %s must match the first two dimensions of "
                "'data' %s (num_frames, num_contours)." % (count_shape, data_shape[:2])
            )
        for name, value in (("is_external", is_external), ("contour_group", contour_group)):
            if value is None:  # only contour_group may be omitted
                continue
            value_shape = get_data_shape(value)
            if not _shapes_agree(count_shape, value_shape):
                raise ValueError(
                    "ContourSeries '%s' shape %s must match 'vertex_count' shape %s "
                    "(num_frames, num_contours)." % (name, value_shape, count_shape)
                )

        # A count larger than the vertex capacity of 'data' would silently truncate a
        # contour: a reader taking data[frame, slot, :vertex_count] would get fewer
        # vertices than it was promised, with nothing to signal the loss.
        capacity = data_shape[2] if data_shape is not None and len(data_shape) > 2 else None
        observed = _max_value(vertex_count)
        if capacity is not None and observed is not None and observed > capacity:
            raise ValueError(
                "ContourSeries 'vertex_count' has a maximum of %d, but each contour slot of "
                "'data' holds only %d vertices." % (observed, capacity)
            )

        super().__init__(**kwargs)
        self.reference_frame = reference_frame
        self.vertex_count = vertex_count
        self.is_external = is_external
        self.contour_group = contour_group


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
        {
            "add": "add_contour_series",
            "get": "get_contour_series",
            "create": "create_contour_series",
            "type": ContourSeries,
            "attr": "contour_series",
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
            "name": "contour_series",
            "type": ("array_data", "data"),
            "doc": "Segmentation contours outlining the instance described by this PoseEstimation.",
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
                "DEPRECATED. Please use the 'source_video' argument instead. "
                "Paths to the original video files. Note: these string paths might be fragile unless relative "
                "paths are used and care is taken to keep them consistent."
            ),
            "default": None,
        },
        {
            "name": "labeled_videos",
            "type": ("array_data", "data"),
            "shape": (None,),
            "doc": (
                "DEPRECATED. Please use the 'labeled_video' argument instead. "
                "Paths to the labeled video files. Note: these string paths might be fragile unless relative "
                "paths are used and care is taken to keep them consistent."
            ),
            "default": None,
        },
        {
            "name": "dimensions",
            "type": ("array_data", "data"),
            "shape": (None, 2),
            "doc": (
                "DEPRECATED. Please use the 'dimension' field of the ImageSeries linked as 'source_video' or "
                "'labeled_video' instead. "
                "Dimensions of each labeled video file."
            ),
            "default": None,
        },
        {
            "name": "device",
            "type": Device,
            "doc": (
                "The camera device used to record the video for this pose estimation. Must be added to the "
                "NWBFile before being linked here. Use a CalibratedCamera instead of a plain Device when "
                "intrinsic/extrinsic calibration coordinates are available."
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
                "It holds the video data, or the path to an external video file, along with the "
                "dimensions and frame timing of the video."
            ),
            "default": None,
        },
        {
            "name": "labeled_video",
            "type": ImageSeries,
            "doc": (
                "Link to an ImageSeries containing the labeled video (with pose estimation overlays) "
                "produced from the source video. The ImageSeries should be stored in the NWBFile "
                "(e.g., in acquisition) and linked here. It holds the video data, or the path to an "
                "external video file, along with the dimensions and frame timing of the video."
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
        if devices is not None:
            if device is not None:
                raise ValueError("Cannot specify both 'device' and 'devices'. Please use 'device' only.")
            if len(devices) > 1:
                raise ValueError(
                    "A PoseEstimation object represents pose estimates from a single camera view and accepts one "
                    "device, but %d devices were passed. Add one PoseEstimation object per camera view to a "
                    "MultiCameraPoseEstimation object." % len(devices)
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
        # warn on new, no warning on construction from existing file
        if not self._in_construct_mode:
            if original_videos is not None:
                msg = (
                    "The 'original_videos' constructor argument is deprecated. Please use the 'source_video' "
                    "argument instead. This will be removed in a future release."
                )
                warnings.warn(msg, DeprecationWarning)
            if labeled_videos is not None:
                msg = (
                    "The 'labeled_videos' constructor argument is deprecated. Please use the 'labeled_video' "
                    "argument instead. This will be removed in a future release."
                )
                warnings.warn(msg, DeprecationWarning)
            if dimensions is not None:
                msg = (
                    "The 'dimensions' constructor argument is deprecated. Please use the 'dimension' field of the "
                    "ImageSeries linked as 'source_video' or 'labeled_video' instead. This will be removed in a "
                    "future release."
                )
                warnings.warn(msg, DeprecationWarning)

        pose_estimation_series, description = popargs("pose_estimation_series", "description", kwargs)
        contour_series = popargs("contour_series", kwargs)
        scorer = popargs("scorer", kwargs)
        source_software, source_software_version = popargs("source_software", "source_software_version", kwargs)
        if source_software_version is not None and source_software is None:
            raise ValueError(
                "'source_software_version' was specified without 'source_software'. The version is stored as an "
                "attribute on the 'source_software' dataset, so 'source_software' must be provided as well."
            )
        super().__init__(**kwargs)

        self.pose_estimation_series = pose_estimation_series
        if contour_series is not None:
            self.contour_series = contour_series
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
    reference from multiple PoseEstimation objects (e.g., one per subject in a multi-subject recording
    session), so the camera rig and its calibration are never duplicated.
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
            "doc": (
                "Lens distortion coefficients for this camera. Length depends on the distortion model "
                "(typically 4 or 5 for radial-tangential)."
            ),
            "default": None,
        },
        # every Device constructor argument except 'name', which is redefined above
        *get_docval(Device.__init__)[1:],
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
    is organized through PoseEstimation children, one per camera view.
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
        if source_software_version is not None and source_software is None:
            raise ValueError(
                "'source_software_version' was specified without 'source_software'. The version is stored as an "
                "attribute on the 'source_software' dataset, so 'source_software' must be provided as well."
            )
        super().__init__(**kwargs)

        self.pose_estimation_series = pose_estimation_series
        self.pose_estimations = pose_estimations
        self.description = description
        self.scorer = scorer
        self.source_software = source_software
        self.source_software_version = source_software_version
        self.skeleton = skeleton
