from hdmf.utils import docval, popargs, get_docval, AllowPositional

from pynwb import register_class, TimeSeries
from pynwb.behavior import SpatialSeries
from pynwb.core import MultiContainerInterface
from pynwb.image import ImageSeries

import warnings


@register_class('PoseEstimationSeries', 'ndx-pose')
class PoseEstimationSeries(SpatialSeries):
    """Estimated position (x, y) or (x, y, z) of a body part over time.
    """  # TODO

    __nwbfields__ = ('confidence', 'confidence_definition')

    # custom mapper in ndx_pose.io.pose maps:
    # 'confidence' dataset -> 'definition' attribute to 'confidence_definition' field

    @docval(
        {'name': 'name', 'type': str,
         'doc': ('Name of this PoseEstimationSeries, usually the name of a body part.')},  # required
        {'name': 'data', 'type': ('array_data', 'data', TimeSeries), 'shape': ((None, 2), (None, 3)),  # required
         'doc': ('Estimated position (x, y) or (x, y, z).')},
        {'name': 'reference_frame', 'type': str,   # required
         'doc': 'Description defining what the zero-position (0, 0) or (0, 0, 0) is.'},
        {'name': 'confidence', 'type': ('array_data', 'data'), 'shape': (None, ),
         'doc': ('Confidence or likelihood of the estimated positions, scaled to be between 0 and 1.'),
          'default': None,},
        {'name': 'unit', 'type': str,
         'doc': ("Base unit of measurement for working with the data. The default value "
                 "is 'pixels'. Actual stored values are not necessarily stored in these units. "
                 "To access the data in these units, multiply 'data' by 'conversion'."),
         'default': 'pixels'},
        {'name': 'confidence_definition', 'type': str,
         'doc': ("Description of how the confidence was computed, e.g., "
                 "'Softmax output of the deep neural network'."),
         'default': None},
        *get_docval(TimeSeries.__init__, 'conversion', 'resolution', 'timestamps', 'starting_time', 'rate',
                    'comments', 'description', 'control', 'control_description'),
        allow_positional=AllowPositional.ERROR
    )
    def __init__(self, **kwargs):
        """Construct a new PoseEstimationSeries representing pose estimates for a particular body part."""
        confidence, confidence_definition = popargs('confidence', 'confidence_definition', kwargs)
        super().__init__(**kwargs)
        self.confidence = confidence
        self.confidence_definition = confidence_definition


@register_class('PoseEstimation', 'ndx-pose')
class PoseEstimation(MultiContainerInterface):
    """Estimated position data for multiple body parts, computed from the same video with the same tool/algorithm.
    The timestamps of each child PoseEstimationSeries type should be the same.
    """

    __clsconf__ = [
        {
            # NOTE pose_estimation_series was remapped in version 0.2.0 to live under the pose_estimates subgroup
            'add': 'add_pose_estimation_series',
            'get': 'get_pose_estimation_series',
            'create': 'create_pose_estimation_series',
            'type': PoseEstimationSeries,
            'attr': 'pose_estimation_series'
        },
        {
            'add': 'add_original_videos_series',
            'get': 'get_original_videos_series',
            'create': 'create_original_videos_series',
            'type': ImageSeries,
            'attr': 'original_videos_series'
        },
        {  # TODO how to check that these are links and not subgroups?
            'add': 'add_labeled_videos_series',
            'get': 'get_labeled_videos_series',
            'create': 'create_labeled_videos_series',
            'type': ImageSeries,
            'attr': 'labeled_videos_series'
        },
    ]

    __nwbfields__ = ('description', 'original_videos', 'labeled_videos', 'dimensions', 'scorer',
                     'source_software', 'source_software_version', 'nodes', 'edges')

    # custom mapper in ndx_pose.io.pose maps:
    # 'source_software' dataset, 'version' attribute to 'source_software_version' field
    # 'pose_estimates' untyped group, 'PoseEstimationSeries' subgroup to 'pose_estimates' field
    # 'original_videos_series' untyped group, 'ImageSeries' subgroup to 'original_videos_series' field
    # 'labeled_videos_series' untyped group, 'ImageSeries' subgroup to 'labeled_videos_series' field

    @docval(  # all fields optional
        {'name': 'pose_estimation_series', 'type': (list, tuple),
         'doc': ('Estimated position data for each body part.'),
         'default': None},
        {'name': 'name', 'type': str,
         'doc': ('Description of the pose estimation procedure and output.'),
         'default': 'PoseEstimation'},
        {'name': 'description', 'type': str,
         'doc': ('Description of the pose estimation procedure and output.'),
         'default': None},
        {'name': 'original_videos', 'type': ('array_data', 'data'), 'shape': (None, ),
         'doc': ('The original video files.'),
         'default': None},
        {'name': 'original_videos_series', 'type': (list, tuple), 'shape': (None, ),
         'doc': ('The original video files.'),
         'default': None},
        {'name': 'labeled_videos', 'type': ('array_data', 'data'), 'shape': (None, ),
         'doc': ('Links to the labeled video files. The number of files should equal the number of original videos.'),
         'default': None},
        {'name': 'labeled_videos_series', 'type': (list, tuple), 'shape': (None, ),
         'doc': ('Links to the labeled videos. The number of files should equal the number of original videos.'),
         'default': None},
        {'name': 'dimensions', 'type': ('array_data', 'data'), 'shape': ((None, 2)),
         'doc': ('Dimensions of each labeled video file. Deprecated in version 0.2.0. '
                 'Use "dimension" in original_videos instead.'),
         'default': None},
        {'name': 'scorer', 'type': str,
         'doc': ('Name of the scorer / algorithm used.'),
         'default': None},
        {'name': 'source_software', 'type': str,
         'doc': ('Name of the software tool used. Specifying the version attribute is strongly encouraged.'),
         'default': None},
        {'name': 'source_software_version', 'type': str,
         'doc': ('Version string of the software tool used.'),
         'default': None},
        {'name': 'nodes', 'type': ('array_data', 'data'),
         'doc': ('Array of body part names corresponding to the names of the PoseEstimationSeries objects within '
                 'this container.'),
         'default': None},
        {'name': 'edges', 'type': ('array_data', 'data'),
         'doc': ("Array of pairs of indices corresponding to edges between nodes. Index values correspond to row "
                 "indices of the 'nodes' field. Index values use 0-indexing."),
         'default': None},
        allow_positional=AllowPositional.ERROR
    )
    def __init__(self, **kwargs):
        pose_estimation_series, description = popargs('pose_estimation_series', 'description', kwargs)
        original_videos, labeled_videos = popargs('original_videos', 'labeled_videos', kwargs)
        original_videos_series = popargs('original_videos_series', kwargs)
        labeled_videos_series = popargs('labeled_videos_series', kwargs)
        dimensions, scorer = popargs('dimensions', 'scorer', kwargs)
        source_software, source_software_version = popargs('source_software', 'source_software_version', kwargs)
        nodes, edges = popargs('nodes', 'edges', kwargs)

        super().__init__(**kwargs)
        self.pose_estimation_series = pose_estimation_series
        self.description = description

        if original_videos is not None:
            warnings.warn("The 'original_videos' field has been deprecated in version 0.2.0. Use "
                          "'original_videos_series' instead. The provided "
                          "file paths will be converted to ImageSeries objects where the 'external_file' field is set "
                          "to each file path.",
                          DeprecationWarning)
            if original_videos_series is None:
                warnings.warn(
                    "The provided file paths in 'original_videos' will be converted to ImageSeries objects where the "
                    "'external_file' field is set to each file path.",
                    DeprecationWarning
                )
                original_videos_series = list()
                for i, file_path in enumerate(original_videos):
                    image_series = ImageSeries(
                        name="original_video" + str(i),
                        format="external",
                        external_file=file_path,
                        dimension=dimensions[0] if dimensions is not None and dimensions[0] is not None else None,
                    )
                    original_videos_series.append(image_series)
        self.original_videos = original_videos
        self.original_videos_series = original_videos_series

        if labeled_videos is not None:
            warnings.warn("The 'labeled_videos' field has been deprecated in version 0.2.0. Use "
                          "'labeled_videos_series' instead.", DeprecationWarning)
            if labeled_videos_series is None:
                warnings.warn(
                    "The provided file paths in 'labeled_videos' will be converted to ImageSeries objects where the "
                    "'external_file' field is set to each file path.",
                    DeprecationWarning
                )
                labeled_videos_series = list()
                for i, file_path in enumerate(labeled_videos):
                    image_series = ImageSeries(
                        name="labeled_video" + str(i),
                        format="external",
                        external_file=file_path,
                        dimension=dimensions[0] if dimensions is not None and dimensions[0] is not None else None,
                    )
                    labeled_videos_series.append(image_series)
        self.labeled_videos = labeled_videos
        self.labeled_videos_series = labeled_videos_series

        if dimensions is not None:
            warnings.warn("The 'dimensions' field has been deprecated in version 0.2.0. "
                          "Use 'dimension' in 'original_videos' instead.", DeprecationWarning)
        self.dimensions = dimensions
        self.scorer = scorer
        self.source_software = source_software
        self.source_software_version = source_software_version
        self.nodes = nodes
        self.edges = edges

        # TODO include calibration images for 3D estimates?

        # TODO validate nodes and edges correspondence, convert edges to uint
