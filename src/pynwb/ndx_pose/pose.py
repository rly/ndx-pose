from hdmf.utils import docval, popargs, get_docval, AllowPositional

from pynwb import register_class, TimeSeries
from pynwb.behavior import SpatialSeries
from pynwb.core import MultiContainerInterface
from pynwb.device import Device

from . import Skeleton

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
            'add': 'add_pose_estimation_series',
            'get': 'get_pose_estimation_series',
            'create': 'create_pose_estimation_series',
            'type': PoseEstimationSeries,
            'attr': 'pose_estimation_series'
        },
        # {
        #     'add': 'add_device',
        #     'get': 'get_devices',
        #     'type': Device,
        #     'attr': 'devices'
        #     # TODO prevent these from being children / add better support for links
        #     # may require update to HDMF to add a key 'child': False
        # }
    ]

    __nwbfields__ = ('description', 'original_videos', 'labeled_videos', 'dimensions', 'scorer', 'source_software',
                     'source_software_version', 'nodes', 'edges')

    # custom mapper in ndx_pose.io.pose maps:
    # 'source_software' dataset -> 'version' attribute to 'source_software_version' field

    @docval(  # all fields optional
        {'name': 'pose_estimation_series', 'type': ('array_data', 'data'),
         'doc': ('Estimated position data for each body part.'),
         'default': None},
        {'name': 'name', 'type': str,
         'doc': ('Description of the pose estimation procedure and output.'),
         'default': 'PoseEstimation'},
        {'name': 'description', 'type': str,
         'doc': ('Description of the pose estimation procedure and output.'),
         'default': None},
        {'name': 'original_videos', 'type': ('array_data', 'data'), 'shape': (None, ),
         'doc': ('Paths to the original video files. The number of files should equal the number of camera devices.'),
         'default': None},
        {'name': 'labeled_videos', 'type': ('array_data', 'data'), 'shape': (None, ),
         'doc': ('Paths to the labeled video files. The number of files should equal the number of camera devices.'),
         'default': None},
        {'name': 'dimensions', 'type': ('array_data', 'data'), 'shape': ((None, 2)),
         'doc': ('Dimensions of each labeled video file.'),
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
        {'name': 'skeleton', 'type': Skeleton,
         'doc': ("Layout of body part locations and connections."),
         'default': None},
        # {'name': 'devices', 'type': ('array_data', 'data'),
        #  'doc': ('Cameras used to record the videos.'),
        #  'default': None},
        allow_positional=AllowPositional.ERROR
    )
    def __init__(self, **kwargs):
        pose_estimation_series, description = popargs('pose_estimation_series', 'description', kwargs)
        original_videos, labeled_videos,  = popargs('original_videos', 'labeled_videos', kwargs)
        dimensions, scorer = popargs('dimensions', 'scorer', kwargs)
        source_software, source_software_version = popargs('source_software', 'source_software_version', kwargs)
        nodes, edges, skeleton = popargs('nodes', 'edges', 'skeleton', kwargs)
        # devices = popargs('devices', kwargs)
        super().__init__(**kwargs)
        self.pose_estimation_series = pose_estimation_series
        self.description = description
        self.original_videos = original_videos
        self.labeled_videos = labeled_videos
        self.dimensions = dimensions
        self.scorer = scorer
        self.source_software = source_software
        self.source_software_version = source_software_version
        self.skeleton = skeleton
        # self.devices = devices

        # TODO include calibration images for 3D estimates?

        # if original_videos is not None and (devices is None or len(original_videos) != len(devices)):
        #     raise ValueError("The number of original videos should equal the number of camera devices.")
        # if labeled_videos is not None and (devices is None or len(labeled_videos) != len(devices)):
        #     raise ValueError("The number of labeled videos should equal the number of camera devices.")
        # if dimensions is not None and (devices is None or len(dimensions) != len(devices)):
        #     raise ValueError("The number of dimensions should equal the number of camera devices.")

        # TODO validate nodes and edges correspondence, convert edges to uint
