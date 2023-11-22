from pynwb import register_map
from pynwb.io.base import TimeSeriesMap
from pynwb.io.core import NWBContainerMapper

from ..pose import PoseEstimation, PoseEstimationSeries, PoseTraining


@register_map(PoseEstimationSeries)
class PoseEstimationSeriesMap(TimeSeriesMap):

    def __init__(self, spec):
        """Map attribute spec "definition" to Python instance attribute "confidence_definition"."""
        super().__init__(spec)
        confidence_spec = self.spec.get_dataset('confidence')
        self.map_spec('confidence_definition', confidence_spec.get_attribute('definition'))


@register_map(PoseEstimation)
class PoseEstimationMap(NWBContainerMapper):

    def __init__(self, spec):
        """Map attribute spec "version" to Python instance attribute "source_software_version"."""
        super().__init__(spec)
        source_software_spec = self.spec.get_dataset('source_software')
        self.map_spec('source_software_version', source_software_spec.get_attribute('version'))

        cameras_spec = self.spec.get_group('cameras')
        self.unmap(cameras_spec)
        self.map_spec('cameras', cameras_spec.get_target_type('Device'))


@register_map(PoseTraining)
class PoseTrainingMap(NWBContainerMapper):

    def __init__(self, spec):
        """Map neurodata type specs within organizational groups to Python instance attributes.

        Map spec "skeletons/<Skeleton>" to Python instance attribute "skeletons".
        Map spec "training_frames/<TrainingFrame>" to Python instance attribute "training_frames".
        Map spec "source_videos/<ImageSeries>" to Python instance attribute "source_videos".
        """
        super().__init__(spec)

        skeletons_spec = self.spec.get_group('skeletons')
        self.unmap(skeletons_spec)
        self.map_spec('skeletons', skeletons_spec.get_neurodata_type('Skeleton'))

        training_frames_spec = self.spec.get_group('training_frames')
        self.unmap(training_frames_spec)
        self.map_spec('training_frames', training_frames_spec.get_neurodata_type('TrainingFrame'))

        source_videos_spec = self.spec.get_group('source_videos')
        self.unmap(source_videos_spec)
        self.map_spec('source_videos', source_videos_spec.get_neurodata_type('ImageSeries'))