from pynwb import register_map
from pynwb.io.base import TimeSeriesMap
from pynwb.io.core import NWBContainerMapper

from ..pose import PoseEstimation, PoseEstimationSeries


@register_map(PoseEstimationSeries)
class PoseEstimationSeriesMap(TimeSeriesMap):

    def __init__(self, spec):
        super().__init__(spec)
        confidence_spec = self.spec.get_dataset('confidence')
        self.map_spec('confidence_definition', confidence_spec.get_attribute('definition'))


@register_map(PoseEstimation)
class PoseEstimationMap(NWBContainerMapper):

    def __init__(self, spec):
        super().__init__(spec)
        source_software_spec = self.spec.get_dataset('source_software')
        self.map_spec('source_software_version', source_software_spec.get_attribute('version'))

        # TODO if reading a file without the pose_estimates group, load the PoseEstimationSeries from the
        # main PoseEstimation group into the pose_estimation_series variable

        pose_estimates_spec = self.spec.get_group('pose_estimation_series')
        self.unmap(pose_estimates_spec)
        self.map_spec('pose_estimation_series', pose_estimates_spec.get_neurodata_type('PoseEstimationSeries'))

        original_videos_series_spec = self.spec.get_group('original_videos_series')
        self.unmap(original_videos_series_spec)
        self.map_spec('original_videos_series', original_videos_series_spec.get_target_type('ImageSeries'))

        labeled_videos_series_spec = self.spec.get_group('labeled_videos_series')
        self.unmap(labeled_videos_series_spec)
        self.map_spec('labeled_videos_series', labeled_videos_series_spec.get_neurodata_type('ImageSeries'))
