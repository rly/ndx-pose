from pynwb import register_map
from pynwb.io.base import TimeSeriesMap
from pynwb.io.core import NWBContainerMapper

from ..pose import PoseEstimation, PoseEstimationSeries


@register_map(PoseEstimationSeries)
class PoseEstimationSeriesMap(TimeSeriesMap):

    def __init__(self, spec):
        """Map attribute spec "definition" to Python instance attribute "confidence_definition"."""
        super().__init__(spec)
        confidence_spec = self.spec.get_dataset("confidence")
        self.map_spec("confidence_definition", confidence_spec.get_attribute("definition"))


@register_map(PoseEstimation)
class PoseEstimationMap(NWBContainerMapper):

    def __init__(self, spec):
        """Map attribute spec "version" to Python instance attribute "source_software_version"."""
        super().__init__(spec)
        source_software_spec = self.spec.get_dataset("source_software")
        self.map_spec("source_software_version", source_software_spec.get_attribute("version"))
