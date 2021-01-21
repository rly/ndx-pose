from pynwb import register_map
from pynwb.io.core import NWBContainerMapper

from ..pose import PoseEstimation


@register_map(PoseEstimation)
class PoseEstimationMap(NWBContainerMapper):

    def __init__(self, spec):
        super().__init__(spec)
        source_software_spec = self.spec.get_dataset('source_software')
        self.map_spec('source_software_version', source_software_spec.get_attribute('version'))
