from hdmf.build import ObjectMapper
from pynwb import register_map
from pynwb.device import Device
from pynwb.io.base import TimeSeriesMap
from pynwb.io.core import NWBContainerMapper

from ..pose import MultiCameraPoseEstimation, PoseEstimation, PoseEstimationSeries

# ObjectMapper.NO_OVERRIDE is the sentinel a constructor_arg override function returns to fall through to
# the value built from the file. hdmf < 6.2.0 has no sentinel and uses a None return for that.
NO_OVERRIDE = getattr(ObjectMapper, "NO_OVERRIDE", None)


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

    @NWBContainerMapper.constructor_arg("device")
    def device(self, builder, manager):
        """Raise when the GroupBuilder links more than one Device.

        Used when constructing a PoseEstimation container from a written file.

        ndx-pose 0.4.0 scoped PoseEstimation to a single camera view and named the Device link "device".
        Data written with earlier versions holds zero or more Device links, each named after its target.
        HDMF resolves links by type, so a PoseEstimation group that links two cameras would set the
        'device' constructor arg to a list of Device objects. Raise here with an explanation of how that
        data is stored as of 0.4.0.

        Returning NO_OVERRIDE leaves the 'device' constructor arg to HDMF's usual link resolution, which
        yields the single linked Device, or nothing when no Device is linked.
        """
        device_links = [link for link in builder.links.values() if issubclass(manager.get_cls(link.builder), Device)]
        if len(device_links) > 1:
            raise ValueError(
                "This PoseEstimation group links %d Device objects, but a PoseEstimation object represents pose "
                "estimates from a single camera view and supports only one device. The file was written with "
                "ndx-pose < 0.4.0, when a PoseEstimation object could link to multiple cameras. Reading it is not "
                "supported; each camera view needs its own PoseEstimation object inside a "
                "MultiCameraPoseEstimation object." % len(device_links)
            )
        return NO_OVERRIDE

    @NWBContainerMapper.constructor_arg("nodes")
    def nodes(self, builder, manager):
        """Set the constructor arg for 'nodes' to the value of the GroupBuilder dataset "nodes".

        Used when constructing a PoseEstimation container from a written file.

        ndx-pose 0.2.0 introduced a new attribute 'skeleton' to the PoseEstimation container. This Skeleton
        container has two datasets, 'nodes' and 'edges', which were previously stored directly in the
        PoseEstimation container. When data written with ndx-pose versions < 0.2.0 are read, the 'nodes' and
        'edges' arguments in the PoseEstimation constructor are set to the values of the "nodes" and "edges"
        DatasetBuilders read from the file. When data written with ndx-pose versions >= 0.2.0 are read,
        'nodes' and 'edges' are left to the PoseEstimation constructor defaults.
        """
        nodes_builder = builder.datasets.get("nodes")
        if nodes_builder:
            return nodes_builder.data
        return NO_OVERRIDE

    @NWBContainerMapper.constructor_arg("edges")
    def edges(self, builder, manager):
        """Set the constructor arg for 'edges' to the value of the GroupBuilder dataset "edges".

        Used when constructing a PoseEstimation container from a written file.

        ndx-pose 0.2.0 introduced a new attribute 'skeleton' to the PoseEstimation container. This Skeleton
        container has two datasets, 'nodes' and 'edges', which were previously stored directly in the
        PoseEstimation container. When data written with ndx-pose versions < 0.2.0 are read, the 'nodes' and
        'edges' arguments in the PoseEstimation constructor are set to the values of the "nodes" and "edges"
        DatasetBuilders read from the file. When data written with ndx-pose versions >= 0.2.0 are read,
        'nodes' and 'edges' are left to the PoseEstimation constructor defaults.
        """
        edges_builder = builder.datasets.get("edges")
        if edges_builder:
            return edges_builder.data
        return NO_OVERRIDE


@register_map(MultiCameraPoseEstimation)
class MultiCameraPoseEstimationMap(NWBContainerMapper):

    def __init__(self, spec):
        """Map attribute spec "version" to Python instance attribute "source_software_version"."""
        super().__init__(spec)
        source_software_spec = self.spec.get_dataset("source_software")
        self.map_spec("source_software_version", source_software_spec.get_attribute("version"))
