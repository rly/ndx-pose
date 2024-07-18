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

    @NWBContainerMapper.constructor_arg("nodes")
    def nodes(self, builder, manager):
        """Set the constructor arg for 'nodes' to the value of the GroupBuilder dataset "nodes".

        Used when constructing a PoseEstimation container from a written file.

        ndx-pose 0.2.0 introduced a new attribute 'skeleton' to the PoseEstimation container. This Skeleton
        container has two datasets, 'nodes' and 'edges', which were previously stored directly in the
        PoseEstimation container. When data written with ndx-pose versions < 0.2.0 are read, the 'nodes' and
        'edges' arguments in the PoseEstimation constructor are set to the values of the "nodes" and "edges"
        DatasetBuilders read from the file. When data written with ndx-pose versions >= 0.2.0 are read,
        'nodes' and 'edges' are set to None in the PoseEstimation constructor.
        """
        nodes_builder = builder.datasets.get("nodes")
        if nodes_builder:
            nodes = nodes_builder.data
        else:
            nodes = None
        return nodes

    @NWBContainerMapper.constructor_arg("edges")
    def edges(self, builder, manager):
        """Set the constructor arg for 'edges' to the value of the GroupBuilder dataset "edges".

        Used when constructing a PoseEstimation container from a written file.

        ndx-pose 0.2.0 introduced a new attribute 'skeleton' to the PoseEstimation container. This Skeleton
        container has two datasets, 'nodes' and 'edges', which were previously stored directly in the
        PoseEstimation container. When data written with ndx-pose versions < 0.2.0 are read, the 'nodes' and
        'edges' arguments in the PoseEstimation constructor are set to the values of the "nodes" and "edges"
        DatasetBuilders read from the file. When data written with ndx-pose versions >= 0.2.0 are read,
        'nodes' and 'edges' are set to None in the PoseEstimation constructor.
        """
        edges_builder = builder.datasets.get("edges")
        if edges_builder:
            edges = edges_builder.data
        else:
            edges = None
        return edges
