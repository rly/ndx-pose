import os
from pynwb import load_namespaces, get_class

# Set path of the namespace.yaml file to the expected install location
ndx_pose_specpath = os.path.join(
    os.path.dirname(__file__), "spec", "ndx-pose.namespace.yaml"
)

# If the extension has not been installed yet but we are running directly from
# the git repo
if not os.path.exists(ndx_pose_specpath):
    ndx_pose_specpath = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "spec",
            "ndx-pose.namespace.yaml",
        )
    )

# Load the namespace
load_namespaces(ndx_pose_specpath)

from . import io as __io  # noqa: E402, F401
from .pose import (
    PoseEstimation,
    PoseEstimationSeries,
    Skeleton,
    Skeletons,
    TrainingFrame,
    TrainingFrames,
    SkeletonInstance,
    SkeletonInstances,
    SourceVideos,
    PoseTraining,
)  # noqa: E402, F401
