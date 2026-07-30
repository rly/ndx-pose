"""Test reading and validating NWB files generated with previous versions of this package."""

import numpy as np
import numpy.testing as npt
from pathlib import Path
from pynwb import NWBHDF5IO, validate
import pytest
import warnings

# NOTE: if this package is not imported, then the custom containers and mappers will not be used
import ndx_pose  # noqa: F401


def get_io(path):
    """Get an NWBHDF5IO object for the given path."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Ignoring cached namespace .*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Ignoring the following cached namespace.*",
            category=UserWarning,
        )
        return NWBHDF5IO(str(path), "r")


def test_0_1_1_poseestimation_nodes_edges():
    """Test that PoseEstimation objects with nodes and edges written before 0.2.0 are read correctly."""
    f = Path(__file__).parent / "0.1.1_poseestimation_nodes_edges.nwb"
    with get_io(f) as io:
        read_nwbfile = io.read()
        npt.assert_array_equal(
            read_nwbfile.processing["behavior"]["PoseEstimation"].skeleton.nodes[:],
            ["front_left_paw", "front_right_paw"],
        )
        npt.assert_array_equal(
            read_nwbfile.processing["behavior"]["PoseEstimation"].skeleton.edges[:], np.array([[0, 1]], dtype="uint8")
        )


def test_0_3_0_poseestimation_one_camera():
    """Test that a PoseEstimation object written before 0.4.0 with one camera device is read correctly.

    The Device link is stored under the name of its target rather than the name "device" used by the 0.4.0
    schema, so this exercises how the link is resolved when reading.
    """
    f = Path(__file__).parent / "0.3.0_poseestimation_one_camera.nwb"
    with get_io(f) as io:
        read_nwbfile = io.read()
        pe = read_nwbfile.processing["behavior"]["PoseEstimation"]
        assert pe.device is read_nwbfile.devices["camera1"]
        npt.assert_array_equal(pe.original_videos[:], ["camera1.mp4"])


def test_0_3_0_poseestimation_two_cameras():
    """Test that a PoseEstimation object written before 0.4.0 with two camera devices raises on read.

    A PoseEstimation object holds at most one device as of 0.4.0, so there is no single device to assign the
    two linked cameras to. The error directs the reader to MultiCameraPoseEstimation.
    """
    f = Path(__file__).parent / "0.3.0_poseestimation_two_cameras.nwb"
    with get_io(f) as io:
        # HDMF reports a failed construction as a ConstructError chained to the underlying error, so match
        # on the message HDMF interpolates and check the chained error for the type ndx-pose raises.
        with pytest.raises(Exception, match="supports only one device") as exc_info:
            io.read()

    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError), cause
    assert "MultiCameraPoseEstimation" in str(cause)


@pytest.mark.parametrize(
    "file_path,expected_warnings,expected_errors",
    [
        (
            Path(__file__).parent / "0.1.1_poseestimation_nodes_edges.nwb",
            [],
            [],
        ),
        (
            Path(__file__).parent / "0.1.1_poseestimation_no_cameras.nwb",
            [],
            [],
        ),
        (
            Path(__file__).parent / "0.3.0_poseestimation_one_camera.nwb",
            [],
            [],
        ),
    ],
)
def test_read(file_path, expected_warnings, expected_errors):
    """Test reading and validating NWB files generated with previous versions of this package."""
    with warnings.catch_warnings(record=True) as warnings_on_read:
        warnings.simplefilter("always")
        with get_io(file_path) as io:
            errors = validate(io=io)
            io.read()
            # NOTE: this does not error if the expected warnings are not present
            for w in warnings_on_read:
                if str(w.message) not in expected_warnings:
                    raise Exception("Unexpected warning: %s: %s" % (file_path, str(w.message)))
            if errors:
                unexpected_errors = []
                for e in errors:
                    if str(e) not in expected_errors:
                        warnings.warn("%s: %s" % (file_path, e))
                        unexpected_errors.append(e)
                if unexpected_errors:
                    raise Exception("%d validation error(s). See warnings." % len(unexpected_errors))
