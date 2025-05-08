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
