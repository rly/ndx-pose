"""Evaluate examples of how to use the ndx-pose extension."""

import subprocess
import sys
from pathlib import Path

# the examples read and write relative to the repository root
REPO_ROOT = Path(__file__).parents[3]


def run_example(script_name, output_name):
    """Run an example script from the repository root and remove the NWB file it writes."""
    try:
        subprocess.run([sys.executable, str(Path("examples") / script_name)], check=True, cwd=REPO_ROOT)
    finally:
        output_path = REPO_ROOT / output_name
        if output_path.exists():
            output_path.unlink()


def test_example_usage_estimates_only():
    """Call examples/write_pose_estimates_only.py and check that it runs without errors."""
    run_example("write_pose_estimates_only.py", "test_pose.nwb")


def test_example_usage_training_only():
    """Call examples/write_pose_training.py and check that it runs without errors."""
    run_example("write_pose_training.py", "test_pose.nwb")


def test_example_usage_multicamera():
    """Call examples/write_multicamera_pose_estimates.py and check that it runs without errors."""
    run_example("write_multicamera_pose_estimates.py", "test_multicamera_pose.nwb")
