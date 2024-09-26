"""Evaluate examples of how to use the ndx-pose extension."""

import subprocess
from pathlib import Path

def test_example_usage_estimates_only():
    """Call examples/write_pose_estimates_only.py and check that it runs without errors."""
    subprocess.run(["python", "examples/write_pose_estimates_only.py"], check=True)

    # Remove the generated test_pose.nwb if it exists
    if Path("test_pose.nwb").exists():
        Path("test_pose.nwb").unlink()


def test_example_usage_training_only():
    """Call examples/write_pose_training.py and check that it runs without errors."""
    subprocess.run(["python", "examples/write_pose_training.py"], check=True)

    # Remove the generated test_pose.nwb if it exists
    if Path("test_pose.nwb").exists():
        Path("test_pose.nwb").unlink()
