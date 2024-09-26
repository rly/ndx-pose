"""Evaluate examples of how to use the ndx-pose extension."""

import subprocess


def test_example_usage_estimates_only():
    """Call examples/write_pose_estimates_only.py and check that it runs without errors."""
    subprocess.run(["python", "examples/write_pose_estimates_only.py"], check=True)


def test_example_usage_training_only():
    """Call examples/write_pose_training.py and check that it runs without errors."""
    subprocess.run(["python", "examples/write_pose_training.py"], check=True)
