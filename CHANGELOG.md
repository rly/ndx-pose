# ndx-pose Changelog

## ndx-pose 0.2.0 (Upcoming)

### Breaking changes
- Removed "nodes" and "edges" fields from `PoseEstimation` neurodata type. Create a `Skeleton` object and pass
  it to the `skeleton` keyword argument of `PoseEstimation.__init__` instead. @rly (#7)

### Minor changes
- Made `PoseEstimation.confidence` optional. @h-mayorquin (#11)
- Added ability to link a `PoseEstimation` object to one or more camera devices.

## ndx-pose 0.1.1 (January 26, 2022)

The schema is unchanged, but the Python package has been updated.

### Bug fix
- Remove hdmf-docutils as an installation dependency. @rly (#6)

## ndx-pose 0.1.0 (January 26, 2022)

Initial release including `PoseEstimation` and `PoseEstimationSeries` neurodata types.
