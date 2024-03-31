# Changelog for ndx-pose

## ndx-pose 0.2.0 (Upcoming)

### Breaking changes
- Removed the `nodes` and `edges` fields from `PoseEstimation` neurodata type. To specify these,
  create a `Skeleton` object with those values, create a `Skeletons` object and pass the `Skeleton`
  object to that, and add the `Skeletons` object to your "behavior" processing module. @rly (#7, #24)

### Major changes
- Added support for storing training data in the new `PoseTraining` neurodata type and other new types.
  @roomrys, @CBroz1, @rly, @talmo, @eberrigan (#7, #21, #24)

### Minor changes
- Made `PoseEstimation.confidence` optional. @h-mayorquin (#11)
- Added ability to link a `PoseEstimation` object to one or more camera devices.

## ndx-pose 0.1.1 (January 26, 2022)

The schema is unchanged, but the Python package has been updated.

### Bug fix
- Remove hdmf-docutils as an installation dependency. @rly (#6)

## ndx-pose 0.1.0 (January 26, 2022)

Initial release including `PoseEstimation` and `PoseEstimationSeries` neurodata types.
