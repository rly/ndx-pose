# Changelog for ndx-pose

## ndx-pose 0.2.2 (May 7, 2025)

The 0.2.0 schema has not changed, but the surrounding infrastructure and Python API has changes:

### Bug fixes
- Treat deprecation warnings as warnings instead of errors. @pauladkisson (#41)

### Minor updates
- Replaced `pip install -r requirements-dev.txt` with `pip install ".[dev]"` and added `pip install ".[test]"` and `pip install ".[docs]"`. @rly (#44)
- Replaced `pip install -r requirements-min.txt` with `pip install ".[min-reqs]"`. @rly (#44)
- Updated GitHub Actions workflows. @rly (#44)
 
## ndx-pose 0.2.1 (September 26, 2024)

### Bug fixes
- Fixed missing files in source distribution (`.tar.gz`). @rly (#35)

## ndx-pose 0.2.0 (September 26, 2024)

### Major changes
- Added support for storing training data in the new `PoseTraining` neurodata type and other new types.
  @roomrys, @CBroz1, @rly, @talmo, @eberrigan (#7, #21, #24)
- Deprecated the `nodes` and `edges` constructor arguments and fields from the `PoseEstimation` neurodata type to
  support linking to the same `Skeleton` object from a `PoseEstimation` object and pose training data.
  @rly (#7, #24, #33)
  - Data from `ndx-pose` versions before `0.2.0` can still be read; a temporary `Skeleton` object with name 
    "subject" will be created and the `nodes` and `edges` fields will be populated with the data from 
    the `PoseEstimation` object.
  - Instead of using the `nodes` and `edges` fields, you should now use a `Skeleton` object:
    - Create a `Skeleton` object with those nodes and edges.
    - Create a `Skeletons` container object and pass the `Skeleton` object to that.
    - Add the `Skeletons` object to your "behavior" processing module (at the same level as the `PoseEstimation` object).
    - Pass the `Skeleton` object to the `PoseEstimation` constructor.
- Added ability to link a `PoseEstimation` object to one or more camera devices. If the number of original videos,
  labeled videos, or dimensions does not equal the number of camera devices when creating the object not from a file,
  a `DeprecationWarning` will be raised. @rly (#33)

### Minor changes
- Made `PoseEstimation.confidence` optional. @h-mayorquin (#11)

## ndx-pose 0.1.1 (January 26, 2022)

The schema is unchanged, but the Python package has been updated.

### Bug fix
- Remove hdmf-docutils as an installation dependency. @rly (#6)

## ndx-pose 0.1.0 (January 26, 2022)

Initial release including `PoseEstimation` and `PoseEstimationSeries` neurodata types.
