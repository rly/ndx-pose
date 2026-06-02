# Changelog for ndx-pose

## ndx-pose 0.3.0 (June 2, 2026)

### Schema changes
- Added optional `source_video` and `labeled_video` links on `PoseEstimation` that reference an `ImageSeries`
  in the NWBFile, providing a formal alternative to the fragile string paths in `original_videos` and
  `labeled_videos`. @h-mayorquin (#56)

### Bug fixes
- Tests were updated to account for a change in the format of warnings from HDMF 4.1.0. @rly (#49)
- Improved documentation. @h-mayorquin (#50)
- Fixed the docstring of the `PoseEstimation` `name` constructor argument, which duplicated the `description`
  docstring. @rly (#58)
- `PoseEstimation.nodes` and `PoseEstimation.edges` now raise an informative error instead of an `AttributeError`
  when the object has no `Skeleton`. @rly (#58)
- Fixed `mock_source_frame` to generate a name when none is provided, instead of passing `None` to `RGBImage`. @rly (#58)
- Documented the `source_video` and `labeled_video` links on `PoseEstimation` in the README diagrams and the
  pose-estimation example, and corrected the `source_software` field name (previously shown as `scorer_software`).
  @rly (#59)

### Minor updates
- Added testing on Python 3.14 and the corresponding PyPI classifier. @rly (#58)
- Updated copyright dates to 2026 and aligned the author lists in `LICENSE.txt` and the docs with `pyproject.toml`.
  @rly (#58)
- Removed the obsolete `importlib_resources` import fallback and added the missing `ndx_pose.testing` package init.
  @rly (#58)
- Removed placeholder project URLs from `pyproject.toml`. @rly (#58)
- Updated GitHub Actions to current major versions (`checkout` v6, `setup-python` v6, `ruff-action` v4,
  `codecov-action` v6) and switched the single-version workflows to Python 3.14. @rly (#59)
- Removed the deprecated `sphinx_rtd_theme.get_html_theme_path()` call from the docs config and registered the
  theme via `extensions`. @rly (#59)
- Regenerated the spec YAML files with the latest `ruamel.yaml`, which changed the line wrapping of the docstrings.
  The schema content is unchanged. @rly (#60)
- Documented the development installation (`pip install -e . --group dev`) in the README. @rly (#60)


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
