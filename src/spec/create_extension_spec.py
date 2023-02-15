# -*- coding: utf-8 -*-
import os.path

from pynwb.spec import export_spec, NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBLinkSpec, NWBNamespaceBuilder


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        doc='NWB extension to store pose estimation data',
        name='ndx-pose',
        version='0.2.0',
        author=['Ryan Ly', 'Ben Dichter', 'Alexander Mathis'],
        contact=['rly@lbl.gov', 'bdichter@lbl.gov', 'alexander.mathis@epfl.ch'],
    )

    ns_builder.include_type('SpatialSeries', namespace='core')
    ns_builder.include_type('NWBDataInterface', namespace='core')

    pose_estimation_series = NWBGroupSpec(
        neurodata_type_def='PoseEstimationSeries',
        neurodata_type_inc='SpatialSeries',
        doc='Estimated position (x, y) or (x, y, z) of a body part over time.',
        datasets=[
            NWBDatasetSpec(
                name='data',
                doc='Estimated position (x, y) or (x, y, z).',
                dtype='float32',
                dims=[['num_frames', 'x, y'], ['num_frames', 'x, y, z']],
                shape=[[None, 2], [None, 3]],
                attributes=[
                    NWBAttributeSpec(
                        name='unit',
                        dtype='text',
                        default_value='pixels',
                        doc=("Base unit of measurement for working with the data. The default value "
                             "is 'pixels'. Actual stored values are not necessarily stored in these units. "
                             "To access the data in these units, multiply 'data' by 'conversion'."),
                        required=True,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name='confidence',
                doc='Confidence or likelihood of the estimated positions, scaled to be between 0 and 1.',
                dtype='float32',
                dims=['num_frames'],
                shape=[None],
                quantity='?',
                attributes=[
                    NWBAttributeSpec(
                        name='definition',
                        dtype='text',
                        doc=("Description of how the confidence was computed, e.g., "
                             "'Softmax output of the deep neural network'."),
                        required=False,
                    ),
                ],
            ),
        ],
    )

    pose_estimation = NWBGroupSpec(
        neurodata_type_def='PoseEstimation',
        neurodata_type_inc='NWBDataInterface',
        doc=('Group that holds estimated position data for multiple body parts, computed from the same video with '
             'the same tool/algorithm. The timestamps of each child PoseEstimationSeries type should be the same.'),
        default_name='PoseEstimation',
        groups=[
            NWBGroupSpec(
                neurodata_type_inc='PoseEstimationSeries',
                doc=('Estimated position data for each body part. Deprecated in version 0.2.0. Use the "positions" '
                     'group instead.'),
                quantity='*',
            ),
            NWBGroupSpec(
                name="pose_estimation_series",
                doc="Estimated position data for each body part.",
                groups=[
                    NWBGroupSpec(
                        neurodata_type_inc='PoseEstimationSeries',
                        doc='Estimated position data for each body part.',
                        quantity='*',
                    )
                ],
                quantity='?',
            ),
            NWBGroupSpec(
                name="original_videos_series",
                doc="Links to the original video files.",
                links=[
                    NWBLinkSpec(
                        target_type='ImageSeries',
                        doc='Links to the original video files.',
                        quantity='*',
                    ),
                ],
                quantity='?',
            ),
            NWBGroupSpec(
                name="labeled_videos_series",
                doc="The labeled videos. The number of files should equal the number of original videos.",
                datasets=[
                    NWBDatasetSpec(
                        neurodata_type_inc="ImageSeries",
                        doc='The labeled videos. The number of files should equal the number of original videos.',
                        quantity='*',
                    ),
                ],
                quantity='?',
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name='description',
                doc='Description of the pose estimation procedure and output.',
                dtype='text',
                quantity='?',
            ),
            NWBDatasetSpec(
                name='original_videos',
                doc=('Paths to the original video files. The number of files should equal the number of '
                     'camera devices. Deprecated in version 0.2.0. Use ImageSeries objects in original_videos_series '
                     'instead'),
                dtype='text',
                dims=['num_files'],
                shape=[None],
                quantity='?',
            ),
            NWBDatasetSpec(
                name='labeled_videos',
                doc=('Paths to the labeled video files. The number of files should equal the number of '
                     'camera devices. Deprecated in version 0.2.0. Use ImageSeries objects in labeled_videos_series '
                     'instead'),
                dtype='text',
                dims=['num_files'],
                shape=[None],
                quantity='?',
            ),
            NWBDatasetSpec(
                name='dimensions',
                doc=('Dimensions of each labeled video file. Deprecated in version 0.2.0. '
                     'Use "dimension" in original_videos_series instead.'),
                dtype='uint8',
                dims=['num_files', 'width, height'],
                shape=[None, 2],
                quantity='?',
            ),
            NWBDatasetSpec(
                name='scorer',
                doc='Name of the scorer / algorithm used.',
                dtype='text',
                quantity='?',
            ),
            NWBDatasetSpec(
                name='source_software',
                doc='Name of the software tool used. Specifying the version attribute is strongly encouraged.',
                dtype='text',
                quantity='?',
                attributes=[
                    NWBAttributeSpec(
                        name='version',
                        doc='Version string of the software tool used.',
                        dtype='text',
                        required=False,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name='nodes',
                doc=('Array of body part names corresponding to the names of the PoseEstimationSeries objects within '
                     'this group.'),
                dtype='text',
                dims=['num_body_parts'],
                shape=[None],
                quantity='?',
            ),
            NWBDatasetSpec(
                name='edges',
                doc=("Array of pairs of indices corresponding to edges between nodes. Index values correspond to row "
                     "indices of the 'nodes' dataset. Index values use 0-indexing."),
                dtype='uint8',
                dims=['num_edges', 'nodes_index, nodes_index'],
                shape=[None, 2],
                quantity='?',
            ),
        ],
    )

    new_data_types = [pose_estimation_series, pose_estimation]

    # export the spec to yaml files in the spec folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spec'))
    export_spec(ns_builder, new_data_types, output_dir)
    print('Spec files generated. Please make sure to rerun `pip install .` to load the changes.')


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()
