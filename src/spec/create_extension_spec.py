# -*- coding: utf-8 -*-
import os.path

from pynwb.spec import export_spec, NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBLinkSpec, NWBNamespaceBuilder


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        doc='NWB extension to store single or multi-animal pose tracking.',
        name='ndx-pose',
        version='0.2.0',
        author=['Ryan Ly', 'Ben Dichter', 'Alexander Mathis', 'Talmo Pereira'],
        contact=['rly@lbl.gov', 'bdichter@lbl.gov', 'alexander.mathis@epfl.ch', 'talmo@princeton.edu'],
    )

    ns_builder.include_type('SpatialSeries', namespace='core')
    ns_builder.include_type('TimeSeries', namespace='core')
    ns_builder.include_type('NWBDataInterface', namespace='core')
    ns_builder.include_type('NWBContainer', namespace='core')

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

    pose_grouping_series = NWBGroupSpec(
        neurodata_type_def='PoseGroupingSeries',
        neurodata_type_inc='TimeSeries',
        doc='Instance-level part grouping timeseries for the individual animal. This contains metadata of the part grouping procedure for multi-animal pose trackers.',
        datasets=[
            NWBDatasetSpec(
                name='score',
                doc='Score of the grouping approach that associated all of the keypoints to the same animal within the frame.',
                dtype='float32',
                dims=['num_frames'],
                shape=[None],
                attributes=[
                    NWBAttributeSpec(
                        name='definition',
                        dtype='text',
                        doc=("Description of how the score was computed, e.g., "
                             "'Centroid localization score' or 'Part matching score'."),
                        required=True,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name='location',
                doc='Animal location for two-stage (top-down) multi-animal models, e.g., centroid or bounding box.',
                dtype='float32',
                dims=[['num_frames', 'x, y'], ['num_frames', 'x, y, z'], ['num_frames', 'x1, y1, x2, y2'], ['num_frames', 'x1, y1, z1, x2, y2, z2']],
                shape=[[None, 2], [None, 3], [None, 4], [None, 6]],
                attributes=[
                    NWBAttributeSpec(
                        name='definition',
                        dtype='text',
                        doc=("Description of the type of localization, e.g., "
                             "'Centroid' or 'Bounding box'."),
                        required=True,
                    ),
                ],
            ),
        ],
    )


    animal_identity_series = NWBGroupSpec(
        neurodata_type_def='AnimalIdentitySeries',
        neurodata_type_inc='TimeSeries',
        doc='Identity of the animal predicted by a tracking or re-ID algorithm in multi-animal experiments.',
        datasets=[
            NWBDatasetSpec(
                name='score',
                doc='Score of the identity assignment approach that associated all of the keypoints to the same animal over frames.',
                dtype='float32',
                dims=['num_frames'],
                shape=[None],
                attributes=[
                    NWBAttributeSpec(
                        name='definition',
                        dtype='text',
                        doc=("Description of how the score was computed, e.g., "
                             "'MOT tracking score' or 'ID classification probability'."),
                        required=True,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name='name',
                doc='Unique animal identifier, track label, or class name used to identify this animal in the experiment.',
                dtype='text',
                quantity='?',
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
                doc='Estimated position data for each body part.',
                quantity='*',
            ),
            NWBGroupSpec(
                neurodata_type_inc='PoseGroupingSeries',
                doc='Part grouping metadata for the individual in multi-animal experiments.',
                quantity='?',
            ),
            NWBGroupSpec(
                neurodata_type_inc='AnimalIdentitySeries',
                doc='Predicted identity of the individual in multi-animal experiments.',
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
                doc='Paths to the original video files. The number of files should equal the number of camera devices.',
                dtype='text',
                dims=['num_files'],
                shape=[None],
                quantity='?',
            ),
            NWBDatasetSpec(
                name='labeled_videos',
                doc='Paths to the labeled video files. The number of files should equal the number of camera devices.',
                dtype='text',
                dims=['num_files'],
                shape=[None],
                quantity='?',
            ),
            NWBDatasetSpec(
                name='dimensions',
                doc='Dimensions of each labeled video file.',
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
                doc=('Array of body part names corresponding to the names of the SpatialSeries objects within this '
                     'group.'),
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
        # TODO: collections of multiple links is currently buggy in PyNWB/HDMF
        # links=[
        #     NWBLinkSpec(
        #         target_type='Device',
        #         doc='Camera(s) used to record the videos.',
        #         quantity='*',
        #     ),
        # ],
    )

    new_data_types = [pose_estimation_series, pose_estimation, pose_grouping_series, animal_identity_series]

    # export the spec to yaml files in the spec folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spec'))
    export_spec(ns_builder, new_data_types, output_dir)


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()
