# -*- coding: utf-8 -*-
import os.path

from pynwb.spec import export_spec, NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBLinkSpec, NWBNamespaceBuilder


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        doc='NWB extension to store pose estimation data',
        name='ndx-pose',
        version='0.1.1',
        author=['Ryan Ly', 'Ben Dichter', 'Alexander Mathis', 'Liezl Maree'],
        contact=['rly@lbl.gov', 'bdichter@lbl.gov', 'alexander.mathis@epfl.ch', 'lmaree@salk.edu'],
    )

    ns_builder.include_type('SpatialSeries', namespace='core')
    ns_builder.include_type('NWBDataInterface', namespace='core')

    identity = NWBGroupSpec(
        neurodata_type_def='Identity',
        neurodata_type_inc='NWBDataInterface',
        doc='Group that holds identity for a unique Instance.',
        default_name='Identity',
        attributes=[
            NWBAttributeSpec(
                name='name',
                doc='Unique name associated with the instance.',
                dtype='text',
            ),
            NWBAttributeSpec(
                name='description',
                doc='Description of unique ID. What makes the instance unique?',
                dtype='text',
            ),
        ],
    )

    skeleton = NWBGroupSpec(
        neurodata_type_def='Skeleton',
        neurodata_type_inc='NWBDataInterface',
        doc='Group that holds node and edge data for defining parts of a pose and their connections to one another.',
        default_name='Skeleton',
        attributes=[
            NWBAttributeSpec(
                name='id',
                doc='Unique id associated with the skeleton.',
                dtype='uint8',
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name='nodes',
                doc=('Array of body part names corresponding to the names of the PoseEstimationSeries objects or '
                     'PoseTraining objects.'),
                dtype='text',
                dims=['num_body_parts'],
                shape=[None],
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
        ]
    )

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
                neurodata_type_inc='Skeleton',
                doc='Layout of body part locations and connections.',
                quantity='*',
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
        ],
        # TODO: collections of multiple links is currently buggy in PyNWB/HDMF
        # links=[
        #     NWBLinkSpec(
        #         target_type='Device',
        #         doc='Cameras used to record the videos.',
        #         quantity='*',
        #     ),
        # ],
    )

    training_frame = NWBGroupSpec(
        neurodata_type_def='TrainingFrame',
        neurodata_type_inc='NWBDataInterface',
        doc='Group that holds ground-truth position data for all instances in a single frame.',
        default_name='TrainingFrame',
        groups=[
            NWBGroupSpec(
                neurodata_type_inc='Instance',
                doc='Position data for all instances in a single training frame.',
                quantity='*',
            ),
            NWBGroupSpec(
                name='source_video',
                doc='Path to original video file and frame used.',
                quantity='?',
                attributes=[
                    NWBAttributeSpec(
                        name='path',
                        doc='Path to original video file.',
                        dtype='text',
                        required=False,
                    ),
                    NWBAttributeSpec(
                        name='frame_index',
                        doc='Frame index of TrainingFrame in original video file.',
                        dtype='uint8',
                        required=False,
                    ),
                ],
            ),
            NWBGroupSpec(
                neurodata_type_inc='Image',
                name='source_frame',
                doc='Image frame used for training (stored either internally or externally).',
                quantity='1',
            ),
        ],
        attributes=[
            NWBAttributeSpec(
                name='annotator',
                doc='Name of annotator who labeled the TrainingFrame.',
                dtype='text',
                required=False,
            ),
        ],
    )

    instance = NWBGroupSpec(
        neurodata_type_def='Instance',
        neurodata_type_inc='NWBDataInterface',
        doc=('Group that holds position data for single subject in a single frame, computed from the same '
            'video with the same tool/algorithm.'),
        default_name='Instance',
        attributes=[
            NWBAttributeSpec(
                name='skeleton_id',
                doc='ID of skeleton used.',
                dtype='uint8',
                required=True,
            ),
            NWBAttributeSpec(
                name='instance_id',
                doc='ID used to differentiate instances.',
                dtype='text',
                required=False,
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name='node_locations',
                doc=('Locations (x, y, visible) or (x, y, z, visible) of nodes for single instance in single frame.'),
                dtype='float',
                # XXX(LM): Should visibility be a part of this array?
                dims=[['num_body_parts', 'x, y, visible'], ['num_body_parts', 'x, y, z, visible']],
                shape=[[None, 3], [None, 4]],
                quantity='?',
            ),
        ],
    )

    pose_training = NWBGroupSpec(
        neurodata_type_def='PoseTraining',
        neurodata_type_inc='NWBDataInterface',
        doc='Group that holds images, ground-truth annotations, and metadata for training a pose estimator.',
        default_name='PoseTraining',
        groups=[
            NWBGroupSpec(
                neurodata_type_inc='Skeleton',
                doc='Skeletons used in project where each skeleton corresponds to a unique morphology.',
                quantity='*',
            ),
            NWBGroupSpec(
                neurodata_type_inc='Identity',
                doc='Unique identifier used to differentiate instances.',
                quantity='*',
            ),
            NWBGroupSpec(
                neurodata_type_inc='TrainingFrame',
                doc='Frames and ground-truth annotations for training a pose estimator.',
                quantity='*',
            ),
        ],
    )

    new_data_types = [skeleton, identity, pose_estimation_series, pose_estimation, training_frame, instance, pose_training]

    # export the spec to yaml files in the spec folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spec'))
    export_spec(ns_builder, new_data_types, output_dir)
    print('Spec files generated. Please make sure to rerun `pip install .` to load the changes.')


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()
