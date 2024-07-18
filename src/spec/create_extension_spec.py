# -*- coding: utf-8 -*-
import os.path

from pynwb.spec import (
    export_spec,
    NWBAttributeSpec,
    NWBDatasetSpec,
    NWBGroupSpec,
    NWBLinkSpec,
    NWBNamespaceBuilder,
)


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        doc="NWB extension to store pose estimation data",
        name="ndx-pose",
        version="0.2.0",
        author=[
            "Ryan Ly",
            "Ben Dichter",
            "Alexander Mathis",
            "Liezl Maree",
            "Chris Brozdowski",
            "Heberto Mayorquin",
            "Talmo Pereira",
            "Elizabeth Berrigan",
        ],
        contact=[
            "rly@lbl.gov",
            "bdichter@lbl.gov",
            "alexander.mathis@epfl.ch",
            "lmaree@salk.edu",
            "cbroz@datajoint.com",
            "h.mayorquin@gmail.com",
            "talmo@salk.edu",
            "eberrigan@salk.edu",
        ],
    )

    ns_builder.include_namespace(namespace="core")

    skeleton = NWBGroupSpec(
        neurodata_type_def="Skeleton",
        neurodata_type_inc="NWBDataInterface",
        doc=(
            "Group that holds node and edge data for defining parts of a pose and their connections to one another. "
            "Names should be unique in a file."
        ),
        datasets=[
            NWBDatasetSpec(
                name="nodes",
                doc=(
                    "Array of body part names corresponding to the names of the PoseEstimationSeries objects or "
                    "PoseTraining objects."
                ),
                dtype="text",
                dims=["num_body_parts"],
                shape=[None],
                quantity=1,
            ),
            NWBDatasetSpec(
                name="edges",
                doc=(
                    "Array of pairs of indices corresponding to edges between nodes. Index values correspond to row "
                    "indices of the 'nodes' dataset. Index values use 0-indexing."
                ),
                dtype="uint8",
                dims=["num_edges", "nodes_index, nodes_index"],
                shape=[None, 2],
                quantity="?",
            ),
        ],
        links=[
            NWBLinkSpec(
                doc="The Subject object in the NWB file, if this Skeleton corresponds to the Subject.",
                target_type="Subject",
                quantity="?",
            ),
        ],
    )

    skeletons = NWBGroupSpec(
        neurodata_type_def="Skeletons",
        neurodata_type_inc="NWBDataInterface",
        doc="Organizational group to hold skeletons.",
        default_name="Skeletons",
        groups=[
            NWBGroupSpec(
                neurodata_type_inc="Skeleton",
                doc="Skeleton used in project where each skeleton corresponds to a unique morphology.",
                quantity="*",
            ),
        ],
    )

    pose_estimation_series = NWBGroupSpec(
        neurodata_type_def="PoseEstimationSeries",
        neurodata_type_inc="SpatialSeries",
        doc="Estimated position (x, y) or (x, y, z) of a body part over time.",
        datasets=[
            NWBDatasetSpec(
                name="data",
                doc="Estimated position (x, y) or (x, y, z).",
                dtype="float32",
                dims=[["num_frames", "x, y"], ["num_frames", "x, y, z"]],
                shape=[[None, 2], [None, 3]],
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        dtype="text",
                        default_value="pixels",
                        doc=(
                            "Base unit of measurement for working with the data. The default value "
                            "is 'pixels'. Actual stored values are not necessarily stored in these units. "
                            "To access the data in these units, multiply 'data' by 'conversion'."
                        ),
                        required=True,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name="confidence",
                doc="Confidence or likelihood of the estimated positions, scaled to be between 0 and 1.",
                dtype="float32",
                dims=["num_frames"],
                shape=[None],
                attributes=[
                    NWBAttributeSpec(
                        name="definition",
                        dtype="text",
                        doc=(
                            "Description of how the confidence was computed, e.g., "
                            "'Softmax output of the deep neural network'."
                        ),
                        required=False,
                    ),
                ],
            ),
        ],
    )

    pose_estimation = NWBGroupSpec(
        neurodata_type_def="PoseEstimation",
        neurodata_type_inc="NWBDataInterface",
        doc=(
            "Group that holds estimated position data for multiple body parts, computed from the same video with "
            "the same tool/algorithm. The timestamps of each child PoseEstimationSeries type should be the same."
        ),
        default_name="PoseEstimation",
        groups=[
            NWBGroupSpec(
                neurodata_type_inc="PoseEstimationSeries",
                doc="Estimated position data for each body part.",
                quantity="*",
            ),
        ],
        links=[
            NWBLinkSpec(
                doc="Layout of body part locations and connections.",
                target_type="Skeleton",
                quantity="?",
            ),
            NWBLinkSpec(
                target_type="Device",
                doc="Cameras used to record the videos.",
                quantity="*",
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name="description",
                doc="Description of the pose estimation procedure and output.",
                dtype="text",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="original_videos",
                doc="Paths to the original video files. The number of files should equal the number of camera devices.",
                dtype="text",
                dims=["num_files"],
                shape=[None],
                quantity="?",
            ),
            NWBDatasetSpec(
                name="labeled_videos",
                doc="Paths to the labeled video files. The number of files should equal the number of camera devices.",
                dtype="text",
                dims=["num_files"],
                shape=[None],
                quantity="?",
            ),
            NWBDatasetSpec(
                name="dimensions",
                doc="Dimensions of each labeled video file.",
                dtype="uint8",
                dims=["num_files", "width, height"],
                shape=[None, 2],
                quantity="?",
            ),
            NWBDatasetSpec(
                name="scorer",
                doc="Name of the scorer / algorithm used.",
                dtype="text",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="source_software",
                doc="Name of the software tool used. Specifying the version attribute is strongly encouraged.",
                dtype="text",
                quantity="?",
                attributes=[
                    NWBAttributeSpec(
                        name="version",
                        doc="Version string of the software tool used.",
                        dtype="text",
                        required=False,
                    ),
                ],
            ),
        ],
    )

    skeleton_instance = NWBGroupSpec(
        neurodata_type_def="SkeletonInstance",
        neurodata_type_inc="NWBDataInterface",
        doc="Group that holds ground-truth pose data for a single instance of a skeleton in a single frame.",
        default_name="skeleton_instance",
        links=[
            NWBLinkSpec(
                doc="Layout of body part locations and connections.",
                target_type="Skeleton",
            ),
        ],
        attributes=[  # TODO since the link to the Skeleton is required, is this necessary?
            NWBAttributeSpec(
                name="id",
                doc="ID used to differentiate skeleton instances.",
                dtype="uint8",
                required=False,
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name="node_locations",
                doc="Locations (x, y) or (x, y, z) of nodes for single instance in single frame.",
                dtype="float",
                dims=[["num_body_parts", "x, y"], ["num_body_parts", "x, y, z"]],
                shape=[[None, 2], [None, 3]],
                quantity=1,
            ),
            NWBDatasetSpec(
                name="node_visibility",
                doc=(
                    "Markers for node visibility where true corresponds to a visible node and false corresponds to "
                    "an occluded node."
                ),
                dtype="bool",
                dims=["num_body_parts"],
                shape=[None],
                quantity="?",
            ),
        ],
    )

    skeleton_instances = NWBGroupSpec(
        neurodata_type_def="SkeletonInstances",
        neurodata_type_inc="NWBDataInterface",
        doc="Organizational group to hold skeleton instances. This is meant to be used within a TrainingFrame.",
        default_name="skeleton_instances",
        groups=[
            NWBGroupSpec(
                neurodata_type_inc="SkeletonInstance",
                doc="Ground-truth position data for a single instance of a skeleton in a single training frame.",
                quantity="*",
            ),
        ],
    )

    source_videos = NWBGroupSpec(
        neurodata_type_def="SourceVideos",
        neurodata_type_inc="NWBDataInterface",
        doc="Organizational group to hold source videos used for training.",
        # this is meant to be used in a PoseTraining object which will enforce this name
        default_name="source_videos",
        groups=[
            NWBGroupSpec(
                neurodata_type_inc="ImageSeries",
                doc="Video of training frames (stored internally or externally",
                quantity="*",
            ),
        ],
    )

    training_frame = NWBGroupSpec(
        neurodata_type_def="TrainingFrame",
        neurodata_type_inc="NWBDataInterface",
        doc="Group that holds ground-truth position data for all instances of a skeleton in a single frame.",
        default_name="TrainingFrame",
        groups=[
            NWBGroupSpec(
                name="skeleton_instances",
                neurodata_type_inc="SkeletonInstances",
                doc="Position data for all instances of a skeleton in a single training frame.",
            ),
        ],
        attributes=[
            NWBAttributeSpec(
                name="annotator",
                doc="Name of annotator who labeled the TrainingFrame.",
                dtype="text",
                required=False,
            ),
            NWBAttributeSpec(
                name="source_video_frame_index",
                doc=(
                    "Frame index of training frame in the original video `source_video`. "
                    "If provided, then `source_video` is required."
                ),
                dtype="uint8",
                required=False,
            ),
            # TODO add inspector check that either both source_video and source_video_frame_index are provided or
            # neither are provided
        ],
        links=[
            NWBLinkSpec(
                name="source_video",
                target_type="ImageSeries",
                doc=(
                    "Link to an ImageSeries representing a video of training frames (stored internally or "
                    "externally). Required if `source_video_frame_index` is provided."
                ),
                quantity="?",
            ),
            NWBLinkSpec(
                name="source_frame",
                target_type="Image",
                doc=(
                    "Link to an internally stored image representing the training frame. The target Image "
                    "should be stored in an Images type in the file."
                ),
                quantity="?",
            ),
        ],
    )

    training_frames = NWBGroupSpec(
        neurodata_type_def="TrainingFrames",
        neurodata_type_inc="NWBDataInterface",
        doc="Organizational group to hold training frames.",
        # this is meant to be used in a PoseTraining object which will enforce this name
        default_name="training_frames",
        groups=[
            NWBGroupSpec(
                neurodata_type_inc="TrainingFrame",
                doc="Ground-truth position data for all instances of a skeleton in a single frame.",
                quantity="*",
            ),
        ],
    )

    pose_training = NWBGroupSpec(
        neurodata_type_def="PoseTraining",
        neurodata_type_inc="NWBDataInterface",
        doc="Group that holds source videos and ground-truth annotations for training a pose estimator.",
        default_name="PoseTraining",
        groups=[
            NWBGroupSpec(
                name="training_frames",
                neurodata_type_inc="TrainingFrames",
                doc="Organizational group to hold training frames.",
                quantity="?",
            ),
            NWBGroupSpec(
                name="source_videos",
                neurodata_type_inc="SourceVideos",
                doc="Organizational group to hold source videos used for training.",
                quantity="?",
            ),
        ],
    )

    new_data_types = [
        skeleton,
        pose_estimation_series,
        pose_estimation,
        training_frame,
        skeleton_instance,
        training_frames,
        skeleton_instances,
        source_videos,
        skeletons,
        pose_training,
    ]

    # export the spec to yaml files in the spec folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "spec"))
    export_spec(ns_builder, new_data_types, output_dir)
    print("Spec files generated. Please make sure to rerun `pip install .` to load the changes.")


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()
