def test_example_usage():
    import datetime
    import numpy as np
    from pynwb import NWBFile, NWBHDF5IO
    from ndx_pose import (
        PoseEstimationSeries,
        PoseEstimation,
        Skeleton,
        SkeletonInstance,
        TrainingFrame,
        PoseTraining,
        Skeletons,
        TrainingFrames,
        SourceVideos,
    )
    from pynwb.image import ImageSeries

    # initialize an NWBFile object
    nwbfile = NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    # create a device for the camera
    camera1 = nwbfile.create_device(
        name="camera1",
        description="camera for recording behavior",
        manufacturer="my manufacturer",
    )

    # a PoseEstimationSeries represents the estimated position of a single marker.
    # in this example, we have three PoseEstimationSeries: one for the body and one for each front paw.
    data = np.random.rand(100, 2)  # num_frames x (x, y) but can be (x, y, z)
    timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
    confidence = np.random.rand(100)  # a confidence value for every frame
    reference_frame = "(0,0,0) corresponds to ..."

    # note the double underscore in "confidence__definition" because this is a property of the "confidence" field
    confidence_definition = "Softmax output of the deep neural network."

    front_left_paw = PoseEstimationSeries(
        name="front_left_paw",
        description="Marker placed around fingers of front left paw.",
        data=data,
        unit="pixels",
        reference_frame=reference_frame,
        timestamps=timestamps,
        confidence=confidence,
        confidence_definition=confidence_definition,
    )

    data = np.random.rand(100, 2)  # num_frames x (x, y) but can be (x, y, z)
    confidence = np.random.rand(100)  # a confidence value for every frame
    body = PoseEstimationSeries(
        name="body",
        description="Marker placed on center of body.",
        data=data,
        unit="pixels",
        reference_frame=reference_frame,
        timestamps=front_left_paw,  # link to timestamps of front_left_paw so we don't have to duplicate them
        confidence=confidence,
        confidence_definition=confidence_definition,
    )

    data = np.random.rand(100, 2)  # num_frames x (x, y) but can be (x, y, z)
    confidence = np.random.rand(100)  # a confidence value for every frame
    front_right_paw = PoseEstimationSeries(
        name="front_right_paw",
        description="Marker placed around fingers of front right paw.",
        data=data,
        unit="pixels",
        reference_frame=reference_frame,
        timestamps=front_left_paw,  # link to timestamps of front_left_paw so we don't have to duplicate them
        confidence=confidence,
        confidence_definition=confidence_definition,
    )

    # store all PoseEstimationSeries in a list
    pose_estimation_series = [front_left_paw, body, front_right_paw]

    # create a skeleton that defines the relationship between the markers
    skeleton = Skeleton(
        name="subject1",
        nodes=["front_left_paw", "body", "front_right_paw"],
        # edge between front left paw and body, edge between body and front right paw.
        # the values are the indices of the nodes in the nodes list.
        edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
    )

    # create a PoseEstimation object that represents the estimated positions of the front paws and body
    # from DLC and references the original video simultaneously recorded from one camera and the labeled
    # video that was generated by DLC. multiple videos and cameras can be referenced.
    pose_estimation = PoseEstimation(
        pose_estimation_series=pose_estimation_series,
        description="Estimated positions of front paws using DeepLabCut.",
        original_videos=["path/to/camera1.mp4"],
        labeled_videos=["path/to/camera1_labeled.mp4"],
        dimensions=np.array([[640, 480]], dtype="uint16"),  # pixel dimensions of the video
        devices=[camera1],
        scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
        source_software="DeepLabCut",
        source_software_version="2.3.8",
        skeleton=skeleton,
    )

    # next, we specify the ground truth data that was used to train the pose estimation model.
    # this includes the training video and the ground truth annotations for each frame.
    # this is optional. if you don't have ground truth data, you can skip this step.

    # create an ImageSeries that represents the raw video that was used to train the pose estimation model
    training_video1 = ImageSeries(
        name="source_video",
        description="Training video used to train the pose estimation model.",
        unit="NA",
        format="external",
        external_file=["path/to/camera1.mp4"],
        dimension=[640, 480],
        starting_frame=[0],
        rate=30.0,
    )

    # create 50 ground truth instances of the skeleton at slightly random positions.
    # in this example, each node is visible on every frame.
    # the mapping of index in node_locations and node_visibilities to label is defined by the skeleton.

    # the node locations are the (x, y) coordinates of each node in the skeleton.
    # the order of the nodes is defined by the skeleton.
    node_locations = np.array(
        [
            [10, 10],  # front_left_paw
            [20, 20],  # body
            [30, 10],  # front_right_paw
        ]
    )

    skeleton_instances = []
    for i in range(50):
        # add some noise to the node locations from the location on the previous frame
        node_locations = node_locations + np.random.rand(3, 2)
        instance = SkeletonInstance(
            id=np.uint(i),
            node_locations=node_locations,
            node_visibility=[
                True,  # front_left_paw
                True,  # body
                True,  # front_right_paw
            ],
            skeleton=skeleton,  # link to the skeleton
        )
        skeleton_instances.append(instance)

    # create 50 training frames using the training video and the skeleton instances.
    # the skeleton instances start with video frame 0 and end with video frame 49.
    training_frames_list = []
    for i in range(50):
        # names must be unique within a PoseTraining object (we will add them to a PoseTraining object below)
        training_frame = TrainingFrame(
            name="frame_{}".format(i),
            annotator="Bilbo Baggins",
            skeleton_instance=skeleton_instances[i],
            source_video=training_video1,
            source_video_frame_index=np.uint(i),
        )
        training_frames_list.append(training_frame)

    # store the skeletons, training frames, and source videos in their corresponding grouping objects
    skeletons = Skeletons(skeletons=[skeleton])
    training_frames = TrainingFrames(training_frames=training_frames_list)
    source_videos = SourceVideos(image_series=[training_video1])

    # store the skeletons group, training frames group, and source videos group in a PoseTraining object
    pose_training = PoseTraining(
        skeletons=skeletons,
        training_frames=training_frames,
        source_videos=source_videos,
    )

    # create a "behavior" processing module to store the PoseEstimation and PoseTraining objects
    behavior_pm = nwbfile.create_processing_module(
        name="behavior",
        description="processed behavioral data",
    )
    behavior_pm.add(pose_estimation)
    behavior_pm.add(pose_training)

    # write the NWBFile to disk
    path = "test_pose.nwb"
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)

    # read the NWBFile from disk and print out the PoseEstimation and PoseTraining objects
    # as well as the first training frame
    with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        read_pe = read_nwbfile.processing["behavior"]["PoseEstimation"]
        print(read_pe)
        read_pt = read_nwbfile.processing["behavior"]["PoseTraining"]
        print(read_pt)
        print(read_pt.training_frames["frame_0"])
