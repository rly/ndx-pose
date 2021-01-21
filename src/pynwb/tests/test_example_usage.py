def test_example_usage():
    import datetime
    import numpy as np
    from pynwb import NWBFile, NWBHDF5IO
    from ndx_pose import PoseEstimationSeries, PoseEstimation

    nwbfile = NWBFile(
        session_description='session_description',
        identifier='identifier',
        session_start_time=datetime.datetime.now(datetime.timezone.utc)
    )

    data = np.random.rand(100, 3)  # num_frames x (x, y, z)
    timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
    confidence = np.random.rand(100)  # a confidence value for every frame
    front_left_paw = PoseEstimationSeries(
        name='front_left_paw',
        description='Marker placed around fingers of front left paw.',
        data=data,
        unit='pixels',
        reference_frame='(0,0,0) corresponds to ...',
        timestamps=timestamps,
        confidence=confidence,
        confidence_definition='Softmax output of the deep neural network.',
    )

    data = np.random.rand(100, 2)  # num_frames x (x, y)
    timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
    confidence = np.random.rand(100)  # a confidence value for every frame
    front_right_paw = PoseEstimationSeries(
        name='front_right_paw',
        description='Marker placed around fingers of front right paw.',
        data=data,
        unit='pixels',
        reference_frame='(0,0,0) corresponds to ...',
        timestamps=timestamps,
        confidence=confidence,
        confidence_definition='Softmax output of the deep neural network.',
    )

    pose_estimation_series = [front_left_paw, front_right_paw]

    pe = PoseEstimation(
        pose_estimation_series=pose_estimation_series,
        description='Estimated positions of front paws using DeepLabCut.',
        original_videos=['camera1.mp4', 'camera2.mp4'],
        labeled_videos=['camera1_labeled.mp4', 'camera2_labeled.mp4'],
        dimensions=[[640, 480], [1024, 768]],
        scorer='DLC_resnet50_openfieldOct30shuffle1_1600',
        source_software='DeepLabCut',
        source_software_version='2.2b8',
        nodes=['front_left_paw', 'front_right_paw'],
        edges=[[0, 1]],
        # devices=[self.nwbfile.devices['camera1'], self.nwbfile.devices['camera2']],
    )

    behavior_pm = nwbfile.create_processing_module(
        name='behavior',
        description='processed behavioral data'
    )
    behavior_pm.add(pe)

    path = 'test_pose.nwb'
    with NWBHDF5IO(path, mode='w') as io:
        io.write(nwbfile)

    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        read_nwbfile = io.read()
        read_pe = read_nwbfile.processing['behavior']['PoseEstimation']
        print(read_pe)
