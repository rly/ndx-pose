import datetime
import numpy as np

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.testing import TestCase
from pynwb.file import Subject

from pynwb.image import ImageSeries

from ndx_pose import (
    CameraCalibration,
    CameraView,
    MultiCameraPoseEstimation,
    PoseEstimationSeries,
    Skeleton,
    PoseEstimation,
    TrainingFrame,
    SkeletonInstance,
    PoseTraining,
    TrainingFrames,
    SourceVideos,
)
from ndx_pose.testing.mock.pose import (
    mock_CameraCalibration,
    mock_CameraView,
    mock_MultiCameraPoseEstimation,
    mock_PoseEstimationSeries,
    mock_SkeletonInstances,
    mock_source_video,
    mock_source_frame,
    mock_Skeleton,
    mock_SkeletonInstance,
    mock_TrainingFrame,
)

# NOTE Skeletons, TrainingFrames, SourceVideos are tested within PoseTraining but not separately tested


class TestPoseEstimationSeriesConstructor(TestCase):
    def test_constructor(self):
        data = np.random.rand(100, 3)  # num_frames x (x, y, z)
        timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
        confidence = np.random.rand(100)  # a confidence value for every frame
        pes = PoseEstimationSeries(
            name="front_left_paw",
            description="Marker placed around fingers of front left paw.",
            data=data,
            unit="pixels",
            reference_frame="(0,0,0) corresponds to ...",
            timestamps=timestamps,
            confidence=confidence,
            confidence_definition="Softmax output of the deep neural network.",
        )

        self.assertEqual(pes.name, "front_left_paw")
        self.assertEqual(pes.description, "Marker placed around fingers of front left paw.")
        np.testing.assert_array_equal(pes.data, data)
        self.assertEqual(pes.unit, "pixels")
        self.assertEqual(pes.reference_frame, "(0,0,0) corresponds to ...")
        np.testing.assert_array_equal(pes.timestamps, timestamps)
        np.testing.assert_array_equal(pes.confidence, confidence)
        self.assertEqual(pes.confidence_definition, "Softmax output of the deep neural network.")


class TestSkeleton(TestCase):
    def test_init(self):
        subject = Subject(subject_id="MOUSE001", species="Mus musculus")
        skeleton = Skeleton(
            name="subject1",
            nodes=["front_left_paw", "body", "front_right_paw"],
            # edge between front left paw and body, edge between body and front right paw.
            # the values are the indices of the nodes in the nodes list.
            edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
            subject=subject,
        )
        self.assertEqual(skeleton.name, "subject1")
        self.assertEqual(skeleton.nodes, ["front_left_paw", "body", "front_right_paw"])
        np.testing.assert_array_equal(skeleton.edges, np.array([[0, 1], [1, 2]], dtype="uint8"))
        self.assertIs(skeleton.subject, subject)

    def test_init_no_subject(self):
        skeleton = Skeleton(
            name="subject1",
            nodes=["front_left_paw", "body", "front_right_paw"],
            edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
        )
        self.assertIsNone(skeleton.subject)


class TestPoseEstimationConstructor(TestCase):
    def setUp(self):
        nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        nwbfile.create_device(name="camera1")
        nwbfile.create_device(name="camera2")

        self.nwbfile = nwbfile

    def test_constructor(self):
        """Test that the constructor for PoseEstimation sets values as expected."""
        front_left_paw = mock_PoseEstimationSeries(
            name="front_left_paw",
        )
        body = mock_PoseEstimationSeries(
            name="body",
            timestamps=front_left_paw,  # link to timestamps of front_left_paw
        )
        front_right_paw = mock_PoseEstimationSeries(
            name="front_right_paw",
            timestamps=front_left_paw,  # link to timestamps of front_left_paw
        )
        pose_estimation_series = [front_left_paw, body, front_right_paw]
        skeleton = mock_Skeleton(
            nodes=["front_left_paw", "body", "front_right_paw"],
            # edge between front left paw and body, edge between body and front right paw.
            # the values are the indices of the nodes in the nodes list.
            edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
        )

        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            description="Estimated positions of front paws using DeepLabCut.",
            original_videos=["camera1.mp4", "camera2.mp4"],
            labeled_videos=["camera1_labeled.mp4", "camera2_labeled.mp4"],
            dimensions=np.array([[640, 480], [1024, 768]], dtype="uint16"),
            devices=[self.nwbfile.devices["camera1"], self.nwbfile.devices["camera2"]],
            scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
            source_software="DeepLabCut",
            source_software_version="2.2b8",
            skeleton=skeleton,
        )

        self.assertEqual(pe.name, "PoseEstimation")
        self.assertEqual(len(pe.pose_estimation_series), 3)
        self.assertIs(pe.pose_estimation_series["front_left_paw"], pose_estimation_series[0])
        self.assertIs(pe.pose_estimation_series["body"], pose_estimation_series[1])
        self.assertIs(pe.pose_estimation_series["front_right_paw"], pose_estimation_series[2])
        self.assertEqual(pe.description, "Estimated positions of front paws using DeepLabCut.")
        self.assertEqual(pe.original_videos, ["camera1.mp4", "camera2.mp4"])
        self.assertEqual(pe.labeled_videos, ["camera1_labeled.mp4", "camera2_labeled.mp4"])
        np.testing.assert_array_equal(pe.dimensions, np.array([[640, 480], [1024, 768]], dtype="uint16"))
        self.assertEqual(len(pe.devices), 2)
        self.assertIs(pe.devices[0], self.nwbfile.devices["camera1"])
        self.assertIs(pe.devices[1], self.nwbfile.devices["camera2"])
        self.assertEqual(pe.scorer, "DLC_resnet50_openfieldOct30shuffle1_1600")
        self.assertEqual(pe.source_software, "DeepLabCut")
        self.assertEqual(pe.source_software_version, "2.2b8")
        self.assertIs(pe.skeleton, skeleton)

    def test_bad_device_link(self):
        front_left_paw = mock_PoseEstimationSeries(
            name="front_left_paw",
        )
        body = mock_PoseEstimationSeries(
            name="body",
            timestamps=front_left_paw,  # link to timestamps of front_left_paw
        )
        front_right_paw = mock_PoseEstimationSeries(
            name="front_right_paw",
            timestamps=front_left_paw,  # link to timestamps of front_left_paw
        )
        pose_estimation_series = [front_left_paw, body, front_right_paw]
        skeleton = mock_Skeleton(
            nodes=["front_left_paw", "body", "front_right_paw"],
            # edge between front left paw and body, edge between body and front right paw.
            # the values are the indices of the nodes in the nodes list.
            edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
        )
        camera1 = Device(name="camera1")
        camera2 = Device(name="camera2")

        msg = "All devices linked to from a PoseEstimation object must be added to the NWBFile first."
        with self.assertRaisesWith(ValueError, msg):
            PoseEstimation(
                pose_estimation_series=pose_estimation_series,
                description="Estimated positions of front paws using DeepLabCut.",
                original_videos=["camera1.mp4", "camera2.mp4"],
                labeled_videos=["camera1_labeled.mp4", "camera2_labeled.mp4"],
                dimensions=np.array([[640, 480], [1024, 768]], dtype="uint16"),
                devices=[camera1, camera2],
                scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
                source_software="DeepLabCut",
                source_software_version="2.2b8",
                skeleton=skeleton,
            )

    def test_constructor_nodes_edges(self):
        """Test the old constructor for PoseEstimation with nodes and edges."""
        front_left_paw = mock_PoseEstimationSeries(
            name="front_left_paw",
        )
        body = mock_PoseEstimationSeries(
            name="body",
            timestamps=front_left_paw,  # link to timestamps of front_left_paw
        )
        front_right_paw = mock_PoseEstimationSeries(
            name="front_right_paw",
            timestamps=front_left_paw,  # link to timestamps of front_left_paw
        )
        pose_estimation_series = [front_left_paw, body, front_right_paw]

        msg = (
            "The 'nodes' and 'edges' constructor arguments are deprecated. Please use the 'skeleton' argument instead. "
            "These will be removed in a future release."
        )
        with self.assertWarnsWith(DeprecationWarning, msg):
            pe = PoseEstimation(
                pose_estimation_series=pose_estimation_series,
                description="Estimated positions of front paws using DeepLabCut.",
                original_videos=["camera1.mp4", "camera2.mp4"],
                labeled_videos=["camera1_labeled.mp4", "camera2_labeled.mp4"],
                dimensions=np.array([[640, 480], [1024, 768]], dtype="uint16"),
                devices=[
                    self.nwbfile.devices["camera1"],
                    self.nwbfile.devices["camera2"],
                ],
                scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
                source_software="DeepLabCut",
                source_software_version="2.2b8",
                nodes=["front_left_paw", "body", "front_right_paw"],
                edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
            )
        self.assertEqual(pe.nodes, ["front_left_paw", "body", "front_right_paw"])
        np.testing.assert_array_equal(pe.edges, np.array([[0, 1], [1, 2]], dtype="uint8"))
        skeleton = Skeleton(
            name="subject",
            nodes=["front_left_paw", "body", "front_right_paw"],
            edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
        )
        self.assertEqual(pe.skeleton.name, skeleton.name)
        self.assertEqual(pe.skeleton.nodes, skeleton.nodes)
        np.testing.assert_array_equal(pe.skeleton.edges, skeleton.edges)

    def test_constructor_source_video(self):
        """Test that source_video link is set correctly."""
        source_video = ImageSeries(
            name="source_video",
            description="Source video for pose estimation.",
            unit="NA",
            format="external",
            external_file=["camera1.mp4"],
            dimension=[640, 480],
            starting_frame=[0],
            rate=30.0,
        )
        skeleton = mock_Skeleton()
        pose_estimation_series = [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]

        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            description="Estimated positions of front paws using DeepLabCut.",
            original_videos=["camera1.mp4"],
            devices=[self.nwbfile.devices["camera1"]],
            scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
            source_software="DeepLabCut",
            source_software_version="2.2b8",
            skeleton=skeleton,
            source_video=source_video,
        )
        self.assertIs(pe.source_video, source_video)

    def test_constructor_source_video_default_none(self):
        """Test that source_video defaults to None."""
        skeleton = mock_Skeleton()
        pose_estimation_series = [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]

        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            skeleton=skeleton,
        )
        self.assertIsNone(pe.source_video)

    def test_constructor_labeled_video(self):
        """Test that labeled_video link is set correctly."""
        labeled_video = ImageSeries(
            name="labeled_video",
            description="Labeled video produced from pose estimation.",
            unit="NA",
            format="external",
            external_file=["camera1_labeled.mp4"],
            dimension=[640, 480],
            starting_frame=[0],
            rate=30.0,
        )
        skeleton = mock_Skeleton()
        pose_estimation_series = [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]

        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            description="Estimated positions of front paws using DeepLabCut.",
            labeled_videos=["camera1_labeled.mp4"],
            devices=[self.nwbfile.devices["camera1"]],
            scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
            source_software="DeepLabCut",
            source_software_version="2.2b8",
            skeleton=skeleton,
            labeled_video=labeled_video,
        )
        self.assertIs(pe.labeled_video, labeled_video)

    def test_constructor_labeled_video_default_none(self):
        """Test that labeled_video defaults to None."""
        skeleton = mock_Skeleton()
        pose_estimation_series = [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]

        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            skeleton=skeleton,
        )
        self.assertIsNone(pe.labeled_video)


class TestSkeletonInstance(TestCase):
    def test_constructor(self):
        skeleton = mock_Skeleton(
            nodes=["left_eye", "right_eye"],
            edges=np.array([[0, 1]], dtype="uint8"),
        )

        node_locations = np.array(
            [
                [1, 5],
                [20, 30],
            ]
        )
        instance = SkeletonInstance(
            id=np.uint(10),
            node_locations=node_locations,
            node_visibility=[
                True,
                False,
            ],
            skeleton=skeleton,
        )
        self.assertEqual(instance.id, np.uint(10))
        np.testing.assert_array_equal(instance.node_locations, node_locations)
        self.assertEqual(instance.node_visibility, [True, False])
        self.assertIs(instance.skeleton, skeleton)


class TestTrainingFrame(TestCase):
    def test_constructor(self):
        skeleton_instances = mock_SkeletonInstances()
        source_video = mock_source_video(name="source_video")
        training_frame = TrainingFrame(
            name="frame0",
            annotator="Awesome Possum",
            skeleton_instances=skeleton_instances,
            source_video=source_video,
            source_video_frame_index=np.uint(0),
        )
        self.assertEqual(training_frame.name, "frame0")
        self.assertEqual(training_frame.annotator, "Awesome Possum")
        self.assertIs(training_frame.skeleton_instances, skeleton_instances)
        self.assertIs(training_frame.source_video, source_video)
        self.assertEqual(training_frame.source_video_frame_index, np.uint(0))


class TestTrainingFrameImage(TestCase):
    def test_constructor(self):
        skeleton_instances = mock_SkeletonInstances()
        source_frame = mock_source_frame(name="frame0_image")
        training_frame = TrainingFrame(
            name="frame0",
            annotator="Awesome Possum",
            skeleton_instances=skeleton_instances,
            source_frame=source_frame,
            source_video_frame_index=np.uint(0),
        )
        self.assertEqual(training_frame.name, "frame0")
        self.assertEqual(training_frame.annotator, "Awesome Possum")
        self.assertIs(training_frame.skeleton_instances, skeleton_instances)
        self.assertIs(training_frame.source_frame, source_frame)
        self.assertEqual(training_frame.source_video_frame_index, np.uint(0))


class TestPoseTraining(TestCase):
    def test_constructor(self):
        skeleton1 = mock_Skeleton(name="subject1")
        skeleton2 = mock_Skeleton(name="subject2")
        source_video = mock_source_video(name="source_video")
        sk1_instance10 = mock_SkeletonInstance(id=np.uint(10), skeleton=skeleton1)
        sk1_instance11 = mock_SkeletonInstance(id=np.uint(11), skeleton=skeleton1)
        sk1_instances = mock_SkeletonInstances(skeleton_instances=[sk1_instance10, sk1_instance11])
        sk1_training_frame = mock_TrainingFrame(
            name="skeleton1_frame10",
            skeleton_instances=sk1_instances,
            source_video=source_video,
            source_video_frame_index=np.uint(10),
        )
        sk2_instance10 = mock_SkeletonInstance(id=np.uint(10), skeleton=skeleton2)
        sk2_instance11 = mock_SkeletonInstance(id=np.uint(11), skeleton=skeleton2)
        sk2_instance12 = mock_SkeletonInstance(id=np.uint(12), skeleton=skeleton2)
        sk2_instances = mock_SkeletonInstances(skeleton_instances=[sk2_instance10, sk2_instance11, sk2_instance12])
        sk2_training_frame = mock_TrainingFrame(
            name="skeleton2_frame10",
            skeleton_instances=sk2_instances,
            source_video=source_video,
            source_video_frame_index=np.uint(10),
        )

        training_frames = TrainingFrames(training_frames=[sk1_training_frame, sk2_training_frame])
        source_videos = SourceVideos(image_series=[source_video])

        pose_training = PoseTraining(
            training_frames=training_frames,
            source_videos=source_videos,
        )
        self.assertEqual(len(pose_training.training_frames.training_frames), 2)
        self.assertIs(
            pose_training.training_frames.training_frames["skeleton1_frame10"],
            sk1_training_frame,
        )
        self.assertIs(
            pose_training.training_frames.training_frames["skeleton2_frame10"],
            sk2_training_frame,
        )
        self.assertEqual(len(pose_training.source_videos.image_series), 1)
        self.assertIs(pose_training.source_videos.image_series["source_video"], source_video)


class TestPoseTrainingImages(TestCase):
    def test_constructor(self):
        skeleton1 = mock_Skeleton(name="subject1")
        skeleton2 = mock_Skeleton(name="subject2")

        source_frame_10 = mock_source_frame(name="source_frame_10")
        sk1_instance10 = mock_SkeletonInstance(id=np.uint(10), skeleton=skeleton1)
        sk1_instance11 = mock_SkeletonInstance(id=np.uint(11), skeleton=skeleton1)
        sk1_instances = mock_SkeletonInstances(skeleton_instances=[sk1_instance10, sk1_instance11])
        sk1_training_frame = mock_TrainingFrame(
            name="frame10",
            skeleton_instances=sk1_instances,
            source_frame=source_frame_10,
            source_video_frame_index=np.uint(10),
        )

        source_frame_11 = mock_source_frame(name="source_frame_11")

        sk2_instance10 = mock_SkeletonInstance(id=np.uint(10), skeleton=skeleton2)
        sk2_instance11 = mock_SkeletonInstance(id=np.uint(11), skeleton=skeleton2)
        sk2_instance12 = mock_SkeletonInstance(id=np.uint(12), skeleton=skeleton2)
        sk2_instances = mock_SkeletonInstances(skeleton_instances=[sk2_instance10, sk2_instance11, sk2_instance12])
        sk2_training_frame = mock_TrainingFrame(
            name="frame11",
            skeleton_instances=sk2_instances,
            source_frame=source_frame_11,
            source_video_frame_index=np.uint(11),
        )

        training_frames = TrainingFrames(training_frames=[sk1_training_frame, sk2_training_frame])

        pose_training = PoseTraining(
            training_frames=training_frames,
        )
        self.assertEqual(len(pose_training.training_frames.training_frames), 2)
        self.assertIs(pose_training.training_frames.training_frames["frame10"], sk1_training_frame)
        self.assertIs(pose_training.training_frames.training_frames["frame11"], sk2_training_frame)
        self.assertIsNone(pose_training.source_videos)


class TestCameraCalibrationConstructor(TestCase):
    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.nwbfile.create_device(name="camera1")
        self.nwbfile.create_device(name="camera2")

    def test_constructor_full(self):
        K = np.tile(np.eye(3, dtype="float32"), (2, 1, 1))
        R = np.tile(np.eye(3, dtype="float32"), (2, 1, 1))
        t = np.zeros((2, 3), dtype="float32")
        d = np.zeros((2, 5), dtype="float32")
        devices = [self.nwbfile.devices["camera1"], self.nwbfile.devices["camera2"]]

        cal = CameraCalibration(
            intrinsic_matrix=K,
            rotation_matrix=R,
            translation_vector=t,
            distortion_coefficients=d,
            devices=devices,
        )

        np.testing.assert_array_equal(cal.intrinsic_matrix, K)
        np.testing.assert_array_equal(cal.rotation_matrix, R)
        np.testing.assert_array_equal(cal.translation_vector, t)
        np.testing.assert_array_equal(cal.distortion_coefficients, d)
        self.assertEqual(len(cal.devices), 2)
        self.assertIs(cal.devices[0], self.nwbfile.devices["camera1"])

    def test_constructor_intrinsics_only(self):
        K = np.tile(np.eye(3, dtype="float32"), (2, 1, 1))
        cal = CameraCalibration(intrinsic_matrix=K)
        np.testing.assert_array_equal(cal.intrinsic_matrix, K)
        self.assertIsNone(cal.rotation_matrix)
        self.assertIsNone(cal.translation_vector)
        self.assertIsNone(cal.distortion_coefficients)
        self.assertIsNone(cal.devices)

    def test_bad_device_raises(self):
        K = np.tile(np.eye(3, dtype="float32"), (1, 1, 1))
        orphan = Device(name="orphan")
        with self.assertRaisesWith(ValueError,
                                   "All devices linked from a CameraCalibration object "
                                   "must be added to the NWBFile first."):
            CameraCalibration(intrinsic_matrix=K, devices=[orphan])

    def test_mock_helper(self):
        cal = mock_CameraCalibration(nwbfile=self.nwbfile, n_cameras=2)
        self.assertIsInstance(cal, CameraCalibration)
        self.assertEqual(cal.intrinsic_matrix.shape, (2, 3, 3))
        self.assertEqual(len(cal.devices), 2)


class TestCameraViewConstructor(TestCase):
    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.nwbfile.create_device(name="camera1")

    def test_constructor_device_only(self):
        cv = CameraView(name="camera1", device=self.nwbfile.devices["camera1"])
        self.assertEqual(cv.name, "camera1")
        self.assertIs(cv.device, self.nwbfile.devices["camera1"])
        self.assertIsNone(cv.source_video)
        self.assertEqual(len(cv.pose_estimation_series), 0)

    def test_constructor_with_source_video(self):
        video = ImageSeries(name="camera1", description="cam1", unit="NA",
                            format="external", external_file=["cam1.mp4"], rate=30.0)
        cv = CameraView(name="camera1", device=self.nwbfile.devices["camera1"], source_video=video)
        self.assertIs(cv.source_video, video)

    def test_constructor_with_2d_estimates(self):
        skeleton = mock_Skeleton()
        pes_2d = [mock_PoseEstimationSeries(name=n, unit="pixels") for n in skeleton.nodes]
        cv = CameraView(
            name="camera1",
            device=self.nwbfile.devices["camera1"],
            pose_estimation_series=pes_2d,
        )
        self.assertEqual(len(cv.pose_estimation_series), len(skeleton.nodes))

    def test_bad_device_raises(self):
        orphan = Device(name="orphan")
        with self.assertRaisesWith(ValueError,
                                   "The device linked from a CameraView must be added to the NWBFile first."):
            CameraView(name="orphan_view", device=orphan)

    def test_mock_helper(self):
        cv = mock_CameraView(nwbfile=self.nwbfile, name="camera1")
        self.assertIsInstance(cv, CameraView)
        self.assertIsNotNone(cv.source_video)


class TestMultiCameraPoseEstimationConstructor(TestCase):
    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.nwbfile.create_device(name="camera1")
        self.nwbfile.create_device(name="camera2")

    def test_constructor_full(self):
        skeleton = mock_Skeleton(nodes=["nose", "spine", "tail"])
        pes_3d = [mock_PoseEstimationSeries(name=n, unit="millimeters") for n in skeleton.nodes]
        cv1 = CameraView(name="camera1", device=self.nwbfile.devices["camera1"])
        cv2 = CameraView(name="camera2", device=self.nwbfile.devices["camera2"])
        cal = CameraCalibration(
            intrinsic_matrix=np.tile(np.eye(3, dtype="float32"), (2, 1, 1)),
            devices=[self.nwbfile.devices["camera1"], self.nwbfile.devices["camera2"]],
        )

        mcpe = MultiCameraPoseEstimation(
            name="DANNCE_pose",
            pose_estimation_series=pes_3d,
            camera_views=[cv1, cv2],
            description="3D pose estimated by DANNCE.",
            scorer="DANNCE",
            source_software="DANNCE",
            source_software_version="2.0.0",
            skeleton=skeleton,
            calibration=cal,
        )

        self.assertEqual(mcpe.name, "DANNCE_pose")
        self.assertEqual(len(mcpe.pose_estimation_series), 3)
        self.assertEqual(len(mcpe.camera_views), 2)
        self.assertIs(mcpe.camera_views["camera1"], cv1)
        self.assertIs(mcpe.camera_views["camera2"], cv2)
        self.assertEqual(mcpe.description, "3D pose estimated by DANNCE.")
        self.assertEqual(mcpe.source_software_version, "2.0.0")
        self.assertIs(mcpe.skeleton, skeleton)
        self.assertIs(mcpe.calibration, cal)

    def test_constructor_minimal(self):
        mcpe = MultiCameraPoseEstimation()
        self.assertEqual(mcpe.name, "MultiCameraPoseEstimation")
        self.assertIsNone(mcpe.description)
        self.assertIsNone(mcpe.skeleton)
        self.assertIsNone(mcpe.calibration)
        self.assertEqual(len(mcpe.camera_views), 0)

    def test_mock_helper(self):
        mcpe = mock_MultiCameraPoseEstimation(nwbfile=self.nwbfile)
        self.assertIsInstance(mcpe, MultiCameraPoseEstimation)
        self.assertEqual(len(mcpe.camera_views), 2)
        for cv in mcpe.camera_views.values():
            self.assertIsNotNone(cv.source_video)
