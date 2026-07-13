import datetime
import numpy as np

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.testing import TestCase
from pynwb.file import Subject

from pynwb.image import ImageSeries

from ndx_pose import (
    CalibratedCamera,
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
    mock_CalibratedCamera,
    mock_MultiCameraPoseEstimation,
    mock_PoseEstimation,
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
            device=self.nwbfile.devices["camera1"],
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
        self.assertIs(pe.device, self.nwbfile.devices["camera1"])
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
        orphan_camera = Device(name="orphan_camera")

        msg = "The device linked from a PoseEstimation object must be added to the NWBFile first."
        with self.assertRaisesWith(ValueError, msg):
            PoseEstimation(
                pose_estimation_series=pose_estimation_series,
                description="Estimated positions of front paws using DeepLabCut.",
                device=orphan_camera,
                scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
                source_software="DeepLabCut",
                source_software_version="2.2b8",
                skeleton=skeleton,
            )

    def test_deprecated_devices_single(self):
        """Test that the deprecated 'devices' argument still works with a single device."""
        skeleton = mock_Skeleton()
        pose_estimation_series = [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]

        msg = (
            "The 'devices' constructor argument is deprecated. Please use the 'device' argument instead. "
            "This will be removed in a future release."
        )
        with self.assertWarnsWith(DeprecationWarning, msg):
            pe = PoseEstimation(
                pose_estimation_series=pose_estimation_series,
                skeleton=skeleton,
                devices=[self.nwbfile.devices["camera1"]],
            )
        self.assertIs(pe.device, self.nwbfile.devices["camera1"])

    def test_deprecated_devices_multiple_raises(self):
        """Test that passing more than one device via the deprecated 'devices' argument raises an error."""
        skeleton = mock_Skeleton()
        pose_estimation_series = [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]

        msg = (
            "PoseEstimation now represents pose estimates from a single camera view and supports only one "
            "device. For multi-camera setups, add one PoseEstimation per camera view to a "
            "MultiCameraPoseEstimation object instead of passing multiple 'devices' here."
        )
        with self.assertRaisesWith(ValueError, msg):
            PoseEstimation(
                pose_estimation_series=pose_estimation_series,
                skeleton=skeleton,
                devices=[self.nwbfile.devices["camera1"], self.nwbfile.devices["camera2"]],
            )

    def test_device_and_devices_raises(self):
        """Test that passing both 'device' and 'devices' raises an error."""
        skeleton = mock_Skeleton()
        pose_estimation_series = [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]

        msg = "Cannot specify both 'device' and 'devices'. Please use 'device' only."
        with self.assertRaisesWith(ValueError, msg):
            PoseEstimation(
                pose_estimation_series=pose_estimation_series,
                skeleton=skeleton,
                device=self.nwbfile.devices["camera1"],
                devices=[self.nwbfile.devices["camera2"]],
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
                device=self.nwbfile.devices["camera1"],
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
            device=self.nwbfile.devices["camera1"],
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
            device=self.nwbfile.devices["camera1"],
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


class TestCalibratedCameraConstructor(TestCase):
    def test_constructor_full(self):
        K = np.eye(3, dtype="float32")
        R = np.eye(3, dtype="float32")
        t = np.zeros(3, dtype="float32")
        d = np.zeros(5, dtype="float32")

        camera = CalibratedCamera(
            name="camera1",
            description="A camera.",
            manufacturer="Basler",
            intrinsic_matrix=K,
            rotation_matrix=R,
            translation_vector=t,
            distortion_coefficients=d,
        )

        self.assertEqual(camera.name, "camera1")
        np.testing.assert_array_equal(camera.intrinsic_matrix, K)
        np.testing.assert_array_equal(camera.rotation_matrix, R)
        np.testing.assert_array_equal(camera.translation_vector, t)
        np.testing.assert_array_equal(camera.distortion_coefficients, d)

    def test_constructor_intrinsics_only(self):
        K = np.eye(3, dtype="float32")
        camera = CalibratedCamera(name="camera1", intrinsic_matrix=K)
        np.testing.assert_array_equal(camera.intrinsic_matrix, K)
        self.assertIsNone(camera.rotation_matrix)
        self.assertIsNone(camera.translation_vector)
        self.assertIsNone(camera.distortion_coefficients)

    def test_mock_helper(self):
        nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        camera = mock_CalibratedCamera(nwbfile=nwbfile, name="camera1")
        self.assertIsInstance(camera, CalibratedCamera)
        self.assertEqual(camera.intrinsic_matrix.shape, (3, 3))
        self.assertIs(nwbfile.devices["camera1"], camera)


class TestMultiCameraPoseEstimationConstructor(TestCase):
    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )

    def test_constructor_full(self):
        self.nwbfile.create_device(name="camera1")
        self.nwbfile.create_device(name="camera2")
        skeleton = mock_Skeleton(nodes=["nose", "spine", "tail"])
        pes_3d = [mock_PoseEstimationSeries(name=n, unit="millimeters") for n in skeleton.nodes]
        pe1 = mock_PoseEstimation(
            nwbfile=self.nwbfile,
            name="PoseEstimation_camera1",
            skeleton=skeleton,
            device=self.nwbfile.devices["camera1"],
            add_to_nwbfile=False,
        )
        pe2 = mock_PoseEstimation(
            nwbfile=self.nwbfile,
            name="PoseEstimation_camera2",
            skeleton=skeleton,
            device=self.nwbfile.devices["camera2"],
            add_to_nwbfile=False,
        )

        mcpe = MultiCameraPoseEstimation(
            name="DANNCE_pose",
            pose_estimation_series=pes_3d,
            pose_estimations=[pe1, pe2],
            description="3D pose estimated by DANNCE.",
            scorer="DANNCE",
            source_software="DANNCE",
            source_software_version="2.0.0",
            skeleton=skeleton,
        )

        self.assertEqual(mcpe.name, "DANNCE_pose")
        self.assertEqual(len(mcpe.pose_estimation_series), 3)
        self.assertEqual(len(mcpe.pose_estimations), 2)
        self.assertIs(mcpe.pose_estimations[pe1.name], pe1)
        self.assertIs(mcpe.pose_estimations[pe2.name], pe2)
        self.assertEqual(mcpe.description, "3D pose estimated by DANNCE.")
        self.assertEqual(mcpe.source_software_version, "2.0.0")
        self.assertIs(mcpe.skeleton, skeleton)

    def test_constructor_minimal(self):
        mcpe = MultiCameraPoseEstimation()
        self.assertEqual(mcpe.name, "MultiCameraPoseEstimation")
        self.assertIsNone(mcpe.description)
        self.assertIsNone(mcpe.skeleton)
        self.assertEqual(len(mcpe.pose_estimations), 0)

    def test_mock_helper(self):
        mcpe = mock_MultiCameraPoseEstimation(nwbfile=self.nwbfile)
        self.assertIsInstance(mcpe, MultiCameraPoseEstimation)
        self.assertEqual(len(mcpe.pose_estimations), 2)
        for pe in mcpe.pose_estimations.values():
            self.assertIsNotNone(pe.device)
            self.assertIsInstance(pe.device, CalibratedCamera)
