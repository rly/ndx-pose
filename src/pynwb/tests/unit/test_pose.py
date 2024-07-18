import datetime
import numpy as np

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.testing import TestCase
from pynwb.file import Subject

from ndx_pose import (
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
