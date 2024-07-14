import datetime
import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.testing import TestCase, remove_test_file, NWBH5IOFlexMixin

from ndx_pose import (
    PoseEstimationSeries,
    PoseEstimation,
    PoseTraining,
    Skeletons,
    SourceVideos,
    TrainingFrames,
)
from ndx_pose.testing.mock.pose import (
    mock_PoseEstimationSeries,
    mock_Skeleton,
    mock_PoseEstimation,
    mock_SkeletonInstance,
    mock_SkeletonInstances,
    mock_TrainingFrame,
    mock_source_video,
)


class TestPoseEstimationSeriesRoundtrip(TestCase):
    """Simple roundtrip test for PoseEstimationSeries."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.path = "test_pose.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """
        Add a PoseEstimationSeries to an NWBFile, write it, read it, and test that the read object matches the original.
        """
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

        # ideally the PoseEstimationSeries is added to a PoseEstimation object but here, test just the series
        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(pes)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(pes, read_nwbfile.processing["behavior"]["front_left_paw"])

    def test_roundtrip_link_timestamps(self):
        """
        Test roundtrip of two PoseEstimationSeries where the timestamps of one links to the timestamps of the other.
        """
        data = np.random.rand(100, 3)  # num_frames x (x, y, z)
        timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
        confidence = np.random.rand(100)  # a confidence value for every frame
        front_left_paw = PoseEstimationSeries(
            name="front_left_paw",
            description="Marker placed around fingers of front left paw.",
            data=data,
            unit="pixels",
            reference_frame="(0,0,0) corresponds to ...",
            timestamps=timestamps,
            confidence=confidence,
            confidence_definition="Softmax output of the deep neural network.",
        )

        data = np.random.rand(100, 3)  # num_frames x (x, y, z)
        confidence = np.random.rand(100)  # a confidence value for every frame
        front_right_paw = PoseEstimationSeries(
            name="front_right_paw",
            description="Marker placed around fingers of front left paw.",
            data=data,
            unit="pixels",
            reference_frame="(0,0,0) corresponds to ...",
            timestamps=front_left_paw,  # link to timestamps of front_left_paw
            confidence=confidence,
            confidence_definition="Softmax output of the deep neural network.",
        )

        # ideally the PoseEstimationSeries is added to a PoseEstiamtion object but here, test just the series
        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(front_left_paw)
        behavior_pm.add(front_right_paw)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(front_left_paw, read_nwbfile.processing["behavior"]["front_left_paw"])
            self.assertContainerEqual(front_right_paw, read_nwbfile.processing["behavior"]["front_right_paw"])
            self.assertIs(
                read_nwbfile.processing["behavior"]["front_left_paw"].timestamps,
                read_nwbfile.processing["behavior"]["front_right_paw"].timestamps,
            )


class TestPoseEstimationSeriesRoundtripPyNWB(NWBH5IOFlexMixin, TestCase):
    """Complex, more complete roundtrip test for PoseEstimationSeries using pynwb.testing infrastructure."""

    def getContainerType(self):
        return "PoseEstimationSeries"

    def addContainer(self):
        """Add the test PoseEstimationSeries to the given NWBFile"""
        pes = mock_PoseEstimationSeries(name="test_PES")

        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(pes)

    def getContainer(self, nwbfile: NWBFile):
        return nwbfile.processing["behavior"]["test_PES"]


class TestPoseEstimationRoundtrip(TestCase):
    """Simple roundtrip test for PoseEstimation."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.nwbfile.create_device(name="camera1")
        self.nwbfile.create_device(name="camera2")
        self.path = "test_pose.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """
        Add a PoseEstimation to an NWBFile, write it, read it, and test that the read object matches the original.
        """
        skeleton = mock_Skeleton()
        skeletons = Skeletons(skeletons=[skeleton])

        pose_estimation_series = [mock_PoseEstimationSeries(name=name) for name in skeleton.nodes]
        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            description="Estimated positions of front paws using DeepLabCut.",
            original_videos=["camera1.mp4"],
            labeled_videos=["camera1_labeled.mp4"],
            dimensions=np.array([[640, 480]], dtype="uint16"),
            devices=[self.nwbfile.devices["camera1"]],
            scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
            source_software="DeepLabCut",
            source_software_version="2.2b8",
            skeleton=skeleton,
        )

        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(pe)
        behavior_pm.add(skeletons)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_pe = read_nwbfile.processing["behavior"]["PoseEstimation"]
            self.assertContainerEqual(read_pe, pe)
            self.assertEqual(len(read_pe.pose_estimation_series), 3)
            self.assertContainerEqual(read_pe.pose_estimation_series["node1"], pose_estimation_series[0])
            self.assertContainerEqual(read_pe.pose_estimation_series["node2"], pose_estimation_series[1])
            self.assertContainerEqual(read_pe.pose_estimation_series["node3"], pose_estimation_series[2])
            self.assertContainerEqual(read_pe.skeleton, skeleton)
            self.assertEqual(len(read_pe.devices), 1)
            self.assertContainerEqual(read_pe.devices[0], self.nwbfile.devices["camera1"])


class TestPoseEstimationRoundtripPyNWB(NWBH5IOFlexMixin, TestCase):
    """Complex, more complete roundtrip test for PoseEstimation using pynwb.testing infrastructure."""

    def getContainerType(self):
        return "PoseEstimation"

    def addContainer(self):
        """Add the test PoseEstimation to the given NWBFile"""
        mock_PoseEstimation(nwbfile=self.nwbfile)

    def getContainer(self, nwbfile: NWBFile):
        return nwbfile.processing["behavior"]["PoseEstimation"]


class TestPoseTrainingRoundtripPyNWB(NWBH5IOFlexMixin, TestCase):
    """Complex, more complete roundtrip test for PoseTraining using pynwb.testing infrastructure."""

    def getContainerType(self):
        return "PoseTraining"

    def addContainer(self):
        """Add the test PoseTraining to the given NWBFile"""
        skeleton1 = mock_Skeleton(name="subject1")
        skeleton2 = mock_Skeleton(name="subject2")
        skeletons = Skeletons(skeletons=[skeleton1, skeleton2])

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

        behavior_pm = self.nwbfile.create_processing_module(
            name="behavior",
            description="processed behavioral data",
        )
        behavior_pm.add(skeletons)
        behavior_pm.add(pose_training)

    def getContainer(self, nwbfile: NWBFile):
        return nwbfile.processing["behavior"]["PoseTraining"]
