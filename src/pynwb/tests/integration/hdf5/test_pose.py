import datetime
import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.image import ImageSeries
from pynwb.testing import TestCase, remove_test_file, NWBH5IOFlexMixin

from ndx_pose import (
    CalibratedCamera,
    MultiCameraPoseEstimation,
    PoseEstimationSeries,
    PoseEstimation,
    PoseTraining,
    Skeletons,
    SourceVideos,
    TrainingFrames,
)
from ndx_pose.testing.mock.pose import (
    mock_CalibratedCamera,
    mock_MultiCameraPoseEstimation,
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
            device=self.nwbfile.devices["camera1"],
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
            self.assertContainerEqual(read_pe.device, self.nwbfile.devices["camera1"])


class TestPoseEstimationRoundtripSourceVideo(TestCase):
    """Roundtrip test for PoseEstimation with source_video link."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.nwbfile.create_device(name="camera1")
        self.path = "test_pose.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test that source_video link survives write/read roundtrip."""
        source_video = ImageSeries(
            name="source_video",
            description="Source video for pose estimation.",
            unit="NA",
            format="external",
            external_file=["camera1.mp4"],
            dimension=[640, 480],
            starting_frame=[0],
            rate=30.0,
            num_samples=10,
        )
        self.nwbfile.add_acquisition(source_video)

        skeleton = mock_Skeleton()
        skeletons = Skeletons(skeletons=[skeleton])

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

        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(pe)
        behavior_pm.add(skeletons)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_pe = read_nwbfile.processing["behavior"]["PoseEstimation"]
            self.assertContainerEqual(read_pe.source_video, source_video)
            self.assertEqual(read_pe.source_video.external_file[0], "camera1.mp4")


class TestPoseEstimationRoundtripLabeledVideo(TestCase):
    """Roundtrip test for PoseEstimation with labeled_video link."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.nwbfile.create_device(name="camera1")
        self.path = "test_pose.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test that labeled_video link survives write/read roundtrip."""
        labeled_video = ImageSeries(
            name="labeled_video",
            description="Labeled video produced from pose estimation.",
            unit="NA",
            format="external",
            external_file=["camera1_labeled.mp4"],
            dimension=[640, 480],
            starting_frame=[0],
            rate=30.0,
            num_samples=10,
        )
        self.nwbfile.add_acquisition(labeled_video)

        skeleton = mock_Skeleton()
        skeletons = Skeletons(skeletons=[skeleton])

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

        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(pe)
        behavior_pm.add(skeletons)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_pe = read_nwbfile.processing["behavior"]["PoseEstimation"]
            self.assertContainerEqual(read_pe.labeled_video, labeled_video)
            self.assertEqual(read_pe.labeled_video.external_file[0], "camera1_labeled.mp4")


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


class TestCalibratedCameraRoundtrip(TestCase):
    """Simple roundtrip test for CalibratedCamera."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.path = "test_calibrated_camera.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Write a CalibratedCamera and verify all matrices survive the roundtrip."""
        rng = np.random.default_rng(0)

        intrinsic = np.eye(3, dtype="float32")
        intrinsic[0, 0] = 800.0
        intrinsic[1, 1] = 800.0
        intrinsic[0, 2] = 320.0
        intrinsic[1, 2] = 240.0
        rotation = np.eye(3, dtype="float32")
        translation = rng.standard_normal(3).astype("float32")
        distortion = np.zeros(5, dtype="float32")

        camera = CalibratedCamera(
            name="camera1",
            intrinsic_matrix=intrinsic,
            rotation_matrix=rotation,
            translation_vector=translation,
            distortion_coefficients=distortion,
        )
        self.nwbfile.add_device(camera)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_camera = read_nwbfile.devices["camera1"]

            self.assertIsInstance(read_camera, CalibratedCamera)
            np.testing.assert_array_almost_equal(read_camera.intrinsic_matrix, intrinsic)
            np.testing.assert_array_almost_equal(read_camera.rotation_matrix, rotation)
            np.testing.assert_array_almost_equal(read_camera.translation_vector, translation)
            np.testing.assert_array_almost_equal(read_camera.distortion_coefficients, distortion)

    def test_roundtrip_intrinsics_only(self):
        """CalibratedCamera with only intrinsic_matrix (all optional fields omitted)."""
        intrinsic = np.eye(3, dtype="float32")

        camera = CalibratedCamera(name="camera1", intrinsic_matrix=intrinsic)
        self.nwbfile.add_device(camera)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_camera = read_nwbfile.devices["camera1"]
            np.testing.assert_array_almost_equal(read_camera.intrinsic_matrix, intrinsic)
            self.assertIsNone(read_camera.rotation_matrix)
            self.assertIsNone(read_camera.translation_vector)
            self.assertIsNone(read_camera.distortion_coefficients)


class TestMultiCameraPoseEstimationRoundtrip(TestCase):
    """Simple roundtrip test for MultiCameraPoseEstimation."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        self.path = "test_multicamera_pose.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Write a MultiCameraPoseEstimation and verify all fields survive the roundtrip."""
        skeleton = mock_Skeleton(nodes=["nose", "spine", "tail"])
        skeletons = Skeletons(skeletons=[skeleton])

        rng = np.random.default_rng(1)
        camera1 = CalibratedCamera(
            name="camera1",
            intrinsic_matrix=np.eye(3, dtype="float32"),
            rotation_matrix=np.eye(3, dtype="float32"),
            translation_vector=rng.standard_normal(3).astype("float32"),
            distortion_coefficients=np.zeros(5, dtype="float32"),
        )
        camera2 = CalibratedCamera(
            name="camera2",
            intrinsic_matrix=np.eye(3, dtype="float32"),
            rotation_matrix=np.eye(3, dtype="float32"),
            translation_vector=rng.standard_normal(3).astype("float32"),
            distortion_coefficients=np.zeros(5, dtype="float32"),
        )
        self.nwbfile.add_device(camera1)
        self.nwbfile.add_device(camera2)

        video1 = ImageSeries(
            name="camera1_video", description="cam1", unit="NA",
            format="external", external_file=["cam1.mp4"], rate=30.0,
        )
        video2 = ImageSeries(
            name="camera2_video", description="cam2", unit="NA",
            format="external", external_file=["cam2.mp4"], rate=30.0,
        )
        self.nwbfile.add_acquisition(video1)
        self.nwbfile.add_acquisition(video2)

        pes_list = [
            mock_PoseEstimationSeries(
                name=node,
                data=np.arange(30, dtype=np.float64).reshape((10, 3)),
                unit="millimeters",
                reference_frame="(0,0,0) is the midpoint of the camera rig.",
            )
            for node in skeleton.nodes
        ]

        pose_estimations = [
            PoseEstimation(
                name="PoseEstimation_camera1",
                pose_estimation_series=[
                    mock_PoseEstimationSeries(name=node, unit="pixels") for node in skeleton.nodes
                ],
                description="2D pose estimates from camera1.",
                device=camera1,
                source_video=video1,
                skeleton=skeleton,
            ),
            PoseEstimation(
                name="PoseEstimation_camera2",
                pose_estimation_series=[
                    mock_PoseEstimationSeries(name=node, unit="pixels") for node in skeleton.nodes
                ],
                description="2D pose estimates from camera2.",
                device=camera2,
                source_video=video2,
                skeleton=skeleton,
            ),
        ]

        mcpe = MultiCameraPoseEstimation(
            pose_estimation_series=pes_list,
            pose_estimations=pose_estimations,
            description="3D pose estimated by DANNCE.",
            scorer="DANNCE",
            source_software="DANNCE",
            source_software_version="2.0.0",
            skeleton=skeleton,
        )

        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(mcpe)
        behavior_pm.add(skeletons)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_mcpe = read_nwbfile.processing["behavior"]["MultiCameraPoseEstimation"]

            self.assertEqual(read_mcpe.description, "3D pose estimated by DANNCE.")
            self.assertEqual(read_mcpe.scorer, "DANNCE")
            self.assertEqual(read_mcpe.source_software, "DANNCE")
            self.assertEqual(read_mcpe.source_software_version, "2.0.0")
            self.assertContainerEqual(read_mcpe.skeleton, skeleton)

            self.assertEqual(len(read_mcpe.pose_estimation_series), 3)
            self.assertIn("nose", read_mcpe.pose_estimation_series)

            self.assertEqual(len(read_mcpe.pose_estimations), 2)
            read_pe1 = read_mcpe.pose_estimations["PoseEstimation_camera1"]
            self.assertContainerEqual(read_pe1.device, camera1)
            self.assertIsInstance(read_pe1.device, CalibratedCamera)
            np.testing.assert_array_almost_equal(read_pe1.device.intrinsic_matrix, camera1.intrinsic_matrix)
            self.assertEqual(read_pe1.source_video.external_file[0], "cam1.mp4")

    def test_roundtrip_minimal(self):
        """MultiCameraPoseEstimation with only required fields (name) survives roundtrip."""
        mcpe = MultiCameraPoseEstimation(name="MultiCameraPoseEstimation")

        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(mcpe)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_mcpe = read_nwbfile.processing["behavior"]["MultiCameraPoseEstimation"]
            self.assertIsNone(read_mcpe.description)
            self.assertIsNone(read_mcpe.scorer)
            self.assertIsNone(read_mcpe.skeleton)
            self.assertEqual(len(read_mcpe.pose_estimation_series), 0)
            self.assertEqual(len(read_mcpe.pose_estimations), 0)


class TestMultiCameraPoseEstimationRoundtripPyNWB(NWBH5IOFlexMixin, TestCase):
    """Full roundtrip test using the pynwb.testing infrastructure."""

    def getContainerType(self):
        return "MultiCameraPoseEstimation"

    def addContainer(self):
        mock_MultiCameraPoseEstimation(nwbfile=self.nwbfile)

    def getContainer(self, nwbfile: NWBFile):
        return nwbfile.processing["behavior"]["MultiCameraPoseEstimation"]
