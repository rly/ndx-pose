import datetime
import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.testing import TestCase, remove_test_file, NWBH5IOMixin

from ndx_pose import PoseEstimationSeries, PoseEstimation, Skeleton, PoseTraining
from ...unit.test_pose import create_series


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

        # ideally the PoseEstimationSeries is added to a PoseEstiamtion object but here, test just the series
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
        timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
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


class TestPoseEstimationSeriesRoundtripPyNWB(NWBH5IOMixin, TestCase):
    """Complex, more complete roundtrip test for PoseEstimationSeries using pynwb.testing infrastructure."""

    def setUpContainer(self):
        """Return the test PoseEstimationSeries to read/write"""
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
        return pes

    def addContainer(self, nwbfile):
        """Add the test PoseEstimationSeries to the given NWBFile"""
        behavior_pm = nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(self.container)

    def getContainer(self, nwbfile):
        """Return the test PoseEstimationSeries from the given NWBFile"""
        return nwbfile.processing["behavior"][self.container.name]


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
        skeleton = Skeleton(
            id="subject1",
            nodes=["front_left_paw", "front_right_paw"],
            edges=np.array([[0, 1]], dtype="uint8"),
        )
        pose_estimation_series = create_series()
        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            description="Estimated positions of front paws using DeepLabCut.",
            original_videos=["camera1.mp4"],
            labeled_videos=["camera1_labeled.mp4"],
            dimensions=np.array([[640, 480]], dtype="uint16"),
            scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
            source_software="DeepLabCut",
            source_software_version="2.2b8",
            skeleton=skeleton,
            cameras=[self.nwbfile.devices["camera1"]],
        )

        pose_training = PoseTraining(skeletons=[skeleton])

        behavior_pm = self.nwbfile.create_processing_module(name="behavior", description="processed behavioral data")
        behavior_pm.add(pe)
        behavior_pm.add(pose_training)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_pe = read_nwbfile.processing["behavior"]["PoseEstimation"]
            self.assertContainerEqual(read_pe, pe)
            self.assertEqual(len(read_pe.pose_estimation_series), 2)
            self.assertContainerEqual(read_pe.pose_estimation_series["front_left_paw"], pose_estimation_series[0])
            self.assertContainerEqual(read_pe.pose_estimation_series["front_right_paw"], pose_estimation_series[1])
            self.assertEqual(len(read_pe.cameras), 1)
            self.assertContainerEqual(read_pe.cameras[0], self.nwbfile.devices["camera1"])


# NOTE it is recommended to add links to devices in the constructor of PoseEstimation. however,
# the current execution flow of NWBH5IOMixin does not allow tests to create devices on the NWB file
# prior to creating the container.
# this may be cleaned up in a future version of PyNWB/HDMF.

# class TestPoseEstimationRoundtripPyNWB(NWBH5IOMixin, TestCase):
#     """Complex, more complete roundtrip test for PoseEstimation using pynwb.testing infrastructure."""
#
#
#     def setUpContainer(self):
#         """ Return the test PoseEstimation to read/write """
#
#         pose_estimation_series = create_series()
#         pe = PoseEstimation(
#             pose_estimation_series=pose_estimation_series,
#             description='Estimated positions of front paws using DeepLabCut.',
#             original_videos=['camera1.mp4', 'camera2.mp4'],
#             labeled_videos=['camera1_labeled.mp4', 'camera2_labeled.mp4'],
#             dimensions=np.array([[640, 480], [1024, 768]], dtype='uint8'),
#             scorer='DLC_resnet50_openfieldOct30shuffle1_1600',
#             source_software='DeepLabCut',
#             source_software_version='2.2b8',
#             nodes=['front_left_paw', 'front_right_paw'],
#             edges=np.array([[0, 1]], dtype='uint8'),
#             devices=[self.camera1, self.camera2]
#         )
#         return pe
#
#     def addContainer(self, nwbfile):
#         """ Add the test PoseEstimation to the given NWBFile """
#         # self.camera1 = nwbfile.create_device(name='camera1')
#         # self.camera2 = nwbfile.create_device(name='camera2')
#
#         behavior_pm = nwbfile.create_processing_module(
#             name='behavior',
#             description='processed behavioral data'
#         )
#         behavior_pm.add(self.container)
#
#     def getContainer(self, nwbfile):
#         """ Return the test PoseEstimation from the given NWBFile """
#         return nwbfile.processing['behavior'][self.container.name]
