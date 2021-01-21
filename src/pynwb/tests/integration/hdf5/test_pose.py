import datetime
import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.testing import TestCase, remove_test_file, NWBH5IOMixin

from ndx_pose import PoseEstimationSeries, PoseEstimation
from ...unit.test_pose import create_series


class TestPoseEstimationSeriesRoundtrip(TestCase):
    """Simple roundtrip test for PoseEstimationSeries."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description='session_description',
            identifier='identifier',
            session_start_time=datetime.datetime.now(datetime.timezone.utc)
        )
        self.path = 'test_pose.nwb'

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
            name='front_left_paw',
            description='Marker placed around fingers of front left paw.',
            data=data,
            unit='pixels',
            reference_frame='(0,0,0) corresponds to ...',
            timestamps=timestamps,
            confidence=confidence,
            confidence_definition='Softmax output of the deep neural network.',
        )

        # ideally the PoseEstimationSeries is added to a PoseEstiamtion object but here, test just the series
        behavior_pm = self.nwbfile.create_processing_module(
            name='behavior',
            description='processed behavioral data'
        )
        behavior_pm.add(pes)

        with NWBHDF5IO(self.path, mode='w') as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(pes, read_nwbfile.processing['behavior']['front_left_paw'])


class TestPoseEstimationSeriesRoundtripPyNWB(NWBH5IOMixin, TestCase):
    """Complex, more complete roundtrip test for PoseEstimationSeries using pynwb.testing infrastructure."""

    def setUpContainer(self):
        """ Return the test PoseEstimationSeries to read/write """
        data = np.random.rand(100, 3)  # num_frames x (x, y, z)
        timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
        confidence = np.random.rand(100)  # a confidence value for every frame
        pes = PoseEstimationSeries(
            name='front_left_paw',
            description='Marker placed around fingers of front left paw.',
            data=data,
            unit='pixels',
            reference_frame='(0,0,0) corresponds to ...',
            timestamps=timestamps,
            confidence=confidence,
            confidence_definition='Softmax output of the deep neural network.',
        )
        return pes

    def addContainer(self, nwbfile):
        """ Add the test PoseEstimationSeries to the given NWBFile """
        behavior_pm = nwbfile.create_processing_module(
            name='behavior',
            description='processed behavioral data'
        )
        behavior_pm.add(self.container)

    def getContainer(self, nwbfile):
        """ Return the test PoseEstimationSeries from the given NWBFile """
        return nwbfile.processing['behavior'][self.container.name]


class TestPoseEstimationRoundtrip(TestCase):
    """Simple roundtrip test for PoseEstimation."""

    def setUp(self):
        self.nwbfile = NWBFile(
            session_description='session_description',
            identifier='identifier',
            session_start_time=datetime.datetime.now(datetime.timezone.utc)
        )
        self.nwbfile.create_device(name='camera1')
        self.nwbfile.create_device(name='camera2')
        self.path = 'test_pose.nwb'

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """
        Add a PoseEstimation to an NWBFile, write it, read it, and test that the read object matches the original.
        """
        pose_estimation_series = create_series()
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

        behavior_pm = self.nwbfile.create_processing_module(
            name='behavior',
            description='processed behavioral data'
        )
        behavior_pm.add(pe)

        with NWBHDF5IO(self.path, mode='w') as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_pe = read_nwbfile.processing['behavior']['PoseEstimation']
            self.assertContainerEqual(read_pe, pe)
            self.assertEqual(len(read_pe.pose_estimation_series), 2)
            self.assertContainerEqual(read_pe.pose_estimation_series['front_left_paw'], pose_estimation_series[0])
            self.assertContainerEqual(read_pe.pose_estimation_series['front_right_paw'], pose_estimation_series[1])
            # self.assertEqual(len(read_pe.devices), 2)
            # self.assertContainerEqual(read_pe.devices['camera1'], self.nwbfile.devices['camera1'])
            # self.assertContainerEqual(read_pe.devices['camera2'], self.nwbfile.devices['camera2'])


class TestPoseEstimationRoundtripPyNWB(NWBH5IOMixin, TestCase):
    """Complex, more complete roundtrip test for PoseEstimation using pynwb.testing infrastructure."""

    def setUpContainer(self):
        """ Return the test PoseEstimation to read/write """
        pose_estimation_series = create_series()
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
        return pe

    def addContainer(self, nwbfile):
        """ Add the test PoseEstimation to the given NWBFile """
        behavior_pm = nwbfile.create_processing_module(
            name='behavior',
            description='processed behavioral data'
        )
        behavior_pm.add(self.container)

    def getContainer(self, nwbfile):
        """ Return the test PoseEstimation from the given NWBFile """
        return nwbfile.processing['behavior'][self.container.name]
