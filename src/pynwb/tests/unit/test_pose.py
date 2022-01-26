import datetime
import numpy as np

from pynwb import NWBFile
from pynwb.testing import TestCase

from ndx_pose import PoseEstimationSeries, PoseEstimation


def create_series():
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
        timestamps=front_left_paw,  # link to timestamps of front_left_paw
        confidence=confidence,
        confidence_definition='Softmax output of the deep neural network.',
    )

    return [front_left_paw, front_right_paw]


class TestPoseEstimationSeriesConstructor(TestCase):

    def test_constructor(self):
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

        self.assertEqual(pes.name, 'front_left_paw')
        self.assertEqual(pes.description, 'Marker placed around fingers of front left paw.')
        np.testing.assert_array_equal(pes.data, data)
        self.assertEqual(pes.unit, 'pixels')
        self.assertEqual(pes.reference_frame, '(0,0,0) corresponds to ...')
        np.testing.assert_array_equal(pes.timestamps, timestamps)
        np.testing.assert_array_equal(pes.confidence, confidence)
        self.assertEqual(pes.confidence_definition, 'Softmax output of the deep neural network.')


class TestPoseEstimationConstructor(TestCase):

    def setUp(self):
        nwbfile = NWBFile(
            session_description='session_description',
            identifier='identifier',
            session_start_time=datetime.datetime.now(datetime.timezone.utc)
        )
        nwbfile.create_device(name='camera1')
        nwbfile.create_device(name='camera2')

        self.nwbfile = nwbfile

    def test_constructor(self):
        """Test that the constructor for PoseEstimation sets values as expected."""
        pose_estimation_series = create_series()
        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            description='Estimated positions of front paws using DeepLabCut.',
            original_videos=['camera1.mp4', 'camera2.mp4'],
            labeled_videos=['camera1_labeled.mp4', 'camera2_labeled.mp4'],
            dimensions=np.array([[640, 480], [1024, 768]], dtype='uint8'),
            scorer='DLC_resnet50_openfieldOct30shuffle1_1600',
            source_software='DeepLabCut',
            source_software_version='2.2b8',
            nodes=['front_left_paw', 'front_right_paw'],
            edges=np.array([[0, 1]], dtype='uint8'),
            # devices=[self.nwbfile.devices['camera1'], self.nwbfile.devices['camera2']],
        )

        self.assertEqual(pe.name, 'PoseEstimation')
        self.assertEqual(len(pe.pose_estimation_series), 2)
        self.assertIs(pe.pose_estimation_series['front_left_paw'], pose_estimation_series[0])
        self.assertIs(pe.pose_estimation_series['front_right_paw'], pose_estimation_series[1])
        self.assertEqual(pe.description, 'Estimated positions of front paws using DeepLabCut.')
        self.assertEqual(pe.original_videos, ['camera1.mp4', 'camera2.mp4'])
        self.assertEqual(pe.labeled_videos, ['camera1_labeled.mp4', 'camera2_labeled.mp4'])
        np.testing.assert_array_equal(pe.dimensions, np.array([[640, 480], [1024, 768]], dtype='uint8'))
        self.assertEqual(pe.scorer, 'DLC_resnet50_openfieldOct30shuffle1_1600')
        self.assertEqual(pe.source_software, 'DeepLabCut')
        self.assertEqual(pe.source_software_version, '2.2b8')
        self.assertEqual(pe.nodes, ['front_left_paw', 'front_right_paw'])
        np.testing.assert_array_equal(pe.edges, np.array([[0, 1]], dtype='uint8'))
        # self.assertEqual(len(pe.devices), 2)
        # self.assertIs(pe.devices['camera1'], self.nwbfile.devices['camera1'])
        # self.assertIs(pe.devices['camera2'], self.nwbfile.devices['camera2'])
