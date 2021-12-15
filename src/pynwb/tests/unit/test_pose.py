import datetime
import numpy as np

from pynwb import NWBFile
from pynwb.testing import TestCase

from ndx_pose import PoseEstimationSeries, PoseEstimation, PoseGroupingSeries, AnimalIdentitySeries


def create_pose_series():
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
        pose_estimation_series = create_pose_series()
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

        self.assertEqual(pe.name, 'PoseEstimation')
        self.assertEqual(len(pe.pose_estimation_series), 2)
        self.assertIs(pe.pose_estimation_series['front_left_paw'], pose_estimation_series[0])
        self.assertIs(pe.pose_estimation_series['front_right_paw'], pose_estimation_series[1])
        self.assertEqual(pe.description, 'Estimated positions of front paws using DeepLabCut.')
        self.assertEqual(pe.original_videos, ['camera1.mp4', 'camera2.mp4'])
        self.assertEqual(pe.labeled_videos, ['camera1_labeled.mp4', 'camera2_labeled.mp4'])
        self.assertEqual(pe.dimensions, [[640, 480], [1024, 768]])
        self.assertEqual(pe.scorer, 'DLC_resnet50_openfieldOct30shuffle1_1600')
        self.assertEqual(pe.source_software, 'DeepLabCut')
        self.assertEqual(pe.source_software_version, '2.2b8')
        self.assertEqual(pe.nodes, ['front_left_paw', 'front_right_paw'])
        self.assertEqual(pe.edges, [[0, 1]])
        # self.assertEqual(len(pe.devices), 2)
        # self.assertIs(pe.devices['camera1'], self.nwbfile.devices['camera1'])
        # self.assertIs(pe.devices['camera2'], self.nwbfile.devices['camera2'])


class TestPoseGroupingSeriesConstructor(TestCase):

    def test_constructor(self):
        timestamps = np.linspace(0, 10, num=10)  # a timestamp for every frame
        centroid = np.random.rand(10, 2)  # location of animal for every frame
        score = np.random.rand(10,)  # num_frames

        s = PoseGroupingSeries(
            name="Centroid",
            timestamps=timestamps,
            data=score,
            location=centroid,
        )
        self.assertEqual(s.name, "Centroid")
        np.testing.assert_array_equal(s.timestamps, timestamps)
        np.testing.assert_array_equal(s.data, score)
        np.testing.assert_array_equal(s.location, centroid)

        bbox = np.random.rand(10, 4)

        s = PoseGroupingSeries(
            name="Bounding box",
            timestamps=timestamps,
            data=score,
            location=bbox,
        )
        self.assertEqual(s.name, "Bounding box")
        np.testing.assert_array_equal(s.timestamps, timestamps)
        np.testing.assert_array_equal(s.data, score)
        np.testing.assert_array_equal(s.location, bbox)

        s = PoseGroupingSeries(
            name="PAF matching score",
            timestamps=timestamps,
            data=score,
        )
        self.assertEqual(s.name, "PAF matching score")
        np.testing.assert_array_equal(s.timestamps, timestamps)
        np.testing.assert_array_equal(s.data, score)


class TestAnimalIdentitySeriesConstructor(TestCase):

    def test_constructor(self):
        timestamps = np.linspace(0, 10, num=10)  # a timestamp for every frame
        score = np.random.rand(10,)  # num_frames

        s = AnimalIdentitySeries(
            name="Mouse1",
            timestamps=timestamps,
            data=score,
        )
        self.assertEqual(s.name, "Mouse1")
        np.testing.assert_array_equal(s.timestamps, timestamps)
        np.testing.assert_array_equal(s.data, score)



class TestPoseEstimationMultiAnimalConstructor(TestCase):

    def test_constructor(self):
        """Test that the constructor for PoseEstimation sets values as expected."""
        pose_estimation_series = create_pose_series()
        n_frames = pose_estimation_series[0].data.shape[0]
        pose_grouping_series = PoseGroupingSeries(
            name="Centroid",
            timestamps=pose_estimation_series[0].timestamps,
            data=np.random.rand(n_frames),
            location=np.random.rand(n_frames, 3),
        )
        animal_identity_series = AnimalIdentitySeries(
            name="Mouse1",
            timestamps=pose_estimation_series[0].timestamps,
            data=np.random.rand(n_frames),
        )

        pe = PoseEstimation(
            pose_estimation_series=pose_estimation_series,
            pose_grouping_series=[pose_grouping_series],
            animal_identity_series=[animal_identity_series],
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

        self.assertEqual(pe.name, 'PoseEstimation')
        self.assertEqual(len(pe.pose_estimation_series), 2)
        self.assertIs(pe.pose_estimation_series['front_left_paw'], pose_estimation_series[0])
        self.assertIs(pe.pose_estimation_series['front_right_paw'], pose_estimation_series[1])
        self.assertIs(pe.pose_grouping_series["Centroid"], pose_grouping_series)
        self.assertIs(pe.animal_identity_series["Mouse1"], animal_identity_series)
        self.assertEqual(pe.description, 'Estimated positions of front paws using DeepLabCut.')
        self.assertEqual(pe.original_videos, ['camera1.mp4', 'camera2.mp4'])
        self.assertEqual(pe.labeled_videos, ['camera1_labeled.mp4', 'camera2_labeled.mp4'])
        self.assertEqual(pe.dimensions, [[640, 480], [1024, 768]])
        self.assertEqual(pe.scorer, 'DLC_resnet50_openfieldOct30shuffle1_1600')
        self.assertEqual(pe.source_software, 'DeepLabCut')
        self.assertEqual(pe.source_software_version, '2.2b8')
        self.assertEqual(pe.nodes, ['front_left_paw', 'front_right_paw'])
        self.assertEqual(pe.edges, [[0, 1]])
