import datetime
import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.core import DynamicTableRegion
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup
from pynwb.file import ElectrodeTable as get_electrode_table
from pynwb.testing import TestCase, remove_test_file, AcquisitionH5IOMixin

from ndx_pose import PoseEstimationSeries, PoseEstimation


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
        )

        self.assertEqual(pes.name, 'front_left_paw')
        self.assertEqual(pes.description, 'Marker placed around fingers of front left paw.')
        np.testing.assert_array_equal(pes.data, data)
        self.assertEqual(pes.unit, 'pixels')
        self.assertEqual(pes.reference_frame, '(0,0,0) corresponds to ...')
        np.testing.assert_array_equal(pes.timestamps, timestamps)
        np.testing.assert_array_equal(pes.confidence, confidence)


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

    def create_series(self):
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
        )

        return [front_left_paw, front_right_paw]

    def test_constructor(self):
        """Test that the constructor for PoseEstimation sets values as expected."""
        pose_estimation_series = self.create_series()
        print(self.nwbfile.devices)
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
            cameras=[self.nwbfile.devices['camera1'], self.nwbfile.devices['camera2']],
        )

        self.assertEqual(pe.name, 'PoseEstimation')
        self.assertEqual(len(pe.pose_estimation_series), 2)
        self.assertIs(pe.pose_estimation_series['front_left_paw'], pose_estimation_series[0]),
        self.assertIs(pe.pose_estimation_series['front_right_paw'], pose_estimation_series[1]),
        self.assertEqual(pe.description, 'Estimated positions of front paws using DeepLabCut.')
        self.assertEqual(pe.original_videos, ['camera1.mp4', 'camera2.mp4'])
        self.assertEqual(pe.labeled_videos, ['camera1_labeled.mp4', 'camera2_labeled.mp4'])
        self.assertEqual(pe.dimensions, [[640, 480], [1024, 768]])
        self.assertEqual(pe.scorer, 'DLC_resnet50_openfieldOct30shuffle1_1600')
        self.assertEqual(pe.source_software, 'DeepLabCut')
        self.assertEqual(pe.source_software_version, '2.2b8')
        self.assertEqual(pe.nodes, ['front_left_paw', 'front_right_paw'])
        self.assertEqual(pe.edges, [[0, 1]])
        self.assertEqual(len(pe.cameras), 2)
        self.assertIs(pe.cameras[0], self.nwbfile.devices['camera1']),
        self.assertIs(pe.cameras[1], self.nwbfile.devices['camera2']),



# class TestTetrodeSeriesRoundtrip(TestCase):
#     """Simple roundtrip test for TetrodeSeries."""
#
#     def setUp(self):
#         self.nwbfile = set_up_nwbfile()
#         self.path = 'test.nwb'
#
#     def tearDown(self):
#         remove_test_file(self.path)
#
#     def test_roundtrip(self):
#         """
#         Add a TetrodeSeries to an NWBFile, write it to file, read the file, and test that the TetrodeSeries from the
#         file matches the original TetrodeSeries.
#         """
#         all_electrodes = self.nwbfile.create_electrode_table_region(
#             region=list(range(0, 10)),
#             description='all the electrodes'
#         )
#
#         data = np.random.rand(100, 3)
#         tetrode_series = TetrodeSeries(
#             name='TetrodeSeries',
#             description='description',
#             data=data,
#             rate=1000.,
#             electrodes=all_electrodes,
#             trode_id=1
#         )
#
#         self.nwbfile.add_acquisition(tetrode_series)
#
#         with NWBHDF5IO(self.path, mode='w') as io:
#             io.write(self.nwbfile)
#
#         with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
#             read_nwbfile = io.read()
#             self.assertContainerEqual(tetrode_series, read_nwbfile.acquisition['TetrodeSeries'])
#
#
# class TestTetrodeSeriesRoundtripPyNWB(AcquisitionH5IOMixin, TestCase):
#     """Complex, more complete roundtrip test for TetrodeSeries using pynwb.testing infrastructure."""
#
#     def setUpContainer(self):
#         """ Return the test TetrodeSeries to read/write """
#         self.device = Device(
#             name='device_name'
#         )
#
#         self.group = ElectrodeGroup(
#             name='electrode_group',
#             description='description',
#             location='location',
#             device=self.device
#         )
#
#         self.table = get_electrode_table()  # manually create a table of electrodes
#         for i in range(10):
#             self.table.add_row(
#                 x=i,
#                 y=i,
#                 z=i,
#                 imp=np.nan,
#                 location='location',
#                 filtering='filtering',
#                 group=self.group,
#                 group_name='electrode_group'
#             )
#
#         all_electrodes = DynamicTableRegion(
#             data=list(range(0, 10)),
#             description='all the electrodes',
#             name='electrodes',
#             table=self.table
#         )
#
#         data = np.random.rand(100, 3)
#         tetrode_series = TetrodeSeries(
#             name='name',
#             description='description',
#             data=data,
#             rate=1000.,
#             electrodes=all_electrodes,
#             trode_id=1
#         )
#         return tetrode_series
#
#     def addContainer(self, nwbfile):
#         """Add the test TetrodeSeries and related objects to the given NWBFile."""
#         nwbfile.add_device(self.device)
#         nwbfile.add_electrode_group(self.group)
#         nwbfile.set_electrode_table(self.table)
#         nwbfile.add_acquisition(self.container)
