import datetime
import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.core import DynamicTableRegion
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup
from pynwb.file import ElectrodeTable as get_electrode_table
from pynwb.testing import TestCase, remove_test_file, AcquisitionH5IOMixin

from ndx_pose import TetrodeSeries


def set_up_nwbfile():
    nwbfile = NWBFile(
        session_description='session_description',
        identifier='identifier',
        session_start_time=datetime.datetime.now(datetime.timezone.utc)
    )

    device = nwbfile.create_device(
        name='device_name'
    )

    electrode_group = nwbfile.create_electrode_group(
        name='electrode_group',
        description='description',
        location='location',
        device=device
    )

    for i in np.arange(10.):
        nwbfile.add_electrode(
            x=i,
            y=i,
            z=i,
            imp=np.nan,
            location='location',
            filtering='filtering',
            group=electrode_group
        )

    return nwbfile


class TestTetrodeSeriesConstructor(TestCase):

    def setUp(self):
        """Set up an NWB file. Necessary because TetrodeSeries requires references to electrodes."""
        self.nwbfile = set_up_nwbfile()

    def test_constructor(self):
        """Test that the constructor for TetrodeSeries sets values as expected."""
        all_electrodes = self.nwbfile.create_electrode_table_region(
            region=list(range(0, 10)),
            description='all the electrodes'
        )

        data = np.random.rand(100, 3)
        tetrode_series = TetrodeSeries(
            name='name',
            description='description',
            data=data,
            rate=1000.,
            electrodes=all_electrodes,
            trode_id=1
        )

        self.assertEqual(tetrode_series.name, 'name')
        self.assertEqual(tetrode_series.description, 'description')
        np.testing.assert_array_equal(tetrode_series.data, data)
        self.assertEqual(tetrode_series.rate, 1000.)
        self.assertEqual(tetrode_series.starting_time, 0)
        self.assertEqual(tetrode_series.electrodes, all_electrodes)
        self.assertEqual(tetrode_series.trode_id, 1)


class TestTetrodeSeriesRoundtrip(TestCase):
    """Simple roundtrip test for TetrodeSeries."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = 'test.nwb'

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """
        Add a TetrodeSeries to an NWBFile, write it to file, read the file, and test that the TetrodeSeries from the
        file matches the original TetrodeSeries.
        """
        all_electrodes = self.nwbfile.create_electrode_table_region(
            region=list(range(0, 10)),
            description='all the electrodes'
        )

        data = np.random.rand(100, 3)
        tetrode_series = TetrodeSeries(
            name='TetrodeSeries',
            description='description',
            data=data,
            rate=1000.,
            electrodes=all_electrodes,
            trode_id=1
        )

        self.nwbfile.add_acquisition(tetrode_series)

        with NWBHDF5IO(self.path, mode='w') as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(tetrode_series, read_nwbfile.acquisition['TetrodeSeries'])


class TestTetrodeSeriesRoundtripPyNWB(AcquisitionH5IOMixin, TestCase):
    """Complex, more complete roundtrip test for TetrodeSeries using pynwb.testing infrastructure."""

    def setUpContainer(self):
        """ Return the test TetrodeSeries to read/write """
        self.device = Device(
            name='device_name'
        )

        self.group = ElectrodeGroup(
            name='electrode_group',
            description='description',
            location='location',
            device=self.device
        )

        self.table = get_electrode_table()  # manually create a table of electrodes
        for i in range(10):
            self.table.add_row(
                x=i,
                y=i,
                z=i,
                imp=np.nan,
                location='location',
                filtering='filtering',
                group=self.group,
                group_name='electrode_group'
            )

        all_electrodes = DynamicTableRegion(
            data=list(range(0, 10)),
            description='all the electrodes',
            name='electrodes',
            table=self.table
        )

        data = np.random.rand(100, 3)
        tetrode_series = TetrodeSeries(
            name='name',
            description='description',
            data=data,
            rate=1000.,
            electrodes=all_electrodes,
            trode_id=1
        )
        return tetrode_series

    def addContainer(self, nwbfile):
        """Add the test TetrodeSeries and related objects to the given NWBFile."""
        nwbfile.add_device(self.device)
        nwbfile.add_electrode_group(self.group)
        nwbfile.set_electrode_table(self.table)
        nwbfile.add_acquisition(self.container)
