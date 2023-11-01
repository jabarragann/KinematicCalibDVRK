import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from kincalib.Transforms.Rotation import Rotation3D
from kincalib.Record.Record import CartesianRecord, JointRecord, RecordCollection


@pytest.mark.parametrize(
    "poses_tuple",
    [
        ([0.5, 0.0, 0.0], np.eye(3, 3)),
        ([-0.5, 0.0, 0.5], R.from_rotvec(np.pi / 2 * np.array([0, 0, 1])).as_matrix()),
        ([-500, 5.0, 500], R.from_rotvec(-np.pi / 6 * np.array([0.5, 0.5, 1])).as_matrix()),
    ],
)
def test_recover_cp_from_cartesian_record(poses_tuple):
    test_cp = np.eye(4)
    test_cp[:3, :3] = poses_tuple[1]
    test_cp[:3, 3] = poses_tuple[0]
    cp_record = CartesianRecord("test_record", "test")
    cp_record.add_data(0, test_cp)
    pos = cp_record.data_array[0][:3]
    rot = R.from_rotvec(cp_record.data_array[0][3:]).as_matrix()

    rec_cp = np.eye(4)
    rec_cp[:3, :3] = rot
    rec_cp[:3, 3] = pos

    assert np.all(np.isclose(rec_cp, test_cp))


def test_exception_for_multiple_records_with_same_headers():
    jp_rec1 = JointRecord("measured_jp1", "measured_")
    jp_rec2 = JointRecord("measured_jp2", "measured_")
    cp_rec = CartesianRecord("measured_cp", "measured_")

    with pytest.raises(AssertionError):
        rec_collection = RecordCollection([jp_rec1, jp_rec2, cp_rec])
