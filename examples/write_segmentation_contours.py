"""
Example script that demonstrates how to write segmentation contours alongside pose estimates.

With one camera, one skeleton, three body parts, and a ContourSeries holding the outline of the
segmented animal on every frame, including frames where an occluder splits the animal in two.
"""

import datetime
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from ndx_pose import (
    ContourSeries,
    PoseEstimation,
    PoseEstimationSeries,
    Skeleton,
    Skeletons,
)

num_frames = 100
frame_rate = 30.0

# initialize an NWBFile object
nwbfile = NWBFile(
    session_description="session_description",
    identifier="identifier",
    session_start_time=datetime.datetime.now(datetime.timezone.utc),
)

# add a subject to the NWB file
subject = Subject(subject_id="subject1", species="Mus musculus")
nwbfile.subject = subject

# create a skeleton that defines the relationship between the markers
skeleton = Skeleton(
    name="subject1_skeleton",
    nodes=["front_left_paw", "body", "front_right_paw"],
    edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
    subject=subject,
)
skeletons = Skeletons(skeletons=[skeleton])

# create a PoseEstimationSeries for each body part
pose_estimation_series = []
for name in skeleton.nodes[:]:
    pose_estimation_series.append(
        PoseEstimationSeries(
            name=name,
            description=f"Marker placed on {name}.",
            data=np.random.rand(num_frames, 2),  # num_frames x (x, y)
            unit="pixels",
            reference_frame="(0,0) is the top left corner of the video frame.",
            rate=frame_rate,
            confidence=np.random.rand(num_frames),
        )
    )

# Contours vary in number and length from frame to frame, so 'data' is padded out to the
# largest of each and 'vertex_count' records how much of every slot is real. This animal needs
# at most three contour slots: on most frames one outer boundary, sometimes with a hole, and on
# a few frames an occluder splits it into two parts and one part holds the hole.
max_contours = 3
max_vertices = 50

contour_data = np.full((num_frames, max_contours, max_vertices, 2), -1, dtype=np.int32)
vertex_count = np.zeros((num_frames, max_contours), dtype=np.uint32)
is_external = np.zeros((num_frames, max_contours), dtype=bool)
# contour_group says which connected component each contour belongs to. Contours of one
# component share a value, and a hole carries the value of the component containing it, so on a
# frame with two parts it is unambiguous which part the hole belongs to.
contour_group = np.zeros((num_frames, max_contours), dtype=np.uint32)

rng = np.random.default_rng(0)


def random_contour(n, origin):
    """Return n random vertices near an origin, standing in for a real outline."""
    return origin + rng.integers(0, 60, size=(n, 2))


for frame in range(num_frames):
    if frame % 7 == 0:
        # an occluder splits the animal: two outer boundaries, plus a hole in the second part
        n_a, n_b, n_hole = rng.integers(20, 40), rng.integers(20, 40), rng.integers(5, 15)
        contour_data[frame, 0, :n_a] = random_contour(n_a, [0, 0])
        contour_data[frame, 1, :n_b] = random_contour(n_b, [200, 0])
        contour_data[frame, 2, :n_hole] = random_contour(n_hole, [220, 20])
        vertex_count[frame] = [n_a, n_b, n_hole]
        is_external[frame] = [True, True, False]
        contour_group[frame] = [0, 1, 1]  # the hole belongs to the second part
    else:
        # one outer boundary, with a hole on some frames
        n_outer = rng.integers(20, max_vertices + 1)
        contour_data[frame, 0, :n_outer] = random_contour(n_outer, [0, 0])
        vertex_count[frame, 0] = n_outer
        is_external[frame, 0] = True
        if frame % 3 == 0:
            n_hole = rng.integers(5, 15)
            contour_data[frame, 1, :n_hole] = random_contour(n_hole, [20, 20])
            vertex_count[frame, 1] = n_hole
            is_external[frame, 1] = False
        # every contour on this frame belongs to the single component, group 0

contour_series = ContourSeries(
    name="contours",
    description="Outline of the segmented animal, with holes and occluder-split parts.",
    data=contour_data,
    reference_frame="(0,0) is the top left corner of the video frame.",
    vertex_count=vertex_count,
    is_external=is_external,
    contour_group=contour_group,
    unit="pixels",
    rate=frame_rate,
)

# store the contours in the same PoseEstimation object as the pose estimates, so both describe
# the same instance of the same subject
pose_estimation = PoseEstimation(
    name="PoseEstimation",
    pose_estimation_series=pose_estimation_series,
    contour_series=[contour_series],
    description="Estimated positions and segmentation contours of subject1.",
    source_software="DeepLabCut",
    source_software_version="2.3.8",
    skeleton=skeleton,
)

behavior_pm = nwbfile.create_processing_module(
    name="behavior",
    description="processed behavioral data",
)
behavior_pm.add(skeletons)
behavior_pm.add(pose_estimation)

path = "test_pose.nwb"
with NWBHDF5IO(path, mode="w") as io:
    io.write(nwbfile)

# read the file back and unpad the contours of a frame where the animal is split in two
with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
    read_nwbfile = io.read()
    read_contours = read_nwbfile.processing["behavior"]["PoseEstimation"].contour_series["contours"]

    frame = 0
    print(f"frame {frame}:")
    for slot in range(read_contours.data.shape[1]):
        n = read_contours.vertex_count[frame, slot]
        if n == 0:
            continue  # this slot holds no contour on this frame
        vertices = read_contours.data[frame, slot, :n, :]
        boundary = "external" if read_contours.is_external[frame, slot] else "internal"
        group = read_contours.contour_group[frame, slot]
        print(f"  contour {slot}: {n:2d} vertices, {boundary} boundary, part {group}")
        print(f"    first vertex (x, y): {vertices[0].tolist()}")

    # group the frame's contours into whole parts, each an outer boundary plus its own holes
    groups = np.asarray(read_contours.contour_group[frame])
    external = np.asarray(read_contours.is_external[frame])
    counts = np.asarray(read_contours.vertex_count[frame])
    for part in np.unique(groups[counts > 0]):
        members = np.flatnonzero((groups == part) & (counts > 0))
        outer = [int(i) for i in members if external[i]]
        holes = [int(i) for i in members if not external[i]]
        print(f"  part {int(part)}: outer boundary {outer}, holes {holes}")
