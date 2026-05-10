import random
from collections import defaultdict


def extract_augmented_object_ids(all_instance_ids, necessary_object_ids, max_num=3):
    num_additional = random.randint(0, max_num)  # Random number between 0 and max_num
    available_ids = set(all_instance_ids) - set(necessary_object_ids)
    if available_ids:  # Only sample if there are available IDs
        augmented_object_ids = list(
            random.sample(list(available_ids), min(num_additional, len(available_ids)))
        )
    else:
        augmented_object_ids = []
    return augmented_object_ids


def reindex_visible_instance_ids(frames, instances):
    new_frames = []
    for frame_idx, frame in enumerate(frames):
        new_visible_instance_ids = []
        for old_id in frame["visible_instance_ids"]:
            new_visible_instance_ids.append(instances[old_id]["bbox_id"])
        frame["new_visible_instance_ids"] = new_visible_instance_ids
        new_frames.append(frame)
    return new_frames


def random_sample_frames(frames, sampled_object_ids, num_frames):
    """
    Sample frames while ensuring coverage of necessary objects and keep uniform.

    frames (list): List of frame dictionaries with 'visible_instance_ids' key
    sampled_object_ids (list): List of object IDs that must be visible
    num_frames (int): Number of frames to sample

    Returns:
    - sampled_frames: List of sampled frame dictionaries
    - object_frame_map: Dictionary mapping object IDs to frame indices where they appear
    """
    # Map each necessary object to frames where it appears
    object_to_frames = defaultdict(list)
    for frame_idx, frame in enumerate(frames):
        for obj_id in sampled_object_ids:
            # this is the re-indexed one
            if obj_id in frame["new_visible_instance_ids"]:
                object_to_frames[obj_id].append(frame_idx)

    # Check if any necessary object is not visible in any frame
    for obj_id in sampled_object_ids.copy():
        if not object_to_frames[obj_id]:
            print(
                f"Object ID {obj_id} is not visible in any frame, removed from object_ids"
            )
            sampled_object_ids.remove(obj_id)

    # Initialize selected frames
    selected_frame_indices = set()
    original_to_sampled = {}  # Maps original frame indices to sampled frame indices

    # First, ensure coverage of necessary objects
    for obj_id in sampled_object_ids:
        available_frames = [
            idx for idx in object_to_frames[obj_id] if idx not in selected_frame_indices
        ]
        if available_frames:
            frame_idx = random.choice(available_frames)
            selected_frame_indices.add(frame_idx)

    # Fill remaining slots with random frames
    remaining_slots = num_frames - len(selected_frame_indices)
    if remaining_slots > 0:
        available_indices = [
            i for i in range(len(frames)) if i not in selected_frame_indices
        ]
        if available_indices:
            additional_indices = random.sample(
                available_indices, min(remaining_slots, len(available_indices))
            )
            selected_frame_indices.update(additional_indices)

    # Convert indices to actual frames and build index mapping
    sorted_indices = sorted(selected_frame_indices)
    for new_idx, old_idx in enumerate(sorted_indices):
        original_to_sampled[old_idx] = new_idx

    sampled_frames = [frames[idx] for idx in sorted_indices]

    # Create object_frame_map with indices relative to sampled_frames
    object_frame_map = {}
    for obj_id in sampled_object_ids:
        for orig_idx in object_to_frames[obj_id]:
            if orig_idx in original_to_sampled:  # If this frame was selected
                object_frame_map[obj_id] = original_to_sampled[orig_idx]
                break

    # Check if we have enough frames
    if len(sampled_frames) < num_frames:
        raise ValueError(
            f"Could only sample {len(sampled_frames)} frames, "
            f"but {num_frames} were requested"
        )

    return sampled_frames, object_frame_map, sampled_object_ids


def build_instance_dict(instances):
    instance_dict = {}
    for instance in instances:
        instance_dict[instance["bbox_id"]] = instance
    return instance_dict


def add_objects_to_frames(sampled_frames, object_frame_map):
    # First, create a frame-to-objects mapping
    frame_object_map = {}
    for obj_id, frame_idx in object_frame_map.items():
        if frame_idx not in frame_object_map:
            frame_object_map[frame_idx] = []
        frame_object_map[frame_idx].append(obj_id)

    # Add object_ids to each frame
    for idx, frame in enumerate(sampled_frames):
        frame["object_ids"] = frame_object_map.get(idx, [])

    return sampled_frames


def extract_necessary_object_ids(entry):
    necessary_object_ids = []
    necessary_object_ids.append(entry["target_id"])
    necessary_object_ids.extend(entry["distractor_ids"])
    necessary_object_ids.extend(entry["anchor_ids"])
    return necessary_object_ids


# def sample_frames(frames, necessary_object_ids):
#     output = defaultdict(List)
#     for object_id in necessary_object_ids:
#         for frame_id, frame in enumerate(frames):
#             if object_id in frame["visible_instance_ids"]:
#                 output.append()


def contain_necessary(sampled_frames, necessary_object_ids):
    visible_ids = set()
    for frame in sampled_frames:
        visible_ids.update(frame["new_visible_instance_ids"])

    # Check if all necessary IDs are in the visible_ids set
    return all(obj_id in visible_ids for obj_id in necessary_object_ids)
