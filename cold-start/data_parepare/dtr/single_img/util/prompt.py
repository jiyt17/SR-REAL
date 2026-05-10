import random
from itertools import combinations

import numpy as np
from scipy.spatial import distance
from util.prompt_template import *
from util.prompt_utils import *


def wide_predicate(A, B):
    template_questions = wide_predicate_questions
    true_responses = wide_true_responses
    false_responses = wide_false_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    width_A = max(A_box.extent[0], A_box.extent[1])  # use max x & y to determine width
    width_B = max(B_box.extent[0], B_box.extent[1])  # use max x & y to determine width

    is_wider = width_A > width_B

    question_template = random.choice(template_questions)
    response_template = random.choice(true_responses if is_wider else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question, answer


def big_predicate(A, B):
    template_questions = big_predicate_questions
    true_responses = big_true_responses
    false_responses = big_false_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    extent_A = A_box.extent
    volume_A = extent_A[0] * extent_A[1] * extent_A[2]

    extent_B = B_box.extent
    volume_B = extent_B[0] * extent_B[1] * extent_B[2]

    is_bigger = volume_A > volume_B

    question_template = random.choice(template_questions)
    response_template = random.choice(true_responses if is_bigger else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question, answer


def tall_predicate(A, B):
    template_questions = tall_predicate_questions
    true_responses = tall_true_responses
    false_responses = tall_false_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    height_A = A_box.extent[2]  # use z for height
    height_B = B_box.extent[2]

    is_taller = height_A > height_B

    question_template = random.choice(template_questions)
    response_template = random.choice(true_responses if is_taller else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question, answer


def short_predicate(A, B):
    template_questions = short_predicate_questions
    true_responses = short_true_responses
    false_responses = short_false_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    height_A = A_box.extent[2]  # use z for height
    height_B = B_box.extent[2]

    is_shorter = height_A < height_B

    question_template = random.choice(template_questions)
    response_template = random.choice(true_responses if is_shorter else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question, answer


def thin_predicate(A, B):
    template_questions = thin_predicate_questions
    true_responses = thin_true_responses
    false_responses = thin_false_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    width_A = max(A_box.extent[0], A_box.extent[1])  # use max x & y to determine width
    width_B = max(B_box.extent[0], B_box.extent[1])  # use max x & y to determine width

    is_thinner = width_A < width_B

    question_template = random.choice(template_questions)
    response_template = random.choice(true_responses if is_thinner else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question, answer


def small_predicate(A, B):
    template_questions = small_predicate_questions
    true_responses = small_true_responses
    false_responses = small_false_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    extent_A = A_box.extent
    volume_A = extent_A[0] * extent_A[1] * extent_A[2]

    extent_B = B_box.extent
    volume_B = extent_B[0] * extent_B[1] * extent_B[2]

    is_smaller = volume_A < volume_B

    question_template = random.choice(template_questions)
    response_template = random.choice(true_responses if is_smaller else false_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return question, answer


def tall_choice(A, B):
    template_questions = tall_choice_questions
    template_responses = tall_choice_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    height_A = A_box.extent[2]  # use z for height
    height_B = B_box.extent[2]

    taller = A_desc if height_A > height_B else B_desc

    question_template = random.choice(template_questions)
    answer_template = random.choice(template_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = answer_template.replace("[X]", taller)

    return question, answer


def short_choice(A, B):
    template_questions = short_choice_questions
    template_responses = short_choice_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    height_A = A_box.extent[2]  # use z for height
    height_B = B_box.extent[2]

    shorter = A_desc if height_A < height_B else B_desc

    question_template = random.choice(template_questions)
    answer_template = random.choice(template_responses)

    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = answer_template.replace("[X]", shorter)

    return question, answer


def generate_spatial_reasoning_data(
    A, B, human_readable_dist, template_questions, template_answers
):
    A_desc, B_desc = A["caption"].lower(), B["caption"].lower()

    question_template = random.choice(template_questions)
    answer_template = random.choice(template_answers)

    # Replace placeholders with actual values
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = (
        answer_template.replace("[A]", A_desc)
        .replace("[B]", B_desc)
        .replace("[X]", human_readable_dist)
    )

    # Add to the dataset
    return (question, answer)


def min_distance_between_obb(obb1, obb2):
    # Get OBB corner points
    corners1 = np.asarray(obb1.get_box_points())
    corners2 = np.asarray(obb2.get_box_points())

    # Compute pairwise distances between corners
    min_dist = np.min(distance.cdist(corners1, corners2))
    return min_dist


def distance_data(A, B):
    template_questions = distance_template_questions
    template_answers = distance_template_answers

    human_readable_dist = human_like_distance(
        min_distance_between_obb(A["oriented_bbox"], B["oriented_bbox"])
    )

    return generate_spatial_reasoning_data(
        A, B, human_readable_dist, template_questions, template_answers
    )


def width_data(A):
    A_desc = A["caption"].lower()

    template_questions = width_questions
    template_answers = width_answers

    width = max(
        A["oriented_bbox"].extent[0], A["oriented_bbox"].extent[1]
    )  # use max x & y to determine width

    human_readable_width = human_like_distance(width)
    question_template = random.choice(template_questions)
    answer_template = random.choice(template_answers)

    question = question_template.replace("[A]", A_desc)
    answer = answer_template.replace("[A]", A_desc).replace("[X]", human_readable_width)

    return question, answer


def height_data(A):
    A_desc = A["caption"].lower()

    template_questions = height_questions
    template_answers = height_answers

    height = A["oriented_bbox"].extent[2]  # z dim is height

    human_readable_height = human_like_distance(height)
    question_template = random.choice(template_questions)
    answer_template = random.choice(template_answers)

    question = question_template.replace("[A]", A_desc)
    answer = answer_template.replace("[A]", A_desc).replace(
        "[X]", human_readable_height
    )

    return question, answer


# def distance_data(A, B):
#     distance = calculate_distances_between_point_clouds(A["pcd"], B["pcd"])
#     return generate_spatial_reasoning_data(A, B, distance, distance_template_questions, distance_template_answers)


def multi_closer(A, B, C):
    template_questions = multi_closer_questions
    template_responses = multi_closer_responses

    A_desc, A_box = A["caption"], A["oriented_bbox"]
    B_desc, B_box = B["caption"], B["oriented_bbox"]
    C_desc, C_box = C["caption"], C["oriented_bbox"]
    A_desc, B_desc, C_desc = A_desc.lower(), B_desc.lower(), C_desc.lower()

    distance_AB = min_distance_between_obb(A_box, B_box)
    distance_AC = min_distance_between_obb(A_box, C_box)

    question_template = random.choice(template_questions)
    answer_template = random.choice(template_responses)
    question = (
        question_template.replace("[A]", A_desc)
        .replace("[B]", B_desc)
        .replace("[C]", C_desc)
    )

    if distance_AC < distance_AB:  # c is closer
        answer = answer_template.replace("[X]", C_desc)
    else:
        answer = answer_template.replace("[X]", B_desc)

    return question, answer


class PromptGenerator:
    def __init__(self):
        """Initialize the class."""
        self.vis = True

    def evaluate_single_objects(self, obj_list):
        prompts = [width_data, height_data]

        results = []
        for obj in obj_list:
            for prompt_func in prompts:
                results.append((prompt_func(obj), [obj], prompt_func.__name__))
        return results

    def evaluate_two_objects(self, obj_list):
        all_combinations = list(combinations(range(len(obj_list)), 2))
        object_pairs = [(obj_list[i], obj_list[j]) for i, j in all_combinations]
        random.shuffle(object_pairs)

        # Grouping related prompts
        wide_thin = [wide_predicate, thin_predicate]
        big_small = [big_predicate, small_predicate]
        tall_short = [tall_predicate, tall_choice, short_predicate, short_choice]
        selected_prompts = [
            random.choice(wide_thin),
            random.choice(big_small),
            random.choice(tall_short),
            distance_data,  # Always included
        ]

        results = []

        for A, B in object_pairs[:10]:
            all_prompt_variants = [
                selected_prompt for selected_prompt in selected_prompts
            ]

            for prompt_func in all_prompt_variants:
                results.append((prompt_func(A, B), [A, B], prompt_func.__name__))

        return results

    def evaluate_three_objects(self, obj_list):
        all_combinations = list(combinations(range(len(obj_list)), 3))
        random.shuffle(all_combinations)
        object_pairs = [
            (obj_list[i], obj_list[j], obj_list[k]) for i, j, k in all_combinations
        ]
        random.shuffle(object_pairs)

        selected_prompts = [
            multi_closer,
        ]

        results = []

        for A, B, C in object_pairs[:5]:
            all_prompt_variants = [
                selected_prompt for selected_prompt in selected_prompts
            ]

            for prompt_func in all_prompt_variants:
                results.append((prompt_func(A, B, C), [A, B, C], prompt_func.__name__))

        return results
