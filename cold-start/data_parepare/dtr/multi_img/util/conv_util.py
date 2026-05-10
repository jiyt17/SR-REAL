# from matplotlib.path import Path
import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_question(text, num_masks):
    question = text.rstrip("?")

    if question.lower().startswith("the"):
        question = "find " + question

    if num_masks > 1:
        choice = random.choice(
            [
                " ".join(["<mask>"] * num_masks),
                " ".join(["<mask>"] * (num_masks - 1)) + " and <mask>",
                ", ".join(["<mask>"] * (num_masks - 1)) + ", and <mask>",
            ]
        )
        masks = choice
    else:
        masks = "<mask>"

    question_templates = [
        f"Among {masks}, {question[0].lower() + question[1:] if question else ''}.",
        f"{question[0].upper() + question[1:] if question else ''} from {masks}.",
        f"Given {masks}, {question[0].lower() + question[1:] if question else ''}.",
        f"Considering {masks}, {question[0].lower() + question[1:] if question else ''}.",
        f"Out of {masks}, {question[0].lower() + question[1:] if question else ''}.",
        f"From the following: {masks}, {question[0].lower() + question[1:] if question else ''}.",
        f"Between {masks}, {question[0].lower() + question[1:] if question else ''}.",
        f"When presented with {masks}, {question[0].lower() + question[1:] if question else ''}.",
        f"With {masks} available, {question[0].lower() + question[1:] if question else ''}.",
        f"Which option among {masks} best fits: {question[0].lower() + question[1:] if question else ''}?",
        f"Based on {masks}, how would you answer: {question[0].lower() + question[1:] if question else ''}?",
        f"Can you determine the answer to '{question}' from {masks}?",
        f"Imagine the choices are {masks}. {question[0].upper() + question[1:] if question else ''}.",
        f"Select one from {masks} and answer: {question[0].lower() + question[1:] if question else ''}.",
        f"If you have {masks} to choose from, {question[0].lower() + question[1:] if question else ''}.",
        f"Out of these options: {masks}, {question[0].lower() + question[1:] if question else ''}.",
        f"{masks} are presented. {question[0].upper() + question[1:] if question else ''}.",
        f"Which of the following {masks} answers: {question[0].lower() + question[1:] if question else ''}?",
        f"Among these choices: {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"Provided {masks}, which option answers: {question[0].lower() + question[1:] if question else ''}?",
        f"If you consider {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"What is the best answer to '{question}' from {masks}?",
        f"Among {masks}, which would you choose for: {question[0].lower() + question[1:] if question else ''}?",
        f"How would you answer '{question}' given {masks}?",
        f"Thinking about {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"If your options are {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"What would you pick from {masks} for: {question[0].lower() + question[1:] if question else ''}?",
        f"Based on the options {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"From this list: {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"Under the scenario of {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"When considering {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"Between these choices: {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"Given these possibilities: {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"From the available options ({masks}), {question[0].lower() + question[1:] if question else ''}?",
        f"Selecting among {masks}, {question[0].lower() + question[1:] if question else ''}?",
        f"Choose from {masks} to answer: {question[0].lower() + question[1:] if question else ''}.",
        f"Imagine these options {masks}. {question[0].upper() + question[1:] if question else ''}.",
        f"Picture the choices: {masks}. {question[0].upper() + question[1:] if question else ''}.",
        f"Review the options {masks} and answer: {question[0].lower() + question[1:] if question else ''}.",
        f"Assessing {masks}, how would you answer: {question[0].lower() + question[1:] if question else ''}?",
        f"What answer fits '{question}' when considering {masks}?",
    ]

    return random.choice(question_templates)


def generate_answer(target_mask_id, target):

    answer_templates = [
        # Short answers
        f"Region [{target_mask_id}].",
        f"Region [{target_mask_id}]",
        f"It's in Region [{target_mask_id}].",
        f"In Region [{target_mask_id}].",
        # Basic descriptions
        f"The {target} is at Region [{target_mask_id}].",
        f"The {target} is in Region [{target_mask_id}].",
        f"The {target} is located in Region [{target_mask_id}].",
        f"You can find the {target} in Region [{target_mask_id}].",
        f"Region [{target_mask_id}] contains the {target}.",
        # More varied descriptions
        f"Look for the {target} in Region [{target_mask_id}].",
        f"The {target} can be found at Region [{target_mask_id}].",
        f"Region [{target_mask_id}] is where the {target} is.",
        f"The {target} is situated in Region [{target_mask_id}].",
        f"Check Region [{target_mask_id}] for the {target}.",
        f"Within Region [{target_mask_id}], you'll find the {target}.",
        # Additional variations
        f"The {target} is positioned in Region [{target_mask_id}].",
        f"You'll see the {target} in Region [{target_mask_id}].",
        f"Head to Region [{target_mask_id}] for the {target}.",
        f"The {target} is placed in Region [{target_mask_id}].",
        f"Region [{target_mask_id}] houses the {target}.",
        f"Navigate to Region [{target_mask_id}] to find the {target}.",
        f"The {target} exists in Region [{target_mask_id}].",
        f"Region [{target_mask_id}] - that's where the {target} is.",
        # Imperative forms
        f"Go to Region [{target_mask_id}].",
        f"Check Region [{target_mask_id}].",
        f"Look in Region [{target_mask_id}].",
        f"Search Region [{target_mask_id}].",
        # Brief answers with context
        f"It's at Region [{target_mask_id}].",
        f"Found in Region [{target_mask_id}].",
        f"Located in Region [{target_mask_id}].",
        f"Present in Region [{target_mask_id}].",
        f"Inside Region [{target_mask_id}].",
    ]

    return random.choice(answer_templates)


def convert_text_to_question(text: str) -> str:
    """Convert the input text to a question."""
    if not text.strip().endswith("?"):
        return f"{text.strip()}?"
    return text.strip()


# def generate_conversation(index: int, scan_id: str, text: str, target_id: int, target: str) -> Dict:
#     """Generate a single conversation entry."""
#     question = convert_text_to_question(text)
#     return {
#         "id": str(index),
#         "image": str(index),
#         "depth":
#         "rle":
#         "conversations": [
#             {
#                 "from": "human",
#                 "value": question
#             },
#             {
#                 "from": "gpt",
#                 "value": answer
#             }
#         ]
#     }
