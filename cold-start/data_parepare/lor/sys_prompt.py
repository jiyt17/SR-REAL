# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

SPAR_cot_generate_prompt = """
# ROLE: Spatial Reasoning Chain-of-thinking Generator

## TASK:
Given a multiple-choice spatial reasoning question, a scene image, and the correct answer option, generate a **step-by-step reasoning chain** that logically derives the correct choice. Your output must demonstrate precise spatial reasoning based on the provided scene context.

## INPUT CONTEXT:
You will receive the following:
1.  **Image:** An image depicting the 3D scene.
2.  **Question:** A multiple-choice question about the image.
3.  **Answer:** The ground truth answer index to the question. (e.g., "A", "B", "C", or "D")

## TASK:
Your primary task is to analyze 3D visual scenes within the image, understand the `Question` and `Answer` accurately, then generate chain-of-thinking for how to get the correct answer.
* Use the `Image` to understand scene context, object appearances, and spatial relationship.
* Use the `Answer` to validate your reasoning process, your reasoning must be able to arrive at the correct answer.

## GUIDELINES & CONSTRAINTS: 
* The reasoning process must be consistent with the spatial information in the image.
* The reasoning process must clearly solve the question, and derive the correct answer.
* The word number of reasoning process should be between 100 and 200.
* No need for explicit output of step 1, step 2 ...

## A REFERENCE FOR REASONING STRUCTURE
1. Scene interpretation
2. Analysis: Position (horizontal/vertical/depth), relationship, distance ...
3. Inference: solve the question, describe the thinking process
4. Conclusion: matching correct answer

## OUTPUT FORMAT:
Provide the output in the string format (the generated reaoning process).

## EXAMPLE:

* **Image:** [A bathroom photo, with two towels]
* **Question:** "In the given image, what is the position of the towel (red bbox) relative to the towel (blue bbox), according to the observer? Calculate or judge based on the 3D center points of these objects.\nThe options describe the spatial relationship between two objects.\nSelect the right option from the choices provided.\nA. left, above, closer\nB. left, , closer\nC. right, below, farther\nD. right, , farther\nYour answer can only include one of options A, B, C or D"
* **Answer:** "D"

* **Expected Output:**
    "The image shows two towels, one inside a red bounding box and the other inside a blue bounding box. The red towel is positioned to the right of the blue towel, and both towels are at the same vertical level, indicating they are at the same height. As the blue towel is positioned in front of the red towel, the red towel is also farther to the viewer than the blue towel. So the answer is 'right, , farther'. "
"""

SPAR_multiview_cot_generate_prompt = """
# ROLE: Spatial Reasoning Chain-of-thinking Generator

## TASK:
Given a multiple-choice spatial reasoning question, multiple scene images, and the correct answer option, generate a **step-by-step reasoning chain** that logically derives the correct choice. Your output must demonstrate precise spatial reasoning based on the provided scene context.

## INPUT CONTEXT:
You will receive the following:
1.  **Images:** Multiple images depicting the 3D scene.
2.  **Question:** A multiple-choice question about the image. The question may be about spatial relationships, distance, depth, camera motion and so on.
3.  **Answer:** The ground truth answer index to the question. (e.g., "A", "B", "C", or "D")

## TASK:
Your primary task is to analyze 3D visual scenes within the images, understand the `Question` and `Answer` accurately, then generate chain-of-thinking for how to get the correct answer.
* Use the `Images` to understand scene context, object appearances, and spatial relationship.
* Use the `Answer` to validate your reasoning process, your reasoning must be able to arrive at the correct answer.

## GUIDELINES & CONSTRAINTS: 
* The reasoning process must be consistent with the spatial information in the image.
* The reasoning process must clearly solve the question, and derive the correct answer. 
* The word number of reasoning process should be between 100 and 200.
* No need for explicit output of step 1, step 2 ...

## A REFERENCE FOR REASONING STRUCTURE
1. Scene interpretation
2. Analysis: Position (horizontal/vertical/depth), relationship, distance ...
3. Inference: solve the question, describe the thinking process
4. Conclusion: matching correct answer, e.g., "So the answer is 'A'."

## OUTPUT FORMAT:
Provide the output in the string format (the generated reaoning process).

## EXAMPLE:

* **Images:** [A bathroom photo, with two towels]
* **Question:** "In the given images, what is the position of the towel (red bbox) relative to the towel (blue bbox), according to the observer? Calculate or judge based on the 3D center points of these objects.\nThe options describe the spatial relationship between two objects.\nSelect the right option from the choices provided.\nA. left, above, closer\nB. left, , closer\nC. right, below, farther\nD. right, , farther\nYour answer can only include one of options A, B, C or D"
* **Answer:** "D"

* **Expected Output:**
    "The first image shows a towel inside a red bounding box, and the second image shows a towel in a blue bounding box. The sink appears in both images. The red towel is positioned to the right of the sink, and the blue towel is at the left of the sink, so the red towel is to the right of the blue towel. Both towels are at the same vertical level, indicating they are at the same height. As the blue towel is positioned in front of the sink, the red towel is also farther to the viewer than the blue towel. So the answer is 'right, , farther', corresponding to 'D'. "
"""