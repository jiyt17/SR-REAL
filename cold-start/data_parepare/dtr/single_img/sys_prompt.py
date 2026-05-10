# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

SPAR_3dcot_generate_prompt = """
# ROLE: Spatial Reasoning Chain-of-thinking Generator

## TASK:
Given a multiple-choice spatial reasoning question, a scene image, the correct answer option, and center 3D coordinates of relevant objects, generate a **step-by-step reasoning chain** that logically derives the correct choice. Your output must demonstrate precise spatial reasoning based on the provided scene context and object positions.

## INPUT CONTEXT:
You will receive the following:
1.  **Image:** An image depicting the 3D scene.
2.  **Question:** A multiple-choice question about the image. The qustion is about some objects annotated by bounding boxes or points in the image.
3.  **Answer:** The ground truth answer index to the question. (e.g., "A", "B", "C", or "D")
4.  **Scene Graph:** A dictionary containing center 3D coordinates of object identifiers in the format: {'red_point': [-0.1, 0.2, 1.4]}

## COORDINATE SYSTEM (Camera Coordinates):
* **+x axis:** Left to Right
* **+y axis:** Top to Bottom
* **+z axis:** Near to Far (depth)
* Note that the x y z coordinate is from the perspective of camera, if the question changes the observer to a different position and direction, the right, above, front of observer can not correspond with original x y z.

## TASK:
Your primary task is to analyze 3D visual scenes within the image, understand the `Question` and `Answer` accurately, combine center positions of objects, then generate chain-of-thinking for how to get the correct answer.
* Use the `Image` to understand scene context, object appearances, and spatial relationship.
* Use the `Scene Graph` to get precise 3D positions of objects mentioned in the question. The scene graph is about annotations (bbox or point), and you need to match the corresponding objects in the question.
* Use the `Answer` to validate your reasoning process, your reasoning must be able to arrive at the correct answer.

## GUIDELINES & CONSTRAINTS: 
* The reasoning process must be consistent with the spatial information in the image.
* The reasoning process must clearly solve the question, and derive the correct answer.
* The word number of reasoning process should be between 100 and 200.
* Keep the mathematical calculations brief, no need to output too many formulas (<3).
* Mainly use the visual content for relationship judgement, with 3D coordinate calculations as supplementary or for verification.
* No need for explicit output of step 1, step 2 ... or 1. 2. ...

## A REFERENCE FOR REASONING STRUCTURE
1. Scene interpretation
2. Analysis: Position (horizontal/vertical/depth), relationship, distance ...
3. Computaion: calculate or compare the 3D coordinates to support spatial reasoning
4. Inference: solve the question, describe the thinking process
5. Conclusion: matching correct answer, e.g., "So the answer is 'A'."

## OUTPUT FORMAT:
Provide the reasoning process in the string format:
* (String) The generated explanation of how the answer was derived, referencing image context and 3D coordinates of objects. The word number of reasoning process should be less than 200.

## EXAMPLE:

* **Image:** [A table with two chairs]
* **Question:** ""Where is chair (red bbox) initially from the observers perspective? How does the position of chair change once the observer moves to the center of table (green bbox) and faces chair (blue bbox)? Calculate or judge based on the 3D center points of these objects. Please use the world coordinate system to determine the up and down position relationship of objects.\nFor multiple-choice questions, consider only the state after the observer has moved.\nThe options describe the spatial relationship between object and observer in terms of left-right (left, right, or empty if indistinguishable), above-below (above, below, or empty if indistinguishable), and front-behind (front, behind, or empty if indistinguishable).\nChoose the appropriate option from the choices provided.\nA. left, above, \nB. right, above, front\nC. left, , front\nD. right, below, behind\nYour answer can only include one of options A, B, C or D."
* **Answer:** "C"
* **Scene Graph:** {'red_bbox': [0.31, -0.27, 1.3], 'blue_bbox': [0.02, 0.26, 0.84], 'green_bbox': [-0.34, -0.08, 1.19]}

* **Expected Output:**
    "
    Assuming observer standing at the center of the table (green bbox), facing the chair (blue bbox), we need to determine the position of the chair (red bbox) from this new viewpoint. From visual content, this is a top-down image, when the user stands at the green bbox and line-of-sight is from the green bbox to the blue bbox, the chair in the red bbox is to the left and in front of the observer.
    Using the 3D coordinates for verification: 
    The observer's new position is the center of the green bbox: `P_obs = [-0.34, -0.08, 1.19]`. Facing the blue bbox: `P_face = [0.02, 0.26, 0.84]`. The target red bbox: `P_target = [0.31, -0.27, 1.3]`.
    The direction vector from the observer to the target: `V_relative = P_target - P_obs = [0.65, -0.19, 0.11]`. The observer's line-of-sight vector: `V_forward = P_face - P_obs = [0.36, 0.34, -0.35]`.
    The dot product of `V_relative` and `V_forward` is positive, indicating the target (red bbox) is in **front** of the observer.
    Analyzing left/right, the red bbox is to the **left** of the line of sight from green to blue bbox.
    For above/below, because it is a top-down image, when the user stands at the green bbox, z value indicates above or below, the difference of observer and target in z-coordinates is negligible, so we leave it blank.
    So the answer is 'left, , front'.
    "
"""

SPAR_3dcot_multiview_generate_prompt = """
# ROLE: Multi-View Spatial Reasoning Chain-of-thinking Generator

## TASK:
Given a multiple-choice spatial reasoning question, a set of images from different perspectives, the index of a "Reference Frame," 3D coordinates of objects relative to that reference frame, and the correct answer, generate a **step-by-step reasoning chain**. Your output must demonstrate how to align objects across multiple views and use the 3D coordinates of the reference frame to derive the correct answer.

## INPUT CONTEXT:
1. **Images:** A sequence of images depicting the same 3D scene from various angles.
2. **Reference Frame Index:** An integer indicating which image's camera coordinate system is used for the provided 3D coordinates (e.g., "Frame 0").
3. **Question:** A multiple-choice question involving objects seen across different images.
4. **Answer:** The ground truth answer index (e.g., "A", "B").
5. **Scene Graph (Aligned to Reference Frame):** A dictionary where each object identifier is mapped to its coordinates **relative to the Reference Frame**:
   - `center3d`: [x, y, z] (3D coordinates in the Reference Frame's camera space).

## COORDINATE SYSTEM (Reference Frame Camera):
* **Origin:** Optical center of the camera of the "Reference Frame".
* **2D Bounding Box:** x increases from left to right, y increases from top to bottom.
* **3D Camera Coordinates:** +x axis: Rightward | +y axis: Downward | +z axis: Forward/far (Depth).
* **Cross-View Alignment:** The `center3d` provided for all objects (even those primarily visible in non-reference images) are already aligned/projected into the Reference Frame's coordinate system.

## TASK GUIDELINES:
* **Object Alignment:** First, identify the objects mentioned in the question across different images. Map them to the corresponding `object_id` in the Scene Graph of the Reference Frame.
* **Spatial Synthesis:** Use visual cues from all images to understand the layout, but rely on the `center3d` coordinates from the Reference Frame for precise spatial judgments (left/right, above/below, near/far).
* **Reasoning Flow:** 1. Identify the target objects in the images.
    2. Note their 3D positions from the Scene Graph.
    3. Perform spatial comparisons (e.g., comparing $x$ for left/right, $z$ for depth).
    4. Link the calculation/comparison back to the question's perspective.
* **Constraints:** * Word count: 100-300 words.
    * Limit formulas to essential comparisons.
    * Ensure the conclusion matches the provided `Answer`, e.g., "So the answer is 'A'."

## OUTPUT FORMAT:
A string representing the reasoning process. No "Step 1, Step 2" labels.

## EXAMPLE:

* **Images:** [Img_0 (Reference), Img_1]
* **Reference Frame Index:** 0
* **Question:** "Which is closer to the sofa (red_bbox): the coffee table (blue_bbox) seen in Img_0 or the lamp (green_bbox) seen in Img_1?
    A. Coffee table
    B. Lamp"
* **Answer:** "A"
* **Scene Graph (Aligned to Img_0):** {
    'red_bbox': {'center3d': [0.5, 0.0, 3.5]}, 
    'blue_bbox': {'center3d': [0.2, 0.4, 2.8]},
    'green_bbox': {'center3d': [-1.2, -0.5, 5.0]} 
  }

* **Expected Output:**
    "In the reference frame (Img_0), we identify the sofa (red_bbox) on the right and the coffee table (blue_bbox) near the center. The lamp (green_bbox), though more clearly visible in Img_1, is projected onto Img_0 at the far left (x-range 0.10-0.20). To determine proximity, we use the 3D coordinates relative to Img_0's camera. The sofa is at [0.5, 0.0, 3.5]. The coffee table's depth (z=2.8) is closer to the camera than the sofa, with a relative distance of approximately 0.85m ($\sqrt{0.3^2 + 0.4^2 + 0.7^2}$). In contrast, the lamp is much further away at a depth of z=5.0 and an x-position of -1.2. The distance between the sofa and the lamp exceeds 2 meters. Therefore, despite the perspective shifts across images, the 3D data confirms the coffee table is significantly closer to the sofa than the lamp is. So the answer is 'A'."
"""


anno_instance_prompt = """
## TASK:
You are responsible for extracting instance names from a spatial question according to visual annotations.

## INPUT CONTEXT:
You will receive the following:
1.  **Ques:** A question containing instances and visual annotations, e.g., tv stand (blue bbox).
2.  **Anno:** (List format) A list of all color annotations in the question.

## GUIDELINES & CONSTRAINTS: 
* Extract the instance names corresponding to each color annotation.

## OUTPUT FORMAT:
Provide the output as a single JSON object, where each key is a color annotation from the `Anno` list, and its value is the corresponding instance name extracted from the `Ques`.
* (JSON) A JSON object mapping each color annotation to its corresponding instance name.

## EXAMPLE:

* **Ques:** "What is the linear distance in meters from the red point representing kitchen cabinets to the blue point representing microwave? Calculate or judge based on the 3D center points of these objects. Choose the correct response from the given choices.\nA. 0.6\nB. 1.0\nC. 1.3\nD. 2.1\nYour answer can only include one of options A, B, C or D."
* **Anno:** "['red point', 'blue point']"

* **Expected Output:**
    ```json
    {
        "red point": "kitchen cabinets",
        "blue point": "microwave"
    }
"""