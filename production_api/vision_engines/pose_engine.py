import numpy as np
import mediapipe as mp
import cv2

class PoseEngine:
    """
    Feature extraction class that translates raw video frames into our target 17-joint tensor
    using MediaPipe Pose.
    """
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )
        
        # Mapping from our 17 COCO joints to MediaPipe pose landmark indices
        self.coco_to_mp = {
            0: 0,   # Nose
            1: 2,   # L_Eye
            2: 5,   # R_Eye
            3: 7,   # L_Ear
            4: 8,   # R_Ear
            5: 11,  # L_Shoulder
            6: 12,  # R_Shoulder
            7: 13,  # L_Elbow
            8: 14,  # R_Elbow
            9: 15,  # L_Wrist
            10: 16, # R_Wrist
            11: 23, # L_Hip
            12: 24, # R_Hip
            13: 25, # L_Knee
            14: 26, # R_Knee
            15: 27, # L_Ankle
            16: 28  # R_Ankle
        }

    def extract_frame_joints(self, frame: np.ndarray, visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Processes a single frame and returns a numpy array of shape [17, 3] representing [x, y, c]
        for the 17 target COCO joints.
        
        If visibility drops below visibility_threshold, confidence 'c' is explicitly set to 0.0.
        """
        # Convert BGR (OpenCV default) to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        # Initialize the output array [17, 3] (x, y, confidence)
        joints = np.zeros((17, 3), dtype=np.float32)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for coco_idx, mp_idx in self.coco_to_mp.items():
                lm = landmarks[mp_idx]
                
                # MediaPipe returns normalized coordinates [0.0, 1.0]
                # We keep them normalized to maintain spatial invariance
                x = lm.x
                y = lm.y
                
                # Use visibility as confidence. If below threshold, set c to 0.0
                c = lm.visibility if lm.visibility >= visibility_threshold else 0.0
                
                joints[coco_idx] = [x, y, c]
                
        return joints

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
