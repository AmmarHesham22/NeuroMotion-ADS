import numpy as np
import cv2
import mediapipe as mp

class PoseEngine:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        # 1. الاستدعاء الآمن داخل الكلاس يمنع الـ Startup Crash
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. تسريع الـ C++ Backend ومنع تعارض الذاكرة
        frame_rgb.flags.writeable = False 
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        joints = np.zeros((17, 3), dtype=np.float32)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for coco_idx, mp_idx in self.coco_to_mp.items():
                lm = landmarks[mp_idx]
                x = lm.x
                y = lm.y
                c = lm.visibility if lm.visibility >= visibility_threshold else 0.0
                joints[coco_idx] = [x, y, c]
                
        return joints

    def close(self):
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()