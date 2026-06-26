import numpy as np
import cv2
import mediapipe as mp

class GazeEngine:
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5):
        # 1. الاستدعاء الآمن داخل الكلاس
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )
        self.prev_landmarks = None

    def extract_gaze_features(self, frame: np.ndarray) -> np.ndarray:
        gaze_features = np.zeros(4, dtype=np.float32)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. تسريع المعالجة
        frame_rgb.flags.writeable = False 
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Yaw & Pitch approximation
            nose = landmarks[1]
            left_eye = landmarks[159]
            right_eye = landmarks[386]
            
            eye_center_x = (left_eye.x + right_eye.x) / 2.0
            eye_center_y = (left_eye.y + right_eye.y) / 2.0
            
            yaw = nose.x - eye_center_x
            pitch = nose.y - eye_center_y
            
            # Blink Rate (EAR)
            def compute_ear(eye_indices):
                h_dist = np.linalg.norm([landmarks[eye_indices[0]].x - landmarks[eye_indices[1]].x,
                                         landmarks[eye_indices[0]].y - landmarks[eye_indices[1]].y])
                v_dist = np.linalg.norm([landmarks[eye_indices[2]].x - landmarks[eye_indices[3]].x,
                                         landmarks[eye_indices[2]].y - landmarks[eye_indices[3]].y])
                return v_dist / (h_dist + 1e-6)
            
            left_ear = compute_ear([33, 133, 159, 145])
            right_ear = compute_ear([362, 263, 386, 374])
            ear = (left_ear + right_ear) / 2.0
            
            # Interaction Density
            interaction_density = 0.0
            if self.prev_landmarks is not None:
                motion = 0.0
                for idx in [1, 33, 263, 61, 291]: 
                    curr_pt = np.array([landmarks[idx].x, landmarks[idx].y])
                    prev_pt = np.array([self.prev_landmarks[idx].x, self.prev_landmarks[idx].y])
                    motion += np.linalg.norm(curr_pt - prev_pt)
                interaction_density = motion / 5.0
                
            self.prev_landmarks = landmarks
            
            gaze_features[0] = float(yaw)
            gaze_features[1] = float(pitch)
            gaze_features[2] = float(ear)
            gaze_features[3] = float(interaction_density)
            
        return gaze_features
        
    def close(self):
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()