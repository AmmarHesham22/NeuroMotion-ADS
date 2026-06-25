import numpy as np
import mediapipe as mp
import cv2

class GazeEngine:
    """
    Feature extraction class that translates raw video frames into our target 4-dimensional gaze/pose vector.
    """
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )
        self.prev_landmarks = None

    def extract_gaze_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a single frame and returns a numpy array of shape [4] representing 
        [yaw, pitch, blink_rate, interaction_density].
        """
        # Default gaze array
        gaze_features = np.zeros(4, dtype=np.float32)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. Yaw & Pitch approximation
            # Basic 2D approximation based on normalized landmarks
            nose = landmarks[1]
            left_eye = landmarks[159]
            right_eye = landmarks[386]
            
            eye_center_x = (left_eye.x + right_eye.x) / 2.0
            eye_center_y = (left_eye.y + right_eye.y) / 2.0
            
            # Relative displacement of nose from eye center
            yaw = nose.x - eye_center_x
            pitch = nose.y - eye_center_y
            
            # 2. Blink Rate (Eye Aspect Ratio - EAR)
            def compute_ear(eye_indices):
                # Horizontal distance
                h_dist = np.linalg.norm([landmarks[eye_indices[0]].x - landmarks[eye_indices[1]].x,
                                         landmarks[eye_indices[0]].y - landmarks[eye_indices[1]].y])
                # Vertical distance
                v_dist = np.linalg.norm([landmarks[eye_indices[2]].x - landmarks[eye_indices[3]].x,
                                         landmarks[eye_indices[2]].y - landmarks[eye_indices[3]].y])
                return v_dist / (h_dist + 1e-6)
            
            # Key indices for left eye: 33 (outer), 133 (inner), 159 (top), 145 (bottom)
            left_ear = compute_ear([33, 133, 159, 145])
            # Key indices for right eye: 362 (outer), 263 (inner), 386 (top), 374 (bottom)
            right_ear = compute_ear([362, 263, 386, 374])
            ear = (left_ear + right_ear) / 2.0
            
            # 3. Interaction Density (motion approximation)
            interaction_density = 0.0
            if self.prev_landmarks is not None:
                motion = 0.0
                # Sum of displacements for key anchor points
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
        """Release MediaPipe resources."""
        self.face_mesh.close()
