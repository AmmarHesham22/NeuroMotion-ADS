import numpy as np

class StreamChunker:
    """
    Maintains a rolling buffer of incoming skeleton and gaze arrays and yields
    model-ready chunks of exactly `window_size` frames.
    """
    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.skeleton_buffer = []
        self.gaze_buffer = []
        self.timestamps = []
        
    def add_frame(self, skeleton: np.ndarray, gaze: np.ndarray, timestamp: float):
        """
        Ingests live data frames.
        
        Args:
            skeleton: [17, 3] numpy array representing the 17 COCO joints (x, y, confidence)
            gaze: [4] numpy array representing [yaw, pitch, ear, interaction_density]
            timestamp: The timestamp of the current frame
        """
        self.skeleton_buffer.append(skeleton)
        self.gaze_buffer.append(gaze)
        self.timestamps.append(timestamp)
        
    def get_chunks(self, overlap_frames: int = 150) -> list:
        """
        Yields synced (skeleton_chunk, gaze_chunk) if the buffer exceeds the window_size,
        and slides the window forward by the overlap amount.
        
        Returns:
            list of tuples: [(skeleton_chunk, gaze_chunk), ...]
            where skeleton_chunk is [3, 300, 17] and gaze_chunk is [300, 4]
        """
        chunks = []
        
        # Keep processing chunks as long as we have enough frames
        while len(self.skeleton_buffer) >= self.window_size:
            # Extract exactly window_size frames
            skel_seq = np.stack(self.skeleton_buffer[:self.window_size]) # Shape: [300, 17, 3]
            gaze_seq = np.stack(self.gaze_buffer[:self.window_size])     # Shape: [300, 4]
            
            # The model expects skeleton as [Channels, Frames, Vertices] -> [3, 300, 17]
            # Currently skel_seq is [Frames, Vertices, Channels] -> [300, 17, 3]
            # We transpose: axis 2 (C) to 0, axis 0 (T) to 1, axis 1 (V) to 2
            skel_chunk = np.transpose(skel_seq, (2, 0, 1)).astype(np.float32)
            
            # The model expects gaze as [Frames, Dims] -> [300, 4]
            gaze_chunk = gaze_seq.astype(np.float32)
            
            chunks.append((skel_chunk, gaze_chunk))
            
            # Slide window forward
            slide_amount = self.window_size - overlap_frames
            if slide_amount <= 0:
                raise ValueError("Overlap frames must be strictly less than window_size.")
                
            self.skeleton_buffer = self.skeleton_buffer[slide_amount:]
            self.gaze_buffer = self.gaze_buffer[slide_amount:]
            self.timestamps = self.timestamps[slide_amount:]
            
        return chunks
