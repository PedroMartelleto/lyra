from typing import List, Optional
from pathlib import Path
from src.models.data.spatialvid import SpatialVid

class SpatialVidWrapper:
    """
    INSTRUMENTED VERSION: A dedicated wrapper for the SpatialVid dataset.
    """
    def __init__(self, **kwargs):
        # print("[Wrapper: __init__] Initializing SpatialVidWrapper.")
        kwargs.pop('cls', None)
        kwargs.pop('is_static', None)
        kwargs.pop('is_multi_view', None)
        self.dataset = SpatialVid(**kwargs)
        root_path = Path(self.dataset.root_path)
        self.mp4_file_paths = [root_path / row['video path'] for _, row in self.dataset.metadata.iterrows()]
        self.sample_list = list(range(len(self.dataset)))
        self.num_cameras = 1
        self.is_static = True
    
    def __len__(self): return len(self.dataset)
    def count_frames(self, idx): return self.dataset.num_frames(idx)
    def count_cameras(self, idx): return self.dataset.num_views(idx)
    
    def get_data(self, idx, data_fields, frame_indices=None, view_indices=None):
        # print(f"[Wrapper: get_data] STARTING for index: {idx}")
        # print(f"[Wrapper: get_data] Received frame_indices (len={len(frame_indices)})")
        # print(f"[Wrapper: get_data] Received view_indices (len={len(view_indices)})")

        # The wrapper's job is to ensure indices match before passing them down.
        # This logic is crucial. Let's verify it.
        num_frames = len(frame_indices)
        if len(view_indices) != num_frames:
            #  print(f"[Wrapper: get_data] MISMATCH DETECTED! Aligning view indices to frame indices.")
            final_view_indices = [view_indices[0]] * num_frames if view_indices else [0] * num_frames
        else:
            final_view_indices = view_indices
        
        # print(f"[Wrapper: get_data] Calling self.dataset.read with frame_indices (len={len(frame_indices)}) and final_view_indices (len={len(final_view_indices)})")
        result = self.dataset.read(idx, frame_indices, final_view_indices, data_fields)
        # print(f"[Wrapper: get_data] self.dataset.read FINISHED.")
        return result