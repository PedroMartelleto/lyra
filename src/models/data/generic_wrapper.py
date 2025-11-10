from typing import List, Optional

class GenericWrapper:
    """
    A generic wrapper to adapt different BaseDataset implementations (like Radym or SpatialVid)
    to the interface expected by the Provider and the rest of the training pipeline.

    It instantiates the underlying dataset class provided in the registry
    and forwards calls to it.
    """
    def __init__(self, is_static: bool = True, is_multi_view: bool = False, **kwargs):
        """
        Initializes the wrapper and the underlying dataset.

        Args:
            is_static (bool): Flag indicating if the dataset contains static scenes.
            is_multi_view (bool): Flag indicating if the dataset has multiple camera views per scene.
            **kwargs: Must contain a 'cls' key with the dataset class to instantiate,
                      and the rest of the kwargs are passed to that class's constructor.
        """
        if 'cls' not in kwargs:
            raise ValueError("The 'cls' key specifying the dataset class is missing from kwargs.")
            
        dataset_cls = kwargs.pop('cls')
        self.dataset = dataset_cls(**kwargs)

        self.is_static = is_static
        # The sample list is simply the range of indices for the dataset.
        self.sample_list = list(range(len(self.dataset)))

        # Determine the number of cameras/views.
        if is_multi_view:
            if hasattr(self.dataset, 'num_cameras'):
                self.num_cameras = self.dataset.num_cameras
            elif hasattr(self.dataset, 'n_views') and self.dataset.n_views != -1:
                self.num_cameras = self.dataset.n_views
            else:
                # Fallback: query the first video for its number of views.
                self.num_cameras = self.dataset.num_views(0) if len(self.dataset) > 0 else 1
        else:
            self.num_cameras = 1

    def __len__(self):
        return len(self.dataset)
    
    def count_frames(self, video_idx: int):
        """Returns the number of frames for a given video index."""
        return self.dataset.num_frames(video_idx)
    
    def count_cameras(self, video_idx: int):
        """Returns the number of views/cameras for a given video index."""
        return self.dataset.num_views(video_idx)
    
    def get_data(
        self,
        idx: int,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: Optional[List[int]] = None,
    ):
        """
        Reads and returns data by forwarding the call to the wrapped dataset instance.
        """
        return self.dataset.read(
            video_idx=idx,
            frame_idxs=frame_indices,
            view_idxs=view_indices,
            data_fields=data_fields
        )