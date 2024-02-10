"""
This example shows how to use the `octo.data` dataloader with PyTorch by wrapping it in a simple PyTorch
dataloader. The config below also happens to be our exact pretraining config (except for the batch size and
shuffle buffer size, which are reduced for demonstration purposes).
"""
import numpy as np
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import tqdm
from einops import rearrange

# DATA_PATH = "gs://rail-octo-central2/resize_256_256"
DATA_PATH = "gs://gresearch/robotics"

tf.config.set_visible_devices([], "GPU")


class TorchRLDSImageDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        transform,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self.transform = transform
        self.size = self.get_len()

        self.cnt = 0
        # for sample in self._rlds_dataset.as_numpy_iterator():
        #     print("##### ", sample['observation']['image_primary'].shape)
        #     import pdb; pdb.set_trace()
        self.iterator = self._rlds_dataset.as_numpy_iterator()
    
    def reset_state(self):
        print("### RESET State")
        self.cnt = 0
        self.iterator = self._rlds_dataset.as_numpy_iterator()
       
    def __iter__(self):
        for sample in self.iterator:
            if self.cnt < self.size:
                processed_sample = sample['observation']['image_primary'].squeeze(0)
                # processed_sample = rearrange(processed_sample, 'h w c -> c h w')
                processed_sample = self.transform(processed_sample)
                self.cnt += 1
                # print(f"yield item, self.cnt: {self.cnt}  self.size: {self.size}")
                yield processed_sample, sample['action']# only use primary camera image for now, after squeeze: [b, h]
            else:
                break

    def get_len(self):
        lengths = np.array(
            [
                stats["num_transitions"].astype('float64')
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"): # sample_weights addup to 1, lengths might not be an integer
            lengths *= np.array(self._rlds_dataset.sample_weights)
        lengths = np.floor(lengths).astype('int64')
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)


    def __len__(self):
        return self.size
        # lengths = np.array(
        #     [
        #         stats["num_transitions"].astype('float64')
        #         for stats in self._rlds_dataset.dataset_statistics
        #     ]
        # )
        # if hasattr(self._rlds_dataset, "sample_weights"): # sample_weights addup to 1, lengths might not be an integer
        #     lengths *= np.array(self._rlds_dataset.sample_weights)
        # lengths = np.floor(lengths).astype('int64')
        # total_len = lengths.sum()
        # if self._is_train:

        #     return int(0.95 * total_len)
        # else:
        #     return int(0.05 * total_len)
        
def make_dset(transform):
    print("load args...")
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        "oxe_magic_soup",
        DATA_PATH,
        load_camera_views=("primary", "wrist"),
    )
    print("interleaving...")
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=True,
        shuffle_buffer_size=1,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
        batch_size=None,  # batching will be handles in PyTorch Dataloader object
        balance_weights=True,
        traj_transform_kwargs=dict(
            goal_relabeling_strategy="uniform",
            window_size=1,
            future_action_window_size=3,
            subsample_length=100,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
                "wrist": dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            },
            resize_size=dict(
                primary=(256, 256),
                wrist=(128, 128),
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,
        traj_read_threads=48,
    )


    pytorch_dataset = TorchRLDSImageDataset(dataset, transform=transform)
    return pytorch_dataset

# dataloader = DataLoader(
#     pytorch_dataset,
#     batch_size=16,
#     num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
# )

# for i, sample in tqdm.tqdm(enumerate(dataloader)):
#     import pdb; pdb.set_trace()
#     if i == 5000:
#         break

