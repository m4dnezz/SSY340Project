from torch.utils.data import Dataset, DataLoader
from itertools import chain
from PIL import Image
from torch.utils.data import Dataset
from path import Path
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

class FER2013(Dataset):
    def __init__(self, root, transform):
        """Constructor

        Args:
            root (Path/str): Filepath to the data root, e.g. './small_train'
            transform (Compose): A composition of image transforms, see below.
        """

        root = Path(root)
        if not (root.exists() and root.is_dir()):
            raise ValueError(f"Data root '{root}' is invalid")

        self.root = root
        self.transform = transform
        self._angry = 0
        self._disgust = 1
        self._fear = 2
        self._happy = 3
        self._neutral = 4
        self._sad = 5
        self._surpries = 6

        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        #self._samples = self._collect_samples()

    def __getitem__(self, index):
        """Get sample by index

        Args:
            index (int)

        Returns:
             The index'th sample (Tensor, int)
        """
        # Access the stored path and label for the correct index
        path, label = self._samples[index]
        # Load the image into memory
        img = Image.open(path)
        # Perform transforms, if any.
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        """Total number of samples"""
        return len(self._samples)

    # def _collect_samples(self):
    #     """Collect all paths and labels

    #     Helper method for the constructor
    #     """
    #     # Iterator over dog filpath
    #     dog_paths = self._collect_imgs_sub_dir(self.root / "dogs")
    #     # Iterator of pairs (path, dog label)
    #     # Again, we use the `map` function to create an iterator. It's use is not as common as the so called
    #     # 'list comprehension' you've previously seen, but a good alternative to have seen.
    #     dog_paths_and_labels = map(lambda path: (path, self._dog_label), dog_paths)
    #     # Same for cats
    #     cat_paths = self._collect_imgs_sub_dir(self.root / "cats")
    #     cat_paths_and_labels = map(lambda path: (path, self._cat_label), cat_paths)
    #     # Sorting is not strictly necessary, but filesystem globbing (wildcard search) is not deterministic,
    #     # and consistency is nice when debugging.
    #     return sorted(
    #         list(chain(dog_paths_and_labels, cat_paths_and_labels)),
    #         key=lambda x: x[0].stem,
    #     )

    @staticmethod
    def _collect_imgs_sub_dir(sub_dir: Path):
        """Collect image paths in a directory

        Helper method for the constructor
        """
        if not sub_dir.exists():
            raise ValueError(
                f"Directory '{sub_dir}' does not exist. Are you sure you have the correct path?"
            )
        return sub_dir.glob("*.jpg")

    def get_sample_by_id(self, id_):
        """Get sample by image id

        Convenience method for exploration.
        The indices does not correspond to the image id's in the filenames.
        Here is a (rather inefficient) way of inspecting a specific image.

        Args:
            id_ (str): Image id, e.g. `dog.321`
        """
        id_index = [path.stem for (path, _) in self._samples].index(id_)
        return self[id_index]