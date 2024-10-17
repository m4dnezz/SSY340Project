from torch.utils.data import Dataset
from itertools import chain
from PIL import Image
from path import Path
from matplotlib import pyplot as plt
import numpy as np


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
        self._angry_label = 0
        self._disgust_label = 1
        self._fear_label = 2
        self._happy_label = 3
        self._neutral_label = 4
        self._sad_label = 5
        self._surprise_label = 6

        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        self._samples = self._collect_samples()

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

    def _collect_samples(self):
        """Collect all paths and labels

        Helper method for the constructor
        """
        # Iterator over dog filpath
        angry_paths = self._collect_imgs_sub_dir(self.root / "angry")
        angry_paths_and_labels = map(lambda path: (path, self._angry_label), angry_paths)

        disgust_paths = self._collect_imgs_sub_dir(self.root / "disgust")
        disgust_paths_and_labels = map(lambda path: (path, self._disgust_label), disgust_paths)

        fear_paths = self._collect_imgs_sub_dir(self.root / "fear")
        fear_paths_and_labels = map(lambda path: (path, self._fear_label), fear_paths)

        happy_paths = self._collect_imgs_sub_dir(self.root / "happy")
        happy_paths_and_labels = map(lambda path: (path, self._happy_label), happy_paths)

        neutral_paths = self._collect_imgs_sub_dir(self.root / "neutral")
        neutral_paths_and_labels = map(lambda path: (path, self._neutral_label), neutral_paths)

        sad_paths = self._collect_imgs_sub_dir(self.root / "sad")
        sad_paths_and_labels = map(lambda path: (path, self._sad_label), sad_paths)

        surprise_paths = self._collect_imgs_sub_dir(self.root / "surprise")
        surprise_paths_and_labels = map(lambda path: (path, self._surprise_label), surprise_paths)

        return sorted(
            list(chain(angry_paths_and_labels, disgust_paths_and_labels, fear_paths_and_labels,
                       happy_paths_and_labels, neutral_paths_and_labels, sad_paths_and_labels, surprise_paths_and_labels)),
            key=lambda x: x[0].stem,
        )

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

    def show_image(self, index):
        img, label = self.__getitem__(index)
        plt.imshow(np.squeeze(img), cmap='grey')
        plt.title(f"index: {index}, label: {label}")
        plt.show()
