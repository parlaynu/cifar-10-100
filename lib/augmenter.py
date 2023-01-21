import torch
import torchvision.transforms as TF


class Augmenter:
    def __init__(self, pipe, mode, mean, std, size):
        self._pipe = pipe

        valid_modes = ["train", "val"]
        if mode not in valid_modes:
            raise ValueError(f"invalid augmentation mode: {mode} is not in {valid_modes}")

        if mode == "val":
            self._transforms = TF.Compose([
                TF.ToTensor(),
                TF.Resize(size),
                TF.Normalize(mean, std),
            ])
        elif mode == "train":
            self._transforms = TF.Compose([
                TF.ToTensor(),
                TF.RandomHorizontalFlip(p=0.5),
                TF.RandomGrayscale(p=0.1),
                TF.Resize(size),
                TF.Normalize(mean, std),
                TF.RandomErasing(p=0.5)
            ])

    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        for item in self._pipe:
            item['image'] = self._transforms(item['image'])
            yield item


def augmenter(pipe, *, mode, mean, std, size):
    pipe = Augmenter(pipe, mode, mean, std, size)
    return pipe

