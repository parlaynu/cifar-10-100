import random
import numpy as np
import torch
import torchvision.transforms as TF


class Mixup:
    def __init__(self, pipe, num_classes, ratio, p):
        self._pipe = pipe
        self._num_classes = num_classes
        self._ratio = ratio
        self._p = p

    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        previous = None
        for item in self._pipe:
            # if we're switched off, do nothing
            if self._p == 0.0:
                yield item
                continue
            
            # one-hot encode the labels when using mixup
            label = torch.tensor(item['label'])
            label = item['label'] = torch.nn.functional.one_hot(label, num_classes=self._num_classes).float()
            image = item['image']
            
            # check probability for mixup... and make sure there's an image to mixup with...
            if previous is None or random.random() > self._p:
                previous = (label, image)
                yield item
                continue
            
            label = self._ratio * label + (1.0 - self._ratio) * previous[0]
            image = (self._ratio * image + (1.0 - self._ratio) * previous[1]).astype(np.uint8)
            
            item['label'] = label
            item['image'] = image
            
            previous = None
            
            yield item


def mixup(pipe, *, num_classes, ratio, p=0.2):
    pipe = Mixup(pipe, num_classes, ratio, p)
    return pipe

