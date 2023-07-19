import torch


class Validator:
    def __init__(self, pipe, model):
        self._pipe = pipe
        self._model = model
        self._device = model.device
        
        self._criterion = torch.nn.CrossEntropyLoss()
        self._criterion.to(self._device)

    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        self._model.eval()
        
        for item in self._pipe:
            item['image'] = images = item['image'].to(self._device, non_blocking=True)
            item['label'] = labels = item['label'].to(self._device, non_blocking=True)

            with torch.no_grad():
                outputs = self._model(images)
                loss = self._criterion(outputs, labels)
            
            item['output'] = outputs
            item['loss'] = loss.item()
            
            yield item


def validator(pipe, model):
    pipe = Validator(pipe, model)
    return pipe

