import torch
import torch.nn as nn


class Trainer:
    def __init__(self, pipe, model, label_smoothing, lr, weight_decay, grad_clip_value, grad_max_norm):
        self._pipe = pipe
        self._model = model
        self._device = model.device
        
        self._grad_clip_value = grad_clip_value
        self._grad_max_norm = grad_max_norm
        
        self._criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._criterion.to(self._device)
        self._optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        

    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        self._model.train()
        
        for item in self._pipe:
            item['image'] = images = item['image'].to(self._device, non_blocking=True)
            item['label'] = labels = item['label'].to(self._device, non_blocking=True)
            
            self._optimizer.zero_grad()
            outputs = self._model(images)
            loss = self._criterion(outputs, labels)
            loss.backward()
                
            if self._grad_clip_value is not None:
                nn.utils.clip_grad_value_(self._model.parameters(), clip_value=self._grad_clip_value)
            if self._grad_max_norm is not None:
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=self._grad_max_norm, norm_type=2)
            
            self._optimizer.step()
            
            item['output'] = outputs
            item['loss'] = loss.item()
            
            yield item


def trainer(pipe, model, *, label_smoothing, lr, weight_decay, grad_clip_value, grad_max_norm):
    pipe = Trainer(pipe, model, label_smoothing, lr, weight_decay, grad_clip_value, grad_max_norm)
    return pipe

