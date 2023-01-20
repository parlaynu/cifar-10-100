import torch
import torch.nn.functional as F


class Assessor:
    def __init__(self, pipe):
        self._pipe = pipe
    
    def __len__(self):
        return len(self._pipe)

    def __iter__(self):
        for item in self._pipe:
            labels = item['label']
            outputs = item['output']
            
            probs = F.softmax(outputs, dim=1)
            pred_probs, pred_labels = torch.max(probs, dim=1)
            
            # check for one-hot encoded labels
            if labels.shape == outputs.shape:
                # REVISIT: hacky, but does produce a curve that trends
                #   similar to an accuracy curve. It's indicative of progress...
                correct = torch.sum(labels * probs, dim=1)
            else:
                correct = torch.eq(pred_labels, labels).long()
            
            item['pred-label'] = pred_labels
            item['pred-prob'] = pred_probs
            item['correct'] = correct
            
            yield item


def assessor(pipe):
    pipe = Assessor(pipe)
    return pipe

