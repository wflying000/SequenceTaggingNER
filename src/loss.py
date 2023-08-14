import torch

class SeqTaggingNerLoss():
    def __init__(self):
        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction="mean",
        )
    
    def __call__(self, inputs, targets):
        loss = self._compute(inputs, targets)
        return loss
    
    def _compute(self, inputs, targets):
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        loss = self.loss_fct(inputs, targets)
        return loss
    