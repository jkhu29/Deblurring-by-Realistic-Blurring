import torch
import torch.nn as nn


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(GradientPenaltyLoss, self).__init__()
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, inputs):
        if self.grad_outputs.size() != inputs.size():
            self.grad_outputs.resize_(inputs.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss