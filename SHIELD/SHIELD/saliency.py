import torch
from typing import Union
from captum.attr import Saliency

def saliencyguided(
    model: torch.nn.Module,
    input: torch.Tensor,
    input_0: torch.Tensor = None,
    segmentation: int = 1,
    device: Union[torch.device, str] = torch.device("cpu"),
    percentage=None,
):
    output = model(input).detach()
    target = output.argmax(-1)
    saliency = Saliency(model)
    grads = saliency.attribute(input,target,abs=False).mean(1).detach().cpu().to(dtype=torch.float)
    # Dimensiones de grads: [batch_size, 3, 224, 224]
    # Poner una máscara en input donde los valores de grads sean mayores a la mediana de los valores de grads

    mask = torch.zeros_like(input)
    # extender grads para que tenga las mismas dimensiones que input
    grads = grads.unsqueeze(1).expand_as(input).abs()

    median_for_example = (
        grads.view(grads.size(0), -1).quantile(0.01*percentage,- 1)[0].view(-1, 1, 1, 1)
    )
    median_for_example = median_for_example.expand_as(grads)

    mask[grads > median_for_example] = 1
    masked_input = input * mask

    masked_output = model(masked_input)

    loss_KLDiv = torch.nn.KLDivLoss(reduction="batchmean",log_target=True)(masked_output, output)# + torch.nn.KLDivLoss(reduction="batchmean")(output, masked_output)

    return loss_KLDiv


def saliencymixup(
    model: torch.nn.Module,
    input: torch.Tensor,
    input_0: torch.Tensor = None,
    segmentation: int = 1,
    device: Union[torch.device, str] = torch.device("cpu"),
    percentage=None,
):

    return torch.tensor(0.0)
