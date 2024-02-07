import torch
from typing import Union
from torch.utils.checkpoint import checkpoint


def shield(
    model: torch.nn.Module,
    input: torch.Tensor,
    input_0: torch.Tensor = None,
    segmentation: int = 1,
    device: Union[torch.device, str] = torch.device("cpu"),
    percentage=None,
):
    """
    Selective Hidden Input Evaluation for Learning Dynamics (SHIELD).
    
    This function calculates the SHIELD score for a given model and input tensor.
    SHIELD is a method for evaluating the importance of different regions in an input tensor
    for the predictions made by a model. It measures the sensitivity of the model's output to changes
    in specific regions of the input.

    :param model: The model to be evaluated.
    :param input: The input tensor for which the SHIELD score is calculated.
    :param input_0: The input tensor to be considered as the `null` input with the same shape as input.
    :param segmentation: The segmentation method to be used.
    :param device: The device to be used for computation.
    :param percentage: The percentage of the input to be evaluated.
    :return: The `shield_score`. It is a scalar tensor representing the SHIELD score for the given input tensor and can
    be used as a regularization term in the training of the model given as parameter.

    Example usage:

    .. code-block:: python

        model = MyModel() \\ Your classification model
        input = torch.rand((1, 3, 224, 224))
        shield_score = shield(model, input, input_0=None, segmentation=1, device='cuda', percentage=2)
        print(shield_score)
    """

    if input_0 == None:
        input_0 = torch.zeros_like(input) + torch.mean(
            input, dim=(1, 2, 3), keepdim=True
        )
    model = model.to(device)
    input = input.to(device)
    input_0 = input_0.to(device)

    output_original = checkpoint(model, input,use_reentrant=True)

    feat_importance = (
        torch.rand((input.shape[0], input.shape[2], input.shape[3])) * 2 - 1
    )
    feat_importance = feat_importance.to(device)

    feat_importance_max = torch.nn.MaxPool2d(
        kernel_size=segmentation, stride=segmentation
    )(feat_importance)
    feat_importance_min = torch.nn.MaxPool2d(
        kernel_size=segmentation, stride=segmentation
    )(-feat_importance)

    feat_importance_final = torch.where(
        feat_importance_max > -feat_importance_min,
        feat_importance_max,
        -feat_importance_min,
    )
    feat_importance_final = feat_importance_final.unsqueeze(1)

    feat_importance_final = torch.nn.UpsamplingNearest2d(size=input.shape[2:])(
        feat_importance_final
    )
    feat_importance_final = feat_importance_final.squeeze(1)
    abs_importance = torch.abs(feat_importance_final)
    quantile_abs = torch.quantile(
        abs_importance, q=percentage / 100.0, dim=1, keepdim=True
    )
    quantile_abs = torch.quantile(
        quantile_abs, q=percentage / 100.0, dim=2, keepdim=True
    )

    quantile_abs = quantile_abs.repeat(
        1, abs_importance.shape[1], abs_importance.shape[2]
    )

    mask = torch.where(
        quantile_abs > abs_importance,
        torch.ones_like(quantile_abs),
        torch.zeros_like(quantile_abs),
    )
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1, input.shape[1], 1, 1)

    modified_input = torch.where(mask == 0, input, input_0)

    modified_output = checkpoint(model, modified_input,use_reentrant=True)

    Px_modif = torch.softmax(modified_output, dim=1)
    Px = torch.softmax(output_original, dim=1)

    constraint_1 = torch.mean(
        -torch.sum(Px * (torch.log(Px_modif) - torch.log(Px)), dim=1)
    )
    constraint_2 = torch.mean(
        -torch.sum(Px_modif * (torch.log(Px) - torch.log(Px_modif)), dim=1)
    )

    constraint = constraint_1 + constraint_2

    if torch.isnan(constraint):
        constraint = torch.tensor(0.0).to(device)

    return constraint
