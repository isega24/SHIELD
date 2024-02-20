import torch
from typing import Union
from torch.utils.checkpoint import checkpoint


def derivative(model: torch.nn.Module, input: torch.Tensor, device: torch.device):
    input.requires_grad = True
    # x dim: (batch_size, channels, height, width)

    output = model(input)
    # y dim: (batch_size, num_classes)

    # dy/dx dim: (batch_size, num_classes, channels, height, width)
    grad = torch.Tensor(
        output.shape[0], output.shape[1], input.shape[1], input.shape[2], input.shape[3]
    ).to(device)

    # grad dim: (batch_size, num_classes, channels, height, width)

    for i in range(output.shape[1]):
        grad[:, i] = torch.autograd.grad(output[:, i].sum(), input, retain_graph=True)[
            0
        ]

    return grad


def importance_matrix(
    model: torch.nn.Module,
    input: torch.Tensor,
    device: torch.device = torch.device("cpu"),
):
    dy_dx = derivative(model, input, device)
    y = checkpoint(model,input)
    S = torch.softmax(y, dim=1)
    # Dims S: (batch_size, num_classes)
    # Expand to (batch_size,1, num_classes)
    S = S.unsqueeze(1)
    # Repeat S for each num_classes over the unsqueezed dimension
    S = S.repeat(1, S.shape[2], 1)
    # repeat over the channels dimension

    softmax_grad = (
        (torch.eye(S.shape[2]).unsqueeze(0)).repeat(S.shape[0], 1, 1) * S
    ) * torch.transpose(S, 1, 2)

    # Matrix multiplication of dy/dx and softmax_grad
    importance = torch.matmul(dy_dx, softmax_grad)

    return importance


def feature_importance(
    model: torch.nn.Module,
    input: torch.Tensor,
    forward_input: Union[torch.Tensor, None] = None,
    device: Union[torch.device, str] = torch.device("cpu"),
):
    # Si input es una leaf node de la gráfica computacional, no se puede calcular la derivada

    # model tiene que guardar el gradiente
    output = model(input)

    S = torch.softmax(output, dim=1)
    S = S.unsqueeze(1)
    S = S.repeat(1, S.shape[2], 1)
    eye = (
        torch.eye(output.shape[1]).unsqueeze(0).repeat(output.shape[0], 1, 1).to(device)
    )

    S_t = torch.transpose(S, 1, 2)
    softmax_grad = (eye - S) * S_t

    # No quiero que softmax_grad repercuta en el gradiente
    softmax_grad = softmax_grad.detach()

    output_weighted = torch.matmul(output, softmax_grad)

    # Cuál es la diferencia entre retain_graph=True y retain_graph=False?
    feature_importance_ = torch.autograd.grad(
        output_weighted.sum(), input, retain_graph=False
    )[0]

    feature_importance_ = torch.sum(feature_importance_, dim=1)
    return feature_importance_


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


def xshield(
    model: torch.nn.Module,
    input: torch.Tensor,
    input_0: torch.Tensor = None,
    segmentation: int = 1,
    device: Union[torch.device, str] = torch.device("cpu"),
    percentage=None,
):
    """
    xAI - Selective Hidden Input Evaluation for Learning Dynamics (SHIELD).

    This function calculates the X-SHIELD score for a given model and input tensor.
    X-SHIELD is a method based on previous SHIELD method for evaluating the importance of different 
    regions in an input tensor for the predictions made by a model. It measures the sensitivity of 
    the model's output to changes in specific regions of the input.
    
    The main difference between SHIELD and X-SHIELD is that X-SHIELD uses an explanation defined by the model
    itself to evaluate the importance of different regions in the input tensor and ocludes the regions with low
    importance.

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
        shield_score = xshield(model, input, input_0=None, segmentation=1, device='cuda', percentage=2)
        print(shield_score)
    """

    if input_0 == None:
        input_0 = torch.zeros_like(input) + torch.mean(
            input, dim=(1, 2, 3), keepdim=True
        )
    model = model.to(device)
    input = input.to(device)
    input_0 = input_0.to(device)

    output_original = checkpoint(model, input, use_reentrant=True)

    feat_importance = feature_importance(model,input,forward_input=output_original,device=device)
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

    modified_output = checkpoint(model, modified_input, use_reentrant=True)

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
