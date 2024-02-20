import torch
import torchvision
import tqdm
from torch.utils.checkpoint import checkpoint


def classifier(pretrained_model, num_classes):
    '''
    Model used for classification on SHIELD paper.
    
    This function returns a model with the specified number of classes and the specified pretrained model of
    the torchvision library.
    
    This method was used as a wrapper to load just the classifier of the SHIELD paper but
    you can use your own model if it accepts the same input format as the torchvision
    models and returns a tensor with the same shape as the number of classes in the logits space.
    
    :param pretrained_model: The name of the pretrained model to be used. It can be one of the following:
        - efficientnet-b2
        - efficientnet_v2_s
        - vit_b_16
        - swin_v2_s
    :param num_classes: The number of classes of the dataset.
    :return: The model with the specified number of classes and the specified pretrained model. We
        delete the last layer of the model and add a new layer with the specified number of classes.
        
    To load a specific model, use the following code:

    .. code-block:: python
    
        model = classifier("efficientnet-b2", 10)
    '''
    
    if "efficientnet-b2" == pretrained_model:
        model_pretrained = torchvision.models.efficientnet_b2(weights="IMAGENET1K_V1")
        pretrained_state_dict = model_pretrained.state_dict()

        model = torchvision.models.efficientnet_b2(num_classes=num_classes)
        state_dict = model.state_dict()

        pretrained_state_dict["classifier.1.weight"] = state_dict["classifier.1.weight"]
        pretrained_state_dict["classifier.1.bias"] = state_dict["classifier.1.bias"]
        model.load_state_dict(pretrained_state_dict)
    elif "efficientnet_v2_s" == pretrained_model:
        model_pretrained = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        pretrained_state_dict = model_pretrained.state_dict()

        model = torchvision.models.efficientnet_v2_s(num_classes=num_classes)
        state_dict = model.state_dict()

        pretrained_state_dict["classifier.1.weight"] = state_dict["classifier.1.weight"]
        pretrained_state_dict["classifier.1.bias"] = state_dict["classifier.1.bias"]
        model.load_state_dict(pretrained_state_dict)
    elif "vit_b_16" == pretrained_model:
        model_pretrained = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        pretrained_state_dict = model_pretrained.state_dict()

        model = torchvision.models.vit_b_16(num_classes=num_classes)
        state_dict = model.state_dict()

        pretrained_state_dict["heads.head.weight"] = state_dict["heads.head.weight"]
        pretrained_state_dict["heads.head.bias"] = state_dict["heads.head.bias"]
        model.load_state_dict(pretrained_state_dict)

    elif "swin_v2_s" == pretrained_model:
        model_pretrained = torchvision.models.swin_v2_s(weights="IMAGENET1K_V1")
        pretrained_state_dict = model_pretrained.state_dict()
        model = torchvision.models.swin_v2_s(num_classes=num_classes)
        state_dict = model.state_dict()

        pretrained_state_dict["head.weight"] = state_dict["head.weight"]
        pretrained_state_dict["head.bias"] = state_dict["head.bias"]
        model.load_state_dict(pretrained_state_dict)
    return model


def train_step(
    ds_loader, model, optimizer, loss_f, reg_f, device, transform=None, train=True
):
    '''
    Train step for the model. This function proceed to do a full epoch of training or validation, detending
    on the `train` parameter. It returns the loss, accuracy and regularization of the model.
    
    :param ds_loader: The dataloader to be used for training or validation.
    :param model: The model to be trained or validated.
    :param optimizer: The optimizer to be used for training. If `train` is False, this parameter is not used.
    :param loss_f: The loss function to be used for training or validation.
    :param reg_f: The regularization function to be used for training or validation. It can be None for no regularization.
    :param device: The device to be used for training or validation.
    :param transform: The transformation to be used for training. If `train` is False, this parameter is not used. It 
        can be None for no data augmentation or transformation needed.
    :param train: A boolean indicating if the model should be trained or validated. 
        - If `train` is True, the model is trained and updated with the optimizer.
        - If `train` is False, the model is validated and the optimizer is not used.
    :return: The loss, accuracy and regularization of the model.
    
    Example of a full step of training and validation using this function:
    
    .. code-block:: python
    
        # We have a model, a train dataloader, a validation dataloader, a loss function and a regularization function. 
        # The previous step had `best_val_loss` as a variable to store the best validation loss over the epochs.
        
        # The model weights are updated with the optimizer on the training phase. The gradient is calculated with the 
        # loss and the regularization function.
        loss, acc, reg = train_step(train_dataloader, model, optimizer, loss_f, reg_f, device, transform, train=True)
        
        # The model is not updated on the validation phase. The best model is saved if the validation `loss+reg` is 
        # better than the previous one.
        val_loss, val_acc, val_reg = train_step(validation_dataloader, model, optimizer, loss_f, reg_f, device, transform, train=False)
        if loss+reg < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
    '''
    ACC, LOSS, REGS = 0.0, 0.0, 0.0

    ds_loader = tqdm.tqdm(ds_loader, desc="Training" if train else "Validation")
    total = 0
    if train == True:
        model.train()
    else:
        model.eval()
    torch.cuda.empty_cache()
    model = model.to(device)
    for data in ds_loader:
        batch_input, batch_labels = data
        total += len(batch_input)
        batch_input, batch_labels = batch_input.float().to(device), batch_labels.to(device)
        if transform != None and train == True:
            batch_input = transform(batch_input)
        batch_input.requires_grad = True
        batch_input, batch_labels = batch_input.float().to(device), batch_labels.to(device)

        if train == True:
            optimizer.zero_grad()
        reg = reg_f(model, batch_input)
        # output = model(batch_input)
        output = checkpoint(model, batch_input,use_reentrant=True)

        loss = loss_f(output, batch_labels)
        LOSS += loss.item()

        loss += reg if reg != None else 0
        if train == True:
            loss.backward()
            optimizer.step()

        ACC += torch.sum(
            (torch.argmax(output, dim=-1) == torch.argmax(batch_labels, dim=-1)).float()
        ).item()
        if isinstance(reg, torch.Tensor):
            REGS += reg.item()
        else:
            REGS +=  reg if reg != None else 0

        ds_loader.set_postfix(
            {
                "loss": LOSS / total,
                "acc": ACC / total,
                "reg": REGS / total,
            }
        )

    return (
        LOSS / total,
        ACC / total,
        REGS / total,
    )


def validation_step(ds_loader, model, loss_f, reg_f, device):
    '''
    Validation step for the model. 
    
    This function is a wrapper for the `train_step` function with the `train` parameter set to False.
    
    :param ds_loader: The dataloader to be used for validation.
    :param model: The model to be validated.
    :param loss_f: The loss function to be used for validation.
    :param reg_f: The regularization function to be used for validation. It can be None for no regularization.
    :param device: The device to be used for validation.
    :return: The loss, accuracy and regularization of the model.
    
    '''
    
    return train_step(
        ds_loader=ds_loader,
        model=model,
        optimizer=None,
        loss_f=loss_f,
        reg_f=reg_f,
        device=device,
        train=False,
    )
