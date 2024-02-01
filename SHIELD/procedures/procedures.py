import torch
import torchvision
import tqdm
from torch.utils.checkpoint import checkpoint


def classifier(pretrained_model, num_classes):
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
        batch_input, batch_labels = batch_input.to(device).float(), batch_labels.to(device)
        if transform != None and train == True:
            batch_input = transform(batch_input)
        batch_input.requires_grad = True
        batch_input, batch_labels = batch_input.to(device).float(), batch_labels.to(device        )

        if train == True:
            optimizer.zero_grad()
        reg = reg_f(model, batch_input)
        # output = model(batch_input)
        output = checkpoint(model, batch_input)#,use_reentrant=True)

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
            REGS += reg

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
    return train_step(
        ds_loader=ds_loader,
        model=model,
        optimizer=None,
        loss_f=loss_f,
        reg_f=reg_f,
        device=device,
        train=False,
    )
