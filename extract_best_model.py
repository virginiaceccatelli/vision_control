import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",  
    in_channels=3,
    classes=1,
    activation=None
)

# load trained weights
model.load_state_dict(torch.load("checkpoints/unet_epoch46_checkpoint7.pt"))
model.eval()

# convert to TorchScript
input = torch.randn(1, 3, 320, 320)
traced_script_module = torch.jit.trace(model, input)

# save TorchScript model
traced_script_module.save("unet_ground_plane.pt")

