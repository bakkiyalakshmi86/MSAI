import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm

def generate_gradcam(model, input_tensor, original_image, target_layer_name=None):
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    if target_layer_name is None:
        target_layer = model.conv_block3  # Use the final conv block
    else:
        target_layer = dict(model.named_modules())[target_layer_name]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    class_idx = int((output > 0.5).item())
    output[0][class_idx].backward()

    grads_val = gradients[0][0].detach().numpy()
    fmap = activations[0][0].detach().numpy()

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam -= np.min(cam)
    cam /= (np.max(cam) + 1e-8)

    cam = Image.fromarray((cam * 255).astype(np.uint8)).resize(original_image.size, Image.BILINEAR)
    cam = np.array(cam)

    heatmap = cm.jet(cam / 255.0)[:, :, :3]  # Drop alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap)

    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    blended = Image.blend(original_image, heatmap, alpha=0.5)

    forward_handle.remove()
    backward_handle.remove()

    return blended