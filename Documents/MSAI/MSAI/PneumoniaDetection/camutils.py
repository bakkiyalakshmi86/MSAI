import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchcam.methods import ScoreCAM
from torchvision.transforms.functional import to_pil_image
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, input_tensor, original_image, target_layer=None, blend=True):
    if target_layer is None:
                target_layer = model.conv_block5[0]  # Use the last convolutional layer

    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()

    prob = torch.sigmoid(output)
    
    # For binary classification, the output is a scalar (single probability value for class 1)
    # compute the gradient of the output w.r.t the class of 1
    class_idx = int((prob > 0.5).item())  

    output.backward(retain_graph=True)  

    grads_val = gradients[0][0].detach().cpu().numpy()
    fmap = activations[0][0].detach().cpu().numpy()

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

def explain_with_shap(model, input_tensor, image):
    background = input_tensor.clone()
    explainer = shap.GradientExplainer((model, model.conv_block5), background)
    shap_values, indexes = explainer.shap_values(input_tensor, ranked_outputs=1)

    # Fix: Ensure 2D array (H, W)
    shap_array = shap_values[0].squeeze()
    if shap_array.ndim == 3:
        shap_array = shap_array[0]

    # Normalize SHAP values to [0, 1]
    shap_array = (shap_array - shap_array.min()) / (shap_array.max() - shap_array.min() + 1e-8)

    # Apply colormap
    colormap = cm.jet(shap_array)[:, :, :3]  # Drop alpha channel
    colormap = (colormap * 255).astype(np.uint8)
    shap_heatmap = Image.fromarray(colormap).resize(image.size)

    # Ensure original image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Blend heatmap over original image
    blended = Image.blend(image, shap_heatmap, alpha=0.5)

    return blended
    
def explain_with_lime(model, original_image):
    model.eval()

    def batch_predict(images):
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  
        ])
        batch = torch.stack([transform(Image.fromarray(img).convert("L")) for img in images], dim=0)
        with torch.no_grad():
            output = model(batch)
            preds = torch.sigmoid(output) 
            #preds=model(batch)
            probs = preds.squeeze().cpu().numpy()  
        return np.vstack([1 - probs, probs]).T  # Return both class probabilities for LIME

    # Prepare image for LIME
    image_np = np.array(original_image.resize((224, 224)).convert("RGB"))
    
    # Create the LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Get explanation from LIME
    explanation = explainer.explain_instance(image_np, batch_predict, top_labels=1, hide_color=0, num_samples=1000)

    # Get the explanation mask and apply it on the image
    lime_img, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    # Mark boundaries on the image with the mask
    lime_output = Image.fromarray((mark_boundaries(lime_img, mask) * 255).astype(np.uint8))

    return lime_output

