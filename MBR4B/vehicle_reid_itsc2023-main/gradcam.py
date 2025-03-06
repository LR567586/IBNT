import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from models.models import MBR_model
from PIL import Image

def visualize_feature_heatmap(feature_map, original_image, title='Feature Map Heatmap'):
    """ Function to visualize the aggregated feature map as heatmap. """
    feature_map = feature_map.detach().cpu().numpy()
    heatmap = np.max(feature_map, axis=1).squeeze()

    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # 'jet' colormap, alpha for transparency
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.title(title, size=20)
    plt.show()


def main():
    # Load the trained model
    model = MBR_model(575, ["R50", "BoT"], n_groups=0, losses="Classical", LAI=False)
    model.load_state_dict(torch.load(r'D:\MBR4B\logs\Veri776\Hybrid_2B\7\best_mAP.pt'))
    model.eval()

    # Prepare input tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_image = Image.open(r"D:\MBR4B\vehicle_reid_itsc2023-main\dataset\VeRi\image_query\0002_c002_00030600_0.jpg")  # Provide the path to your image
    input_tensor = transform(input_image).unsqueeze(0)
    cam = torch.tensor([0])
    view = torch.tensor([0])

    # Forward pass to get feature maps
    with torch.no_grad():
        preds, embs, ffs, output = model(input_tensor, cam, view)

    # Assume the last element of output is the feature map to be visualized
    feature_map = output[-1]

    # Upsample feature map to match input image size
    upsampled_feature_map = torch.nn.functional.interpolate(feature_map, size=(256, 256), mode='bilinear', align_corners=False)

    # Reformat original image for visualization
    original_image = np.array(input_image.resize((256, 256))) / 255.

    # Visualize the feature map
    visualize_feature_heatmap(upsampled_feature_map, original_image, title='Feature Map of Last Layer')


if __name__ == "__main__":
    main()