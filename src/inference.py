import os 
import torch
import segmentation_models_pytorch as smp
from typing import Union, Sequence

from src.utils.transforms import basic_transform
import rasterio
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise AssertionError("CUDA not available")

# transform = basic_transform(resize_size=512,triple=True)

def predict_smp(smp_model_path:str, tile_path_list:Union[str, Sequence[str]],transform, out_pred_path:str):
    """
    smpp_model_path: path to SMP model .pth file,
    tile_path_list: path to tile or tiles folder,
    transform: torchvision transform as used in training,
    out_pred_path: folder path
    """
    model = smp.from_pretrained(smp_model_path)
    if os.path.isfile(tile_path_list) and tile_path_list.endswith('.tif'): # If tile_path_list is single image
        tile_path_list = [tile_path_list]
    batch = []
    for img_path in tile_path_list:
        with rasterio.open(img_path) as src:
            data = src.read() # read all bands
            image = transform(data) # as tensor
        # image = image.unsqueeze(0).to(device) # add batch dim
        batch.append(image)

    batch_torch = torch.stack(batch, dim=0).to(device)


    # predict 
    model.eval()
    model.to(device)
    with torch.no_grad():
        output = model(batch_torch)
        pred = torch.sigmoid(output)
        pred_masks = pred.argmax(dim=1).cpu().numpy()

        ## SAVE PRED MASKS IN OUT PRED PATH
    c=0
    ll = len(tile_path_list)
    for pred_mask, img_path in zip(pred_masks, tile_path_list):
        img_name = os.path.basename(img_path)
        out_path = os.path.join(out_pred_path, img_name)

        with rasterio.open(img_path) as src:
            profile = src.profile.copy()

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pred_mask, 1)

        c+=1
        print(f'Pred {c}/{ll}: Saved prediction at {out_path}')





