import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image
from sklearn.decomposition import PCA
import coremltools as ct
import time
import pickle
import torch
import torchvision.transforms.functional as TF
from scipy import signal

foreground_segmentation_model = pickle.load(open('notebooks/foreground_segmentation_model.pkl', 'rb'))
# model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', source='github', weights="https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoieGJ3dzM4M2o4djlnbmQ0d3NhNHI5c3dkIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTUzNzA0Mjl9fX1dfQ__&Signature=F%7E7LBWyNTy0c4nSnOIjIaGuiUd-AHIgIQzmfwivW28ducAs3kLadd2wj2taoY4K588BkPsp4OItjzYXImnLUGYnPGhzISTR%7EgDgyRhF9yukwsC8uCumExqnzM8sB9KBP0Lracpo3BdCRe9wWgucX21we09jH%7EBuCleOoDVeLZeve9%7EUasCwr-miQhu0vmHirfWdUZ4A%7E%7EHrJ1KU-IxigsCwAq6pfE66tCRIF9wtZaw62RbZi29Frr3AEDrCu3aDJzV6OPemu%7EOX0L-BPliKIh-HZhUI38sDMmvJ2bKKOGArKVuZKuw60rJx0N7FAmiKRjJMfAd-hqK8Qf3vzdj%7EsIQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082201730553524")
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_convnext_tiny', source='github', weights="https://dinov3.llamameta.net/dinov3_convnext_tiny/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoieGJ3dzM4M2o4djlnbmQ0d3NhNHI5c3dkIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTUzNzA0Mjl9fX1dfQ__&Signature=F%7E7LBWyNTy0c4nSnOIjIaGuiUd-AHIgIQzmfwivW28ducAs3kLadd2wj2taoY4K588BkPsp4OItjzYXImnLUGYnPGhzISTR%7EgDgyRhF9yukwsC8uCumExqnzM8sB9KBP0Lracpo3BdCRe9wWgucX21we09jH%7EBuCleOoDVeLZeve9%7EUasCwr-miQhu0vmHirfWdUZ4A%7E%7EHrJ1KU-IxigsCwAq6pfE66tCRIF9wtZaw62RbZi29Frr3AEDrCu3aDJzV6OPemu%7EOX0L-BPliKIh-HZhUI38sDMmvJ2bKKOGArKVuZKuw60rJx0N7FAmiKRjJMfAd-hqK8Qf3vzdj%7EsIQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082201730553524")
model.to("mps")

# model = ct.models.MLModel("dinov3_vits16_768_1360_fp16.mlpackage", compute_units=ct.ComputeUnit.ALL)

patch_size = 16

def resize_transform(
    mask_image: Image,
    image_size: int,
    patch_size: int = patch_size
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

n_layers = 12

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (512, 512))
    pil_frame = Image.fromarray(frame)

    frame_resized = resize_transform(pil_frame, image_size=512)
    frame_normalized = TF.normalize(frame_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    frame_normalized = frame_normalized.unsqueeze(0).to("mps")
    print(frame_normalized.shape)

    start_time = time.time()
    with torch.inference_mode():
        # with torch.autocast(device_type='mps', dtype=torch.float16):
        feats = model.get_intermediate_layers(frame_normalized, n=range(n_layers), reshape=True, norm=True)
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

    # x = model.predict({"image": frame_normalized})["var_2187"]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
        
    h_patches, w_patches = [int(d / patch_size) for d in frame_resized.shape[1:]]

    fg_score = foreground_segmentation_model.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
    fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

    foreground_selection = fg_score_mf.view(-1) > 0.5
    fg_patches = x[foreground_selection]


    pca = PCA(n_components=3, whiten=True)
    pca.fit(fg_patches)

    projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)

    # multiply by 2.0 and pass through a sigmoid to get vibrant colors 
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

    # mask the background using the fg_score_mf
    projected_image *= (fg_score_mf.unsqueeze(0) > 0.5)
    fg_score_mf = fg_score_mf.numpy()

    projected_image = projected_image.permute(1, 2, 0).numpy()

    cv2.imshow('frame', frame)
    int_fg_score = (fg_score_mf > 0.5).astype(np.uint8) * 255
    int_fg_score = cv2.resize(int_fg_score, (frame.shape[1], frame.shape[0]))
    cv2.imshow('fg_score', int_fg_score)

    projected_image = cv2.resize(projected_image, (frame.shape[1], frame.shape[0]))

    cv2.imshow('projected_image', projected_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()