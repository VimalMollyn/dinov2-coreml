import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image
from sklearn.decomposition import PCA
import coremltools as ct
import time

model = ct.models.MLModel("dinov3_vits16_518_fp16.mlpackage", compute_units=ct.ComputeUnit.ALL)
patch_size = 16

# Define image transformation
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
interpolation_mode = T.InterpolationMode.BICUBIC

transform = T.Compose([
    T.Resize(size=512, interpolation=interpolation_mode, antialias=True),
    T.CenterCrop(size=512),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

def minmax_scale(patches):
    min_val = np.min(patches, axis=0)
    max_val = np.max(patches, axis=0)
    return (patches - min_val) / (max_val - min_val)

def process_frame(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)
    
    # Apply transformation
    image_tensor = transform(image)
    
    # Crop to multiple of patch size
    height, width = image_tensor.shape[1:]  # C x H x W
    cropped_width = width - width % patch_size
    cropped_height = height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]
    
    grid_size = (cropped_height // patch_size, cropped_width // patch_size)
    image_batch = image_tensor.unsqueeze(0).numpy()
    
    patch_tokens = model.predict({"image": image_batch})["x_norm_patchtokens"] # 32*32, 1024
    # print(patch_tokens.shape)

    fg_pca = PCA(n_components=1)
    reduced_patches = fg_pca.fit_transform(patch_tokens.squeeze()).view(grid_size[0], grid_size[1], 3)
    
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)


    # # Normalize the norms to [0, 1] range
    # norm_patches = minmax_scale(reduced_patches)
    
    # Apply threshold to create mask
    # mask = (norm_patches > 0.4).ravel()
    mask = (norm_patches > 0).ravel()
    masks = [mask]

    norm_patches[np.logical_not(mask)] = 0
    norm_patches = norm_patches.reshape(grid_size)
    # cv2.imshow('Norm Patches', norm_patches)

    object_pca = PCA(n_components=3)

    # extract foreground patches
    mask_indices = [0, *np.cumsum([np.sum(m) for m in masks]), -1]
    fg_patches = np.vstack([patch_tokens[i,masks[i],:] for i in range(1)])

    # fit PCA to foreground, scale each feature to (0,1)
    reduced_patches = object_pca.fit_transform(fg_patches)
    reduced_patches = minmax_scale(reduced_patches)
    patch_image = np.zeros((grid_size[0] * grid_size[1], 3), dtype='float32')
    patch_image[masks[0], :] = reduced_patches[mask_indices[0]:mask_indices[1], :]

    # reshape to grid
    color_patches = patch_image.reshape(grid_size[0], grid_size[1], 3)
    # resize to original frame size for display
    color_patches = cv2.resize(color_patches, (cropped_width, cropped_height), interpolation=cv2.INTER_NEAREST)
    # convert RGB to BGR for OpenCV
    color_patches = cv2.cvtColor(color_patches, cv2.COLOR_RGB2BGR)
    
    return color_patches

def main():
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or change to specific camera index
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit")
    
    # Variables for FPS calculation
    prev_frame_time = 0
    curr_frame_time = 0
    fps = 0
    
    while True:
        # Calculate FPS
        curr_frame_time = time.time()
        if prev_frame_time > 0:
            fps = 1 / (curr_frame_time - prev_frame_time)
        prev_frame_time = curr_frame_time
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Process the frame
        visualization = process_frame(frame)

        cv2.putText(visualization, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Color Patches', visualization)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
