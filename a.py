import cv2
import torch
from edgevos import EdgeVOSPredictor

# 1. Load the model
model = EdgeVOSPredictor("checkpoints/edgevos_spatial_v1.pt")

# 2. Open your custom video file
video = cv2.VideoCapture(r"e:\Github\EdgeVOS\b.mp4")

# Read the first frame and convert to PyTorch tensor [3, H, W]
ret, frame0 = video.read()
frame0_tensor = torch.from_numpy(frame0).permute(2, 0, 1).cuda().float() / 255.0

# Set up the video writer to save the output
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_video = cv2.VideoWriter("tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 3. Initialize from text
mask = model.init_from_text(frame0_tensor, text_prompt="cat")

# Draw green mask on the first frame and save it
overlay = frame0.copy()
overlay[mask.cpu().numpy()] = [0, 255, 0]
out_video.write(cv2.addWeighted(overlay, 0.5, frame0, 0.5, 0))

# 4. Loop through the rest of the video
print("Tracking video... this may take a moment depending on the video length.")
frame_count = 1
while True:
    ret, frame = video.read()
    if not ret: break  # Video is over
    
    if frame_count % 10 == 0:
        print(f"Processing frame {frame_count}...")
        
    # Convert frame to tensor
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0
    
    # Track!
    mask = model.process_frame(frame_tensor)
    
    # Draw the red mask if tracking was successful
    if mask is not None:
        mask_np = mask.cpu().numpy()
        overlay = frame.copy()
        overlay[mask_np] = [0, 0, 255] # Red mask
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
    # Write the frame to the new video file
    out_video.write(frame)
    frame_count += 1

# Clean up
video.release()
out_video.release()
print("Tracking complete! Saved as tracked_output.mp4")
