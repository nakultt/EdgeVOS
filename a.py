import cv2
from edgevos import EdgeVOSPredictor

VIDEO_IN = r"e:\Github\EdgeVOS\b.mp4"
VIDEO_OUT = "tracked.mp4"
PROMPT = "cat"

# 1. Load the model
model = EdgeVOSPredictor("checkpoints/edgevos_spatial_v1.pt")

# 2. Open your custom video file
video = cv2.VideoCapture(VIDEO_IN)
if not video.isOpened():
    raise SystemExit(f"Could not open {VIDEO_IN}")

# Set up the video writer to save the output
fps = video.get(cv2.CAP_PROP_FPS) or 30
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
out_video = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 3. Initialize from text on the first frame.
#    frame_to_tensor converts OpenCV BGR uint8 -> RGB float tensor on the model's device.
ret, frame = video.read()
mask = model.init_from_text(model.frame_to_tensor(frame), text_prompt=PROMPT)
if mask is None:
    print(f"'{PROMPT}' not visible in the first frame; will keep searching.")

# 4. Loop through the video. Green = (re)grounded frame, red = tracked frame.
print("Tracking video... this may take a moment depending on the video length.")
frame_count = 0
while ret:
    if mask is not None:
        overlay = frame.copy()
        grounded = (model.last_event or "").startswith(("init", "reground"))
        overlay[mask.cpu().numpy()] = [0, 255, 0] if grounded else [0, 0, 255]
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    out_video.write(frame)

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count}/{total} frames...")

    ret, frame = video.read()
    if ret:
        mask = model.process_frame(model.frame_to_tensor(frame))

# Clean up
video.release()
out_video.release()
print(f"Tracking complete! Saved as {VIDEO_OUT}")
