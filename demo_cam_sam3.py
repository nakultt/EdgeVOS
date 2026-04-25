import cv2
import torch
import numpy as np
from PIL import Image
import time
import os
import sys

# Assume script is placed in EdgeVOS root; add sam3 repository to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3"))

try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Error importing SAM 3 related modules: {e}")
    print("Ensure you have installed the required dependencies inside sam3/.")
    sys.exit(1)

def main():
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = os.path.join(sam3_root, "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    print("Loading SAM 3.1 model...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # SAM 3 usually expects bfloat16 for efficiency on Ampere+ GPUs
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Activate bfloat16
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    
    processor = Sam3Processor(model, confidence_threshold=0.5)
    
    print("\n=================")
    prompt_text = input("Enter a text prompt to detect (e.g., 'face', 'shoe', 'person'): ").strip()
    if not prompt_text:
        prompt_text = "person"
        
    print(f"Opening camera... Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Create random colors for different detected objects
    np.random.seed(42)
    colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Convert OpenCV BGR frame to RGB for PIL processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Pass image to SAM 3 processor
        inference_state = processor.set_image(pil_img)
        processor.reset_all_prompts(inference_state)
        
        # Infer mask with text prompt
        results = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
        
        # Visualize the outputs
        if "masks" in results and len(results["masks"]) > 0:
            nb_objects = len(results["scores"])
            for i in range(nb_objects):
                score = results["scores"][i].item()
                if score < 0.5:
                    continue
                    
                # The mask shape should be [1, H, W]
                mask = results["masks"][i].squeeze(0).cpu().numpy() > 0.0
                
                # Overlay solid color on target region
                color = colors[i % len(colors)]
                colored_overlay = np.zeros_like(frame)
                colored_overlay[mask] = color
                
                # Blend
                frame = cv2.addWeighted(frame, 1.0, colored_overlay, 0.5, 0)
                
                # Draw bounding box
                if "boxes" in results and len(results["boxes"]) > i:
                    box = results["boxes"][i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)
                    
                    label = f"ID: {i} ({score:.2f})"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
                    
        # Timing / FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"Prompt: {prompt_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Display the result
        cv2.imshow("SAM 3.1 Webcam Inference", frame)
        
        # Wait for 'q' to break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
