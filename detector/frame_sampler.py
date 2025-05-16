import cv2
import os

def extract_keyframes(video, output_dir, target_frames=50):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // target_frames)
    
    os.makedirs(output_dir, exist_ok=True)
    keyframes = []
    
    for idx in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
            cv2.imwrite(output_path, frame)
            keyframes.append(output_path)
    
    cap.release()
    return keyframes