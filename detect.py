import torch
from ultralytics import YOLO
import cv2
from utils import extract_frames

# YOLO 모델 로드
model = YOLO("models/yolov8m.pt")  # yolov8 가중치 파일 위치

def detect_objects_in_video(video_path):
    detections = []
    frames = extract_frames(video_path, max_frames=10)  # 성능상 샘플 프레임만 사용
    for frame in frames:
        results = model(frame)[0]  # 프레임당 감지 수행
        for box in results.boxes:
            cls = model.names[int(box.cls)]
            xyxy = box.xyxy.tolist()[0]
            detections.append({"class": cls, "bbox": xyxy})
    return detections
