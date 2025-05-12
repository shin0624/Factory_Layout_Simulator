from ultralytics import YOLO
import cv2

# 감지할 클래스 매핑 (YOLO COCO 클래스 기준)
TARGET_CLASSES = {
    0: 'person',
    2: 'car',  # forklift 대체 (fine-tuned 모델에서 forklift 사용 가능)
    39: 'screwdriver',  # machine 임시 대체
    67: 'cell phone'  # conveyor belt 임시 대체
}

def run_detection(image, model):
    results = model(image)[0]
    detections = []
    
    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id in TARGET_CLASSES:
            name = TARGET_CLASSES[cls_id]
            x, y = box.xywh[0][:2].tolist()
            detections.append((name, (x, y)))

    return detections