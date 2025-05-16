import os
import cv2
import numpy as np
import trimesh
from ultralytics import YOLO
from utils.pointcloud_tools import PointCloudProcessor

# YOLO 모델 인스턴스 초기화
_MODEL = None

def get_yolo_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO('yolov8m.pt')
    return _MODEL

class YOLO3DReconstructor:
    def __init__(self):
        self.model = get_yolo_model()
        self.class_mapping = {
            0: 'person',
            7: 'truck',  # 트럭이 포크리프트에 가장 가까운 COCO 클래스
            24: 'backpack',  # 기타 공장 장비로 활용 가능한 대체 클래스
            41: 'cup',  # 기타 공장 장비로 활용 가능한 대체 클래스
            73: 'laptop'  # 기타 공장 장비로 활용 가능한 대체 클래스
        }
        # 원본 코드의 'forklift'와 'conveyor' 클래스가 COCO 데이터셋에 없어서 유사한 클래스로 대체
    
    def process_frames(self, frame_paths):
        point_cloud = []
        detections = {v: 0 for v in self.class_mapping.values()}

        for frame_idx, path in enumerate(frame_paths):
            img = cv2.imread(path)
            if img is None:
                continue

            results = self.model(img)[0]

            # 감지 결과가 없는 경우 건너뛰기
            if len(results.boxes) == 0:
                continue

            boxes_xywh = results.boxes.xywh.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes_xywh, class_ids):
                cls_id = int(cls)
                if cls_id in self.class_mapping:
                    label = self.class_mapping[cls_id]
                    detections[label] += 1
                    x, y, w, h = box[:4]
                    point_cloud.append([x, y, frame_idx * 50])  # z축에 프레임 순서 반영

        if not point_cloud:
            return None

        # 포인트 클라우드 후처리
        points = np.array(point_cloud)
        points = PointCloudProcessor.filter_outliers(points)
        points = PointCloudProcessor.normalize_points(points)

        return (trimesh.points.PointCloud(points), detections)