import os
import cv2
import numpy as np
import trimesh
from ultralytics import YOLO
from utils.pointcloud_tools import PointCloudProcessor

# YOLO �� �ν��Ͻ� �ʱ�ȭ
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
            7: 'truck',  # Ʈ���� ��ũ����Ʈ�� ���� ����� COCO Ŭ����
            24: 'backpack',  # ��Ÿ ���� ���� Ȱ�� ������ ��ü Ŭ����
            41: 'cup',  # ��Ÿ ���� ���� Ȱ�� ������ ��ü Ŭ����
            73: 'laptop'  # ��Ÿ ���� ���� Ȱ�� ������ ��ü Ŭ����
        }
        # ���� �ڵ��� 'forklift'�� 'conveyor' Ŭ������ COCO �����ͼ¿� ��� ������ Ŭ������ ��ü
    
    def process_frames(self, frame_paths):
        point_cloud = []
        detections = {v: 0 for v in self.class_mapping.values()}

        for frame_idx, path in enumerate(frame_paths):
            img = cv2.imread(path)
            if img is None:
                continue

            results = self.model(img)[0]

            # ���� ����� ���� ��� �ǳʶٱ�
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
                    point_cloud.append([x, y, frame_idx * 50])  # z�࿡ ������ ���� �ݿ�

        if not point_cloud:
            return None

        # ����Ʈ Ŭ���� ��ó��
        points = np.array(point_cloud)
        points = PointCloudProcessor.filter_outliers(points)
        points = PointCloudProcessor.normalize_points(points)

        return (trimesh.points.PointCloud(points), detections)