# utils/pointcloud_tools.py
import numpy as np
import trimesh
from typing import List

class PointCloudProcessor:
    @staticmethod
    def normalize_points(points: np.ndarray) -> np.ndarray:
        """����Ʈ Ŭ���� ����ȭ"""
        if len(points) == 0:
            return points
            
        # Z-score ����ȭ
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        # ǥ�������� 0�� ��� ���
        std = np.where(std < 1e-8, 1, std)
        return (points - mean) / std

    @staticmethod
    def filter_outliers(points: np.ndarray, z_thresh=3.0) -> np.ndarray:
        """�̻�ġ ����"""
        if len(points) == 0:
            return points
        
        if len(points) <= 3:  # ����Ʈ�� �ʹ� ������ ���͸� ����
            return points
            
        z_scores = np.abs(points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 1e-8)
        mask = (z_scores < z_thresh).all(axis=1)
        
        # ��� ����Ʈ�� ���͸����� �ʵ��� ��ȣ
        if not np.any(mask):
            return points
            
        return points[mask]

    @staticmethod
    def create_mesh(points: np.ndarray) -> trimesh.Trimesh:
        """����Ʈ Ŭ���忡�� �޽� ����"""
        return trimesh.points.PointCloud(points)

    @staticmethod
    def merge_pointclouds(clouds: List[np.ndarray]) -> np.ndarray:
        """���� ����Ʈ Ŭ���� ����"""
        return np.vstack(clouds) if clouds else np.array([])

    @staticmethod
    def save_as_obj(points: np.ndarray, filename: str):
        """OBJ ���Ϸ� ����"""
        mesh = trimesh.points.PointCloud(points)
        mesh.export(filename)