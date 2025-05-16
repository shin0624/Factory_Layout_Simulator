# utils/pointcloud_tools.py
import numpy as np
import trimesh
from typing import List

class PointCloudProcessor:
    @staticmethod
    def normalize_points(points: np.ndarray) -> np.ndarray:
        """포인트 클라우드 정규화"""
        if len(points) == 0:
            return points
            
        # Z-score 정규화
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        return (points - mean) / (std + 1e-8)

    @staticmethod
    def filter_outliers(points: np.ndarray, z_thresh=3.0) -> np.ndarray:
        """이상치 제거"""
        if len(points) == 0:
            return points
            
        z_scores = np.abs(points - np.mean(points, axis=0)) / np.std(points, axis=0)
        return points[(z_scores < z_thresh).all(axis=1)]

    @staticmethod
    def create_mesh(points: np.ndarray) -> trimesh.Trimesh:
        """포인트 클라우드에서 메시 생성"""
        return trimesh.points.PointCloud(points)

    @staticmethod
    def merge_pointclouds(clouds: List[np.ndarray]) -> np.ndarray:
        """다중 포인트 클라우드 병합"""
        return np.vstack(clouds) if clouds else np.array([])

    @staticmethod
    def save_as_obj(points: np.ndarray, filename: str):
        """OBJ 파일로 저장"""
        mesh = trimesh.points.PointCloud(points)
        mesh.export(filename)