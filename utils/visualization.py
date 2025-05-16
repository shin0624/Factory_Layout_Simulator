# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class FactoryVisualizer:
    @staticmethod
    def plot_3d_pointcloud(points, save_path=None):
        """3D 포인트 클라우드 시각화"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(
            points[:,0], 
            points[:,1], 
            points[:,2],
            c=points[:,2], 
            cmap='viridis',
            s=20,
            alpha=0.7
        )
        
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Frame Sequence (Z)')
        ax.set_title('3D Factory Point Cloud')
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def create_top_down_view(points, save_path):
        """탑뷰 2D 프로젝션 생성"""
        plt.figure(figsize=(10, 8))
        plt.scatter(points[:,0], points[:,1], 
                   c=points[:,2], cmap='plasma',
                   alpha=0.5, s=15)
        plt.colorbar(label='Frame Sequence')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Top-Down View')
        plt.grid(True)
        plt.savefig(save_path, dpi=100)
        plt.close()

    @staticmethod
    def plot_class_distribution(detections, save_path):
        """객체 분포 시각화"""
        # 수정: 원본은 list of detections를 기대하지만, 실제 입력은 class: count 딕셔너리
        classes = list(detections.keys())
        counts = list(detections.values())
        
        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA726']
        bars = plt.bar(classes, counts, color=colors[:len(classes)])
        
        plt.ylabel('Detection Count')
        plt.title('Object Distribution')
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()