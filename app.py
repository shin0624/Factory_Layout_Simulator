import gradio as gr
import tempfile
import os
import numpy as np
from detector.frame_sampler import extract_keyframes
from detector.yolo_pipeline import YOLO3DReconstructor
from exporters.webgl_exporter import export_to_webgl
from utils.visualization import FactoryVisualizer

def process_video(video):
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 키프레임 추출
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = extract_keyframes(video, frames_dir)
        
        if not frame_paths:
            raise gr.Error("비디오 파일을 처리할 수 없습니다!")
        
        # 2. 3D 포인트 클라우드 생성
        reconstructor = YOLO3DReconstructor()
        result = reconstructor.process_frames(frame_paths)
        
        if result is None:
            raise gr.Error("영상에서 감지 가능한 객체를 찾을 수 없습니다!")
            
        pointcloud, detections = result
        
        # 3. 시각화 결과 생성
        vis_dir = os.path.join(tmpdir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 3D 포인트 클라우드 시각화
        pc_points = np.array([point[:3] for point in pointcloud.vertices])
        FactoryVisualizer.plot_3d_pointcloud(
            pc_points, 
            save_path=os.path.join(vis_dir, "3d_view.png")
        )
        
        # 탑뷰 시각화
        FactoryVisualizer.create_top_down_view(
            pc_points,
            save_path=os.path.join(vis_dir, "top_view.png")
        )
        
        # 클래스 분포 시각화
        FactoryVisualizer.plot_class_distribution(
            detections,
            save_path=os.path.join(vis_dir, "distribution.png")
        )

        # 4. 모델 및 결과 파일 저장
        obj_path = os.path.join(tmpdir, "factory.obj")
        pointcloud.export(obj_path)
        html_path = export_to_webgl(obj_path, tmpdir)

        return (
            obj_path,
            html_path,
            os.path.join(vis_dir, "3d_view.png"),
            os.path.join(vis_dir, "top_view.png"),
            os.path.join(vis_dir, "distribution.png")
        )

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="공장 영상 업로드 (MP4/AVI)"),
    outputs=[
        gr.Model3D(label="3D 포인트 클라우드 모델"),
        gr.File(label="WebGL 인터랙티브 뷰어"),
        gr.Image(label="3D 공장 구조", type="filepath"),
        gr.Image(label="탑뷰 구조맵", type="filepath"),
        gr.Image(label="객체 분포 분석", type="filepath")
    ],
    title="🏭 AI 공장 구조 분석 시스템",
    description="업로드된 공장 영상에서 3D 구조맵과 분석 리포트를 자동 생성합니다",
    allow_flagging="never",
    examples=[],  # 예시 추가 가능
)

if __name__ == "__main__":
    iface.launch(debug=True)