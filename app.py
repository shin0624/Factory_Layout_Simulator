import gradio as gr
from detect import run_detection
from utils import extract_frames
from plot_map import generate_layout
from ultralytics import YOLO

# GPU에서 작동 가능한 중간 사이즈 모델
model = YOLO('yolov8m.pt')

def process_video(video):
    frames = extract_frames(video.name)
    detections_per_frame = []

    for frame in frames:
        detections = run_detection(frame, model)
        detections_per_frame.append(detections)

    layout_img = generate_layout(detections_per_frame)
    return layout_img

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="공장 작업 영상 업로드"),
    outputs=gr.Image(label="2D 공장 미니맵"),
    title="공장 구조 인식 시뮬레이터",
    description="작업 현장 영상을 분석하여 구조물의 위치를 추론하고 2D 평면도를 생성합니다."
)

if __name__ == "__main__":
    iface.launch()
