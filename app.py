import gradio as gr
from detect import run_detection
from utils import extract_frames
from plot_map import generate_layout
from ultralytics import YOLO

# GPU 자원 활용 모델 로딩
model = YOLO('yolov8m.pt')  # Hugging Face GPU에서 충분히 작동

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
    description="영상에서 구조물을 인식하여 2D 레이아웃을 생성합니다."
)

if __name__ == "__main__":
    iface.launch()