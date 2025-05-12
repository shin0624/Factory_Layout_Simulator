import gradio as gr
from detect import detect_objects_in_video
from map_builder import generate_factory_map

def process_videos(video_paths):
    print("[INFO] 입력된 비디오 수:", len(video_paths))

    all_detections = []
    for path in video_paths:
        print(f"[INFO] 비디오 처리 중: {path}")
        detections = detect_objects_in_video(path)
        all_detections.extend(detections)  # 모든 비디오의 구조물 감지 결과를 누적

    # 공장 구조 미니맵 생성
    minimap_image = generate_factory_map(all_detections)
    return minimap_image

demo = gr.Interface(
    fn=process_videos,
    inputs=gr.File(file_types=["video"], label="작업장 영상 업로드"),
    outputs=gr.Image(type="pil", label="공장 구조 미니맵 시뮬레이션"),
    title="Factory Layout Simulator",
    description="공장 작업 영상에서 구조물을 감지하고 2D 미니맵 형태로 시각화합니다."
)

demo.launch()
