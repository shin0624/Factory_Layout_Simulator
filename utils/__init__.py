# utils/__init__.py
# 패키지 초기화 파일

def extract_frames(video_path, max_frames=10, skip=10):
    """
    동영상에서 프레임 추출 (기존 utils.py 함수를 통합)
    
    Args:
        video_path: 동영상 파일 경로
        max_frames: 추출할 최대 프레임 수
        skip: 건너뛸 프레임 수
        
    Returns:
        추출된 프레임 리스트
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        count += 1
    cap.release()
    return frames