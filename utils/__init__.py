# utils/__init__.py
# ��Ű�� �ʱ�ȭ ����

def extract_frames(video_path, max_frames=10, skip=10):
    """
    �����󿡼� ������ ���� (���� utils.py �Լ��� ����)
    
    Args:
        video_path: ������ ���� ���
        max_frames: ������ �ִ� ������ ��
        skip: �ǳʶ� ������ ��
        
    Returns:
        ����� ������ ����Ʈ
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