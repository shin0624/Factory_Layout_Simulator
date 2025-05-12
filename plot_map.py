import matplotlib.pyplot as plt
import io
import numpy as np

def generate_layout(detections_per_frame):
    all_positions = {}

    for detections in detections_per_frame:
        for name, (x, y) in detections:
            if name not in all_positions:
                all_positions[name] = []
            all_positions[name].append((x, y))

    # 평균 좌표 계산
    avg_positions = {
        name: np.mean(pos, axis=0)
        for name, pos in all_positions.items()
    }

    # 시각화
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)  # y축 뒤집기 (영상 좌표계 기준)

    for name, (x, y) in avg_positions.items():
        ax.plot(x, y, 'o', label=name)
        ax.text(x + 10, y, name, fontsize=10)

    ax.set_title("Factory 2D Layout (Prototype)")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf