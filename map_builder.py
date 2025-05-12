from PIL import Image, ImageDraw

def generate_factory_map(detections, canvas_size=(512, 512)):
    img = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(img)

    class_colors = {
        "conveyor": "blue",
        "machine": "gray",
        "person": "green",
        "forklift": "orange"
    }

    for det in detections:
        cls = det["class"]
        bbox = det["bbox"]
        color = class_colors.get(cls, "red")
        # 단순히 bbox 좌표를 축소하여 canvas에 표시
        x0 = int(bbox[0] / 4)
        y0 = int(bbox[1] / 4)
        x1 = int(bbox[2] / 4)
        y1 = int(bbox[3] / 4)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, y0), cls, fill=color)

    return img
