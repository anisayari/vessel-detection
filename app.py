from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
MODEL_URL = "https://huggingface.co/DefendIntelligence/vessel-detection/resolve/main/models/best.pt"
EXAMPLES_DIR = ROOT / "examples"
MAX_TILES = 196
BATCH_SIZE = 8


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run `python run_local.py` or download it from {MODEL_URL}."
        )
    return YOLO(str(MODEL_PATH))


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _iter_tiles(image: Image.Image, tile_size: int, overlap: int) -> list[tuple[Image.Image, int, int]]:
    width, height = image.size
    x_starts = _tile_starts(width, tile_size, overlap)
    y_starts = _tile_starts(height, tile_size, overlap)
    tiles: list[tuple[Image.Image, int, int]] = []
    for y in y_starts:
        for x in x_starts:
            right = min(width, x + tile_size)
            bottom = min(height, y + tile_size)
            tiles.append((image.crop((x, y, right, bottom)), x, y))
    return tiles


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    remaining = sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)
    kept: list[dict] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            item
            for item in remaining
            if item["class_id"] != current["class_id"]
            or _box_iou(item["box"], current["box"]) < iou_threshold
        ]
    return kept


def _model_names(model: YOLO) -> dict[int, str]:
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def _predict_tiles(
    image: Image.Image,
    *,
    confidence: float,
    iou: float,
    tile_size: int,
    overlap: int,
    max_det: int,
) -> tuple[list[dict], int]:
    model = load_model()
    names = _model_names(model)
    rgb_image = image.convert("RGB")
    safe_tile_size = max(320, int(tile_size))
    safe_overlap = max(0, min(int(overlap), safe_tile_size - 32))
    tiles = _iter_tiles(rgb_image, safe_tile_size, safe_overlap)

    if len(tiles) > MAX_TILES:
        raise ValueError(
            f"Image too large for this CPU Space: {len(tiles)} tiles. "
            f"Resize the image or increase the tile size."
        )

    detections: list[dict] = []
    for start in range(0, len(tiles), BATCH_SIZE):
        batch = tiles[start : start + BATCH_SIZE]
        batch_images = [tile for tile, _, _ in batch]
        results = model.predict(
            source=batch_images,
            conf=float(confidence),
            iou=float(iou),
            imgsz=safe_tile_size,
            max_det=int(max_det),
            verbose=False,
        )
        for result, (_, offset_x, offset_y) in zip(results, batch):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    {
                        "label": names.get(int(class_id), f"class_{int(class_id)}"),
                        "class_id": int(class_id),
                        "confidence": float(score),
                        "box": [
                            float(x1 + offset_x),
                            float(y1 + offset_y),
                            float(x2 + offset_x),
                            float(y2 + offset_y),
                        ],
                    }
                )

    detections = _nms(detections, float(iou))
    detections = detections[: int(max_det)]
    return detections, len(tiles)


def _draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    line_width = max(2, round(max(annotated.size) / 420))

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['label']} {detection['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 48), width=line_width)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle((x1, label_y, x1 + text_w + 8, label_y + text_h + 6), fill=(255, 64, 48))
        draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)

    return annotated


def _table_rows(detections: list[dict]) -> list[list[object]]:
    rows: list[list[object]] = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        rows.append(
            [
                index,
                detection["label"],
                round(float(detection["confidence"]), 4),
                round(x1, 1),
                round(y1, 1),
                round(x2, 1),
                round(y2, 1),
                round(x2 - x1, 1),
                round(y2 - y1, 1),
            ]
        )
    return rows


def detect_boats(
    image: Image.Image | None,
    confidence: float,
    iou: float,
    tile_size: int,
    overlap: int,
    max_det: int,
) -> tuple[Image.Image | None, list[list[object]], str]:
    if image is None:
        return None, [], "Upload a satellite image to run detection."

    try:
        detections, tile_count = _predict_tiles(
            image,
            confidence=confidence,
            iou=iou,
            tile_size=tile_size,
            overlap=overlap,
            max_det=max_det,
        )
    except Exception as exc:
        return image, [], f"Inference error: {exc}"

    annotated = _draw_detections(image, detections)
    rows = _table_rows(detections)
    if detections:
        summary = f"{len(detections)} detection(s) above {confidence:.2f}. Tiles analyzed: {tile_count}."
    else:
        summary = f"No detections above {confidence:.2f}. Tiles analyzed: {tile_count}."
    return annotated, rows, summary


def _example_paths() -> list[list[str]]:
    paths = sorted(EXAMPLES_DIR.glob("*.png"))
    return [[str(path)] for path in paths[:10]]


with gr.Blocks(title="Vessel Detection") as demo:
    gr.Markdown(
        """
        # Vessel Detection

        Fine-tuned YOLOv8s model for detecting vessels in RGB satellite imagery.
        Upload a satellite image or select an example, then run detection.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Satellite image")
            confidence_input = gr.Slider(0.01, 0.95, value=0.20, step=0.01, label="Confidence threshold")
            iou_input = gr.Slider(0.05, 0.90, value=0.45, step=0.05, label="IoU NMS")
            tile_size_input = gr.Slider(320, 1024, value=640, step=32, label="Tile size")
            overlap_input = gr.Slider(0, 256, value=96, step=16, label="Tile overlap")
            max_det_input = gr.Slider(1, 200, value=80, step=1, label="Max detections")
            run_button = gr.Button("Detect vessels", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Annotated image")
            summary_output = gr.Markdown()
            table_output = gr.Dataframe(
                headers=["#", "label", "confidence", "x1", "y1", "x2", "y2", "width", "height"],
                datatype=["number", "str", "number", "number", "number", "number", "number", "number", "number"],
                label="Detections",
            )

    run_button.click(
        fn=detect_boats,
        inputs=[image_input, confidence_input, iou_input, tile_size_input, overlap_input, max_det_input],
        outputs=[output_image, table_output, summary_output],
    )

    gr.Examples(
        examples=_example_paths(),
        inputs=[image_input],
        label="Example images",
    )


if __name__ == "__main__":
    launch_kwargs = {}
    if os.environ.get("GRADIO_SERVER_NAME"):
        launch_kwargs["server_name"] = os.environ["GRADIO_SERVER_NAME"]
    if os.environ.get("GRADIO_SERVER_PORT"):
        launch_kwargs["server_port"] = int(os.environ["GRADIO_SERVER_PORT"])
    demo.launch(**launch_kwargs)
