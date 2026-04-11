from ultralytics import YOLO

from core.config import DEFAULT_CONF
from core.logger_config import logger

def detect_defects(image_path: str, model_path: str) -> list[dict]:
    model = YOLO(model_path)
    image = image_path
    results = model(image, conf=DEFAULT_CONF)
    all_detections = []

    try: 
        for r in results:
            label = r.names[0]

            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                data = {"box": coords, "confidence": conf, "label": label}
                all_detections.append(data)

    except ValueError as e:
        logger.error(f"Error: {e}") 

    return all_detections