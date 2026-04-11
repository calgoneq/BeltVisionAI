from ultralytics import YOLO

def detect_defects(image_path: str, model_path: str) -> list:
    model = YOLO(model_path)
    image = image_path
    results = model(image, conf=0.05)
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
        print(f"Error: {e}") 

    return all_detections

if __name__ == "__main__":
    print(detect_defects())