import cv2
import numpy as np

from core.utils import measure_width
from core.ai import detect_defects

if __name__ == "__main__":
    image_path="sample_seams.jpg"
    model_path="best.pt"

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray_image.shape
    scaled_width = width/100

    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    purple = (116, 20, 73)
    orange = (21, 71, 229)

    colors = [blue, green, red, purple, orange]

    positions = [0.2, 0.35, 0.5, 0.65, 0.8]
    belt_widths = []

    l_min = round(scaled_width*10)
    l_max = round(scaled_width*50)
    r_min = round(scaled_width*60)
    r_max = round(scaled_width*90)

    for i, v in enumerate(positions):
        position = int(height * v)
        line_data = gray_image[position, :]
        belt_width, left_x, right_x = measure_width(line_data, kernel_size=13, l_min=l_min, l_max=l_max, r_min=r_min, r_max=r_max)
        belt_widths.append(belt_width)
        cv2.line(image, (int(scaled_width), position), (int(scaled_width*100), position), colors[i], 3)
        cv2.circle(image, (left_x, position), 10, colors[i], 3)
        cv2.circle(image, (right_x, position), 10, colors[i], 3)
        cv2.imwrite(f"result_{i}.jpg", image)

    detection = detect_defects(image_path=image_path, model_path=model_path)

    for det in detection:
        x1 = int(det['box'][0])
        y1 = int(det['box'][1])
        x2 = int(det['box'][2])
        y2 = int(det['box'][3])
        
        conf = det['confidence']

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"confidence: {conf:.2f}", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[2], 2)
    
    cv2.imwrite(f"detection_result.jpg", image)

    avg_width = np.mean(belt_widths)
    avg_stability = np.std(belt_widths)
    cv2.putText(gray_image, f"avg_width: {avg_width} px", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[2], 2)
    cv2.putText(gray_image, f"avg_stability: {avg_stability:.2f}", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[2], 2)

    cv2.imwrite(f"result_all.png", gray_image)