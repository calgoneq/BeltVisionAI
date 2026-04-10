import cv2

from core.utils import measure_width

if __name__ == "__main__":
    image = cv2.imread("test_belt.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray_image.shape
    row = height // 2
    line_data = gray_image[row, :]
    scaled_width = width/100

    red = (0, 0, 255)
    green = (0, 255, 0)
    cv2.line(image, (int(scaled_width), row), (int(scaled_width*100), row), red, 3)

    l_min = round(scaled_width*10)
    l_max = round(scaled_width*50)
    r_min = round(scaled_width*60)
    r_max = round(scaled_width*90)

    belt_width, left_x, right_x = measure_width(line_data, kernel_size=13, l_min=l_min, l_max=l_max, r_min=r_min, r_max=r_max)

    cv2.circle(image, (left_x, row), 10, green, 3)
    cv2.circle(image, (right_x, row), 10, green, 3)
    cv2.putText(image, f"Width: {belt_width} px", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)

    cv2.imshow("Belt", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("result.jpg", image)