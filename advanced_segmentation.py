
import cv2
import numpy as np
from pathlib import Path

# Create output directory
Path("outputs").mkdir(exist_ok=True)

# Load an image
image = cv2.imread("images/photo.jpg")

# Create a synthetic test image if no photograph exists
if image is None:
    image = np.zeros((500, 700, 3), dtype=np.uint8)
    image[:] = (210, 220, 210)  # Light background in BGR

    # Simulate green leaves
    cv2.ellipse(
        image, (220, 230), (100, 45),
        -30, 0, 360, (30, 170, 40), -1
    )
    cv2.ellipse(
        image, (420, 300), (110, 50),
        25, 0, 360, (40, 190, 50), -1
    )
    print("Using a synthetic test image.")

# Step 1: Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 2: Define green color range
lower_green = np.array([35, 50, 40], dtype=np.uint8)
upper_green = np.array([90, 255, 255], dtype=np.uint8)

# Step 3: Create binary mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# Step 4: Clean the mask using morphological operations
kernel = np.ones((5, 5), dtype=np.uint8)

clean_mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN, kernel
)

clean_mask = cv2.morphologyEx(
    clean_mask, cv2.MORPH_CLOSE, kernel
)

# Step 5: Find external contours
contours, _ = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Step 6: Extract green regions and draw bounding boxes
result = image.copy()

for contour in contours:
    area = cv2.contourArea(contour)

    # Ignore small regions
    if area < 300:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(
        result, (x, y), (x + w, y + h),
        (0, 0, 255), 2
    )

    cv2.putText(
        result,
        f"Area: {int(area)}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

# Step 7: Save results
cv2.imwrite("outputs/green_mask.png", mask)
cv2.imwrite("outputs/clean_mask.png", clean_mask)
cv2.imwrite("outputs/segmented_result.jpg", result)

# Step 8: Display results
cv2.imshow("Original Image", image)
cv2.imshow("Green Mask", mask)
cv2.imshow("Clean Mask", clean_mask)
cv2.imshow("Detected Green Regions", result)

print("Contours found:", len(contours))
print("Results saved in outputs/")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("I love ALLIANCE")