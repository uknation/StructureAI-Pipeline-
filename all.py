import cv2
import numpy as np
import easyocr

# ==============================
# 1. LOAD IMAGE (COMMON)
# ==============================
image_path = 'enimg.png'
img = cv2.imread(image_path)
output = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==============================
# 2. PREPROCESSING (COMMON)
# ==============================
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Two thresholds for different tasks
_, binary_walls = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
_, binary_windows = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

# ==============================
# 3. SCALE CALIBRATION
# ==============================
pixels_4m = 100
ppm = pixels_4m / 4  # Pixels per meter

# ==============================
# 4. WALL DETECTION
# ==============================
lines = cv2.HoughLinesP(binary_walls, 1, np.pi/180, 
                        threshold=100, minLineLength=50, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ==============================
# 5. ROOM AREA DETECTION
# ==============================
contours, _ = cv2.findContours(binary_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area_px = cv2.contourArea(cnt)
    if area_px < 500:  # noise filter
        continue

    area_m2 = area_px / (ppm ** 2)
    x, y, w, h = cv2.boundingRect(cnt)

    cv2.putText(output, f"{area_m2:.1f} sqm", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

# ==============================
# 6. WINDOW DETECTION
# ==============================
contours, _ = cv2.findContours(binary_windows, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

window_count = 0

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = float(w)/h if h != 0 else 0
    inv_aspect_ratio = float(h)/w if w != 0 else 0

    is_window = (aspect_ratio > 3.5 or inv_aspect_ratio > 3.5) and (30 < w < 200 or 30 < h < 200)

    if is_window:
        window_count += 1
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(output, f"W{window_count}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# ==============================
# 7. OCR ROOM NAME DETECTION
# ==============================
reader = easyocr.Reader(['en'])

allowed_rooms = [
    "LIVING ROOM", "BEDROOM", "KITCHEN", "BATHROOM",
    "TOILET", "DINING ROOM", "STUDY ROOM", "GUEST ROOM",
    "STORE ROOM", "BALCONY", "BATH"
]

results = reader.readtext(img)
detected_rooms = []

for (bbox, text, prob) in results:
    name = text.strip().upper()

    match = next((item for item in allowed_rooms if item in name), None)

    if match:
        detected_rooms.append(name)

        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        cv2.rectangle(output, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(output, str(len(detected_rooms)),
                    (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# ==============================
# 8. SUMMARY PANEL
# ==============================
h, w, _ = output.shape
bottom_margin = 180

canvas = np.full((h + bottom_margin, w, 3), 255, dtype=np.uint8)
canvas[0:h, 0:w] = output

start_y = h + 30

cv2.putText(canvas, "ROOM SUMMARY:", (30, start_y),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2)

for i, room in enumerate(detected_rooms):
    col = i // 4
    row = i % 4

    cv2.putText(canvas, f"[{i+1}] {room}",
                (40 + col * 350, start_y + 40 + row * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2)

# ==============================
# 9. FINAL OUTPUT
# ==============================
print(f"Total Rooms: {len(detected_rooms)}")
print(f"Total Windows: {window_count}")

cv2.imshow('Final Combined Output', canvas)
cv2.imwrite('Final_Output.png', canvas)

cv2.waitKey(0)
cv2.destroyAllWindows()
