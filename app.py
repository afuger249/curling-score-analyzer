import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import math

# Store API details in memory
ROBOFLOW_MODEL = "curling-rock-detection/8"
ROBOFLOW_API_KEY = "ui1CMmQxdGFZRQ8Vs14E"
ROBOFLOW_SIZE = 416

def classify_stone_color(image, x, y, width, height):
    """Classifies stone color based on HSV values."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    stone_roi = hsv[y - height // 2: y + height // 2, x - width // 2: x + width // 2]

    # Define HSV ranges for red and yellow
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Create masks
    red_mask1 = cv2.inRange(stone_roi, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(stone_roi, red_lower2, red_upper2)
    yellow_mask = cv2.inRange(stone_roi, yellow_lower, yellow_upper)

    # Count pixels
    red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    if red_pixels > yellow_pixels:
        return "red"
    else:
        return "yellow"

st.title("Curling Score Analyzer")

option = st.radio("Choose input method", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a photo of the curling end", type=["jpg", "png", "jpeg"], key="curling_upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        process_image(image)

elif option == "Use Webcam":
    st.write("Using Webcam... Press 'q' to exit.")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        process_image(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    response = requests.post(
        f"https://detect.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}",
        files={"file": img_bytes},
        data={"size": ROBOFLOW_SIZE}
    )

    if response.status_code == 200:
        data = response.json()
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        process_detections(image_cv, data)
    else:
        st.error("Error analyzing the image. Check your API key and model name.")

def process_detections(image_cv, data):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=50, maxRadius=150)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        HOUSE_CENTER_X, HOUSE_CENTER_Y, HOUSE_RADIUS = circles[0][0]
    else:
        HOUSE_CENTER_X = image_cv.shape[1] // 2
        HOUSE_CENTER_Y = image_cv.shape[0] // 2
        HOUSE_RADIUS = min(image_cv.shape[0], image_cv.shape[1]) // 4

    cv2.circle(image_cv, (HOUSE_CENTER_X, HOUSE_CENTER_Y), HOUSE_RADIUS, (255, 0, 0), 3)

    stones = []
    red_stones = []
    yellow_stones = []

    for prediction in data.get("predictions", []):
        x, y, width, height = int(prediction["x"]), int(prediction["y"]), int(prediction["width"]), int(prediction["height"])
        distance = math.sqrt((x - HOUSE_CENTER_X) ** 2 + (y - HOUSE_CENTER_Y) ** 2)
        team = classify_stone_color(image_cv, x, y, width, height)
        stones.append((x, y, width, height, distance, team))
        if team == "red":
            red_stones.append(distance)
        else:
            yellow_stones.append(distance)

    stones.sort(key=lambda s: s[4])
    closest_team = stones[0][5] if stones and stones[0][4] <= 183 else None

    score = 0
    if closest_team:
        opponent_stones = red_stones if closest_team == "yellow" else yellow_stones
        for stone in stones:
            if stone[5] == closest_team and (not opponent_stones or stone[4] < min(opponent_stones)):
                score += 1
            else:
                break

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_cv, f"Score: {score}-0", (50, 50), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    image_output = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    st.image(image_output, caption="Detected Curling Rocks with Score", use_column_width=True)






