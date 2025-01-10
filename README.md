# Real-Time Object Detection with Hierarchical Association

This project is a **real-time object detection application** built using Python. It uses the **YOLOv8 model** for detecting objects in a video feed from a camera. The detections are enhanced with a **hierarchical association**, meaning it can associate main objects (like a person) with their related sub-objects (like a backpack).

---

## Simplified Explanation

### What It Does:
- Detects objects in a live video feed using a lightweight YOLO model (good for performance).
- Associates main objects (e.g., "person") with sub-objects (e.g., "backpack") based on their spatial overlap.
- Displays the detections on a graphical interface (GUI) in real-time.
- Allows you to start, stop, and quit the live detection from the GUI.
- Saves detection results into a JSON file for later use.

### How It Works:
- **Object Detection**: The YOLO model processes the video frames, detecting objects and their locations.
- **Object Association**: If one object is close to another (like a backpack near a person), they are grouped together.
- **Visualization**: Detected objects are drawn on the video frame with labels and boxes, and the frames are displayed in the GUI.
- **Interaction**: Users can control the detection via buttons in the GUI.

---

## 🛠 Tech Stack

- **YOLOv8**: For object detection.
- **OpenCV**: For video processing and display.
- **Tkinter**: To create a graphical user interface.
- **PIL (Pillow)**: To display images in the GUI.

---

## 🚀 How to Run the Project

### Prerequisites
1. Python installed (preferably Python 3.8+).
2. Install required libraries by running the following command in your terminal:
   ```bash
   pip install torch ultralytics opencv-python-headless pillow

## Steps to Run:

### Download the YOLO Model:
The script uses the YOLOv8 model (`yolov8n.pt`), which is downloaded automatically when the code runs for the first time.

### Prepare the Code:
Save the entire script in a Python file, e.g., `hierarchical_detection.py`.

### Run the Application:
Execute the script by running:
```bash
python hierarchical_detection.py
### Control the GUI:
A window will appear with buttons:
- **Start Detection**: Starts detecting objects in the camera feed.
- **Stop**: Stops the detection loop.
- **Quit**: Safely exits the application.

Detected objects will appear as labeled boxes on the video feed.

### View Saved Detections:
After stopping the detection, the program saves the results in a JSON file (e.g., `detections_YYYYMMDD_HHMMSS.json`).

---

## 📹 How to Test Without a Camera

If you don't have a camera or want to use a video file:
1. Replace the `cv2.VideoCapture(camera_id)` line with:
   ```python
   cap = cv2.VideoCapture('path_to_video.mp4')


## Dependencies

To run this project, you need to install the following Python dependencies:

1. **torch** - PyTorch library for deep learning.
2. **ultralytics** - For YOLOv8 (the object detection model).
3. **opencv-python** - For handling video feed and displaying images.
4. **Pillow** - For handling images in the Tkinter GUI.
5. **tkinter** - For the GUI framework (should already be included with Python).
6. **numpy** - For handling arrays and numerical operations.
7. **dataclasses** - For handling structured data (included in Python 3.7+).
8. **json** - For saving detections in JSON format (built-in Python module).
9. **datetime** - For timestamps in the JSON file (built-in Python module).

### Installation

You can install the required dependencies using `pip`:

```bash
pip install torch
pip install ultralytics
pip install opencv-python
pip install pillow
pip install numpy

