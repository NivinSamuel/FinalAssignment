import torch
from ultralytics import YOLO
import cv2
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, field
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


@dataclass
class DetectionConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    main_objects: List[str] = field(default_factory=list)
    sub_objects: Dict[str, List[str]] = field(default_factory=dict)


class HierarchicalDetector:
    def __init__(self, config: DetectionConfig):
        """Initialize the detector with models and configuration"""
        self.config = config
        # Load YOLOv8 model - using nano for CPU efficiency
        self.model = YOLO('yolov8n.pt')
        # Initialize object counter
        self.object_counter = {}
        self.frame_detections = {}
        self.person_counter = 1  # Counter for person objects
        self.running = False  # Flag to control detection loop

    def _get_unique_id(self, object_name: str) -> int:
        """Generate unique ID for each object type"""
        if object_name not in self.object_counter:
            self.object_counter[object_name] = 0
        self.object_counter[object_name] += 1
        return self.object_counter[object_name]

    def _get_person_name(self) -> str:
        """Generate dynamic name for each person detected"""
        person_name = f"person{self.person_counter}"
        self.person_counter += 1
        return person_name

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _associate_objects(self, main_dets: List[Dict], sub_dets: List[Dict]) -> List[Dict]:
        """Associate main objects with their sub-objects based on IoU"""
        hierarchical_detections = []

        for main_det in main_dets:
            main_bbox = main_det['bbox']
            associated_subs = []

            for sub_det in sub_dets:
                if self._calculate_iou(main_bbox, sub_det['bbox']) > self.config.iou_threshold:
                    associated_subs.append(sub_det)

            if associated_subs:
                for sub in associated_subs:
                    hierarchical_detections.append({
                        "object": main_det['object'],
                        "id": main_det['id'],
                        "bbox": main_det['bbox'],
                        "confidence": main_det['confidence'],
                        "subobject": {
                            "object": sub['object'],
                            "id": sub['id'],
                            "bbox": sub['bbox'],
                            "confidence": sub['confidence']
                        }
                    })
            else:
                hierarchical_detections.append({
                    "object": main_det['object'],
                    "id": main_det['id'],
                    "bbox": main_det['bbox'],
                    "confidence": main_det['confidence'],
                    "subobject": None
                })

        return hierarchical_detections

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Process a single frame and return detections and annotated frame"""
        if frame is None:
            return [], np.zeros((480, 640, 3), dtype=np.uint8)

        results = self.model(frame)[0]
        frame_detections = []
        annotated_frame = frame.copy()

        # Process main objects
        main_detections = []
        sub_detections = []

        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            cls_name = results.names[int(cls)]

            if conf < self.config.conf_threshold:
                continue

            detection = {
                'object': cls_name,
                'id': self._get_unique_id(cls_name),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf)
            }

            if cls_name in self.config.main_objects:
                if cls_name == "person":
                    detection['object'] = self._get_person_name()
                main_detections.append(detection)
            elif any(cls_name in subs for subs in self.config.sub_objects.values()):
                sub_detections.append(detection)

        hierarchical_detections = self._associate_objects(
            main_detections, sub_detections)

        # Draw detections
        for det in hierarchical_detections:
            main_bbox = det['bbox']
            label = f"{det['object']} ({det['confidence']:.2f})"

            cv2.rectangle(annotated_frame,
                          (int(main_bbox[0]), int(main_bbox[1])),
                          (int(main_bbox[2]), int(main_bbox[3])),
                          (0, 255, 0), 2)
            cv2.putText(annotated_frame, label,
                        (int(main_bbox[0]), int(main_bbox[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if det['subobject']:
                sub_bbox = det['subobject']['bbox']
                sub_label = f"{det['subobject']['object']} ({det['subobject']['confidence']:.2f})"

                cv2.rectangle(annotated_frame,
                              (int(sub_bbox[0]), int(sub_bbox[1])),
                              (int(sub_bbox[2]), int(sub_bbox[3])),
                              (255, 0, 0), 2)
                cv2.putText(annotated_frame, sub_label,
                            (int(sub_bbox[0]), int(sub_bbox[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return hierarchical_detections, annotated_frame

    def save_detections(self):
        """Save all detections to JSON file"""
        if not self.frame_detections:
            print("No detections to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'detections_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(self.frame_detections, f, indent=2)

        print(f"Detections saved to {filename}")

    def stop_detection(self):
        """Stop the detection loop"""
        self.running = False

    def run_live_detection(self, camera_id: int = 0, gui: tk.Tk = None):
        """Run real-time detection from camera feed and update GUI"""
        self.running = True
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        frame_count = 0
        start_time = time.time()
        fps = 0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame_count += 1

                # Process frame
                detections, annotated_frame = self.process_frame(frame)

                # Store detections
                self.frame_detections[frame_count] = {
                    'frame': frame_count,
                    'timestamp': datetime.now().isoformat(),
                    'detections': detections
                }

                # Calculate FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time

                # Convert the frame for Tkinter
                if gui:
                    frame_rgb = cv2.cvtColor(
                        annotated_frame, cv2.COLOR_BGR2RGB)
                    # Resize frame to fit GUI if necessary
                    height, width = frame_rgb.shape[:2]
                    max_height = 600  # Adjust as needed
                    if height > max_height:
                        scale = max_height / height
                        new_width = int(width * scale)
                        frame_rgb = cv2.resize(
                            frame_rgb, (new_width, max_height))

                    img = Image.fromarray(frame_rgb)
                    img_tk = ImageTk.PhotoImage(image=img)
                    gui.image_label.config(image=img_tk)
                    gui.image_label.image = img_tk
                    gui.update_title(fps)
                    gui.update()  # Update the GUI
                else:
                    cv2.imshow('Object Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            print(f"Error during detection: {str(e)}")

        finally:
            self.save_detections()
            cap.release()
            cv2.destroyAllWindows()


class DetectionGUI(tk.Tk):
    def __init__(self, detector: HierarchicalDetector):
        super().__init__()

        self.detector = detector
        self.title("Real-Time Object Detection")
        self.geometry("800x800")

        # Create main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Image label
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(expand=True, fill='both')

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=10)

        # Centering the button frame's contents
        button_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Buttons
        self.start_button = ttk.Button(
            button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side='left', padx=5)

        self.stop_button = ttk.Button(
            button_frame, text="Stop", command=self.stop_detection)
        self.stop_button.pack(side='left', padx=5)

        self.quit_button = ttk.Button(
            button_frame, text="Quit", command=self.quit_app)
        self.quit_button.pack(side='left', padx=5)

        # Protocol for window closing
        self.protocol("WM_DELETE_WINDOW", self.quit_app)

    def update_title(self, fps: str):
        """Update window title with current FPS"""
        self.title(f"Real-Time Object Detection - {fps}")

    def start_detection(self):
        """Start the detection in the GUI"""
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.detector.run_live_detection(camera_id=0, gui=self)

    def stop_detection(self):
        """Stop the detection"""
        self.detector.stop_detection()
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def quit_app(self):
        """Safely quit the application"""
        self.detector.stop_detection()
        self.quit()


def main():
    # Configuration
    config = DetectionConfig(
        conf_threshold=0.25,
        iou_threshold=0.45,
        main_objects=['person', 'car', 'bike'],
        sub_objects={
            'person': ['helmet', 'backpack'],
            'car': ['license plate', 'wheel'],
            'bike': ['helmet']
        }
    )

    # Initialize detector
    detector = HierarchicalDetector(config)

    # Initialize GUI and run
    gui = DetectionGUI(detector)
    gui.mainloop()


if __name__ == "__main__":
    main()
