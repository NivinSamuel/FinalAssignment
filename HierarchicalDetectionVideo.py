import torch
from ultralytics import YOLO
import cv2
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
import time


@dataclass
class DetectionConfig:
    """Configuration for object detection"""
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    main_objects: List[str] = None
    sub_objects: Dict[str, List[str]] = None


class HierarchicalDetector:
    def __init__(self, config: DetectionConfig):
        """Initialize the detector with models and configuration"""
        self.config = config
        # Load YOLOv8 model - using nano for CPU efficiency
        self.model = YOLO('yolov8n.pt')
        # Initialize object counter
        self.object_counter = {}

    def _get_unique_id(self, object_name: str) -> int:
        """Generate unique ID for each object type"""
        if object_name not in self.object_counter:
            self.object_counter[object_name] = 0
        self.object_counter[object_name] += 1
        return self.object_counter[object_name]

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
                # Create hierarchical structure for each valid association
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

        return hierarchical_detections

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Process a single frame and return detections and annotated frame"""
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
                main_detections.append(detection)
            elif any(cls_name in subs for subs in self.config.sub_objects.values()):
                sub_detections.append(detection)

        # Associate objects and sub-objects
        hierarchical_detections = self._associate_objects(
            main_detections, sub_detections)

        # Draw detections
        for det in hierarchical_detections:
            main_bbox = det['bbox']
            cv2.rectangle(annotated_frame,
                          (int(main_bbox[0]), int(main_bbox[1])),
                          (int(main_bbox[2]), int(main_bbox[3])),
                          (0, 255, 0), 2)

            sub_bbox = det['subobject']['bbox']
            cv2.rectangle(annotated_frame,
                          (int(sub_bbox[0]), int(sub_bbox[1])),
                          (int(sub_bbox[2]), int(sub_bbox[3])),
                          (255, 0, 0), 2)

        return hierarchical_detections, annotated_frame

    def extract_subobject(self, frame: np.ndarray, detection: Dict, save_path: str = None) -> np.ndarray:
        """Extract and optionally save a sub-object image"""
        bbox = detection['subobject']['bbox']
        crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, crop)

        return crop

    def process_video(self, video_path: str, output_path: str, save_detections: bool = True):
        """Process video file and save results"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              (width, height))

        frame_count = 0
        total_time = 0
        all_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            detections, annotated_frame = self.process_frame(frame)
            process_time = time.time() - start_time

            total_time += process_time
            frame_count += 1

            if save_detections:
                all_detections.append({
                    'frame': frame_count,
                    'detections': detections
                })

            out.write(annotated_frame)

            # Print progress
            if frame_count % 30 == 0:
                avg_fps = frame_count / total_time
                print(
                    f"Processing frame {frame_count}, Average FPS: {avg_fps:.2f}")

        cap.release()
        out.release()

        if save_detections:
            with open('detections.json', 'w') as f:
                json.dump(all_detections, f, indent=2)

        return frame_count / total_time  # Return average FPS


def main():
    # Configuration
    config = DetectionConfig(
        conf_threshold=0.25,
        iou_threshold=0.45,
        main_objects=['person', 'car'],
        sub_objects={
            'person': ['helmet', 'backpack'],
            'car': ['license plate', 'wheel']
        }
    )

    # Initialize detector
    detector = HierarchicalDetector(config)

    # Process video
    video_path = 'input_video.mp4'
    output_path = 'output_video.mp4'

    avg_fps = detector.process_video(video_path, output_path)
    print(f"Processing complete. Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
