"""
Object Detection Module for Construction Safety AI.
Uses YOLOv11n (Ultralytics) to detect workers and construction machinery in images.

Stage 1: Object Detection - YOLOv11n detector localizes entities within construction scenes.
"""

import time
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ultralytics import YOLO

from config import (
    YOLO_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    CONSTRUCTION_CLASSES,
    CONSTRUCTION_ENTITY_LABELS,
    PROXIMITY_THRESHOLD_PERCENT,
)


@dataclass
class Detection:
    """Represents a single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    bbox_center: Tuple[float, float]               # (cx, cy)
    bbox_area: float
    entity_label: str  # Human-readable construction label (e.g., "Worker")
    entity_type: str   # Category type (e.g., "worker", "vehicle", "machinery")


@dataclass
class DetectionResult:
    """Represents the full detection results for an image."""
    detections: List[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    inference_time_ms: float = 0.0
    detection_summary_text: str = ""
    spatial_relations_text: str = ""
    spatial_relations: List[dict] = field(default_factory=list)


class ConstructionDetector:
    """
    YOLOv11n-based detector for construction site entities.
    
    Implements Stage 1 of the detection-guided sVLM framework:
    - Detects workers (person class) and construction machinery/vehicles
    - Computes bounding box positions, areas, and spatial relationships
    - Generates natural language summaries for prompt conditioning
    """

    def __init__(self, model_path: str = YOLO_MODEL):
        """Initialize the YOLOv11n detector."""
        self.model_path = model_path
        self._model: Optional[YOLO] = None

    def _load_model(self) -> YOLO:
        """Lazy-load the YOLO model."""
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> DetectionResult:
        """
        Run object detection on a construction site image.
        
        Args:
            image: PIL Image of the construction site
            confidence_threshold: Minimum confidence for detection filtering
            
        Returns:
            DetectionResult containing all detected entities and spatial relations
        """
        model = self._load_model()
        
        # Convert PIL image to numpy for YOLO
        img_array = np.array(image)
        image_width, image_height = image.size
        
        # Run YOLOv11n inference
        start_time = time.time()
        results = model.predict(
            source=img_array,
            conf=confidence_threshold,
            verbose=False,
        )
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Parse detections
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes[i]
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Bounding box in xyxy format
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Compute center and area
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Map to construction entity labels
                    entity_label = CONSTRUCTION_ENTITY_LABELS.get(class_name, class_name.capitalize())
                    entity_type = self._classify_entity_type(class_name)
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        bbox_center=(float(cx), float(cy)),
                        bbox_area=float(area),
                        entity_label=entity_label,
                        entity_type=entity_type,
                    )
                    detections.append(detection)
        
        # Compute spatial relationships
        spatial_relations = self._compute_spatial_relations(detections, image_width, image_height)
        
        # Generate text summaries
        detection_summary = self._generate_detection_summary(detections)
        spatial_summary = self._generate_spatial_relations_summary(spatial_relations, detections)
        
        return DetectionResult(
            detections=detections,
            image_width=image_width,
            image_height=image_height,
            inference_time_ms=inference_time_ms,
            detection_summary_text=detection_summary,
            spatial_relations_text=spatial_summary,
            spatial_relations=spatial_relations,
        )

    def _classify_entity_type(self, class_name: str) -> str:
        """Classify a detected object into a construction entity type."""
        if class_name in ["person"]:
            return "worker"
        elif class_name in ["truck", "car", "bus", "motorcycle", "bicycle"]:
            return "vehicle"
        elif class_name in ["excavator", "crane", "bulldozer", "loader", "cement mixer"]:
            return "machinery"
        else:
            return "other"

    def _compute_spatial_relations(
        self,
        detections: List[Detection],
        image_width: int,
        image_height: int,
    ) -> List[dict]:
        """
        Compute spatial relationships between detected entities.
        
        Analyzes proximity between workers and machinery/vehicles,
        relative positions, and distance as percentage of image diagonal.
        """
        if len(detections) < 2:
            return []
        
        diagonal = np.sqrt(image_width**2 + image_height**2)
        proximity_threshold = diagonal * (PROXIMITY_THRESHOLD_PERCENT / 100.0)
        
        relations = []
        
        # Find all worker detections
        workers = [d for d in detections if d.entity_type == "worker"]
        # Find all machinery/vehicle detections (potential hazard sources)
        hazard_sources = [d for d in detections if d.entity_type in ["vehicle", "machinery"]]
        
        # Compute worker-to-hazard-source relationships
        for worker in workers:
            for source in hazard_sources:
                # Euclidean distance between centers
                dist = np.sqrt(
                    (worker.bbox_center[0] - source.bbox_center[0])**2 +
                    (worker.bbox_center[1] - source.bbox_center[1])**2
                )
                
                # Relative position
                dx = source.bbox_center[0] - worker.bbox_center[0]
                dy = source.bbox_center[1] - worker.bbox_center[1]
                
                position_desc = self._describe_relative_position(dx, dy, image_width, image_height)
                
                # Proximity classification
                is_close = dist < proximity_threshold
                proximity_desc = "nearby" if is_close else "at a distance"
                
                relations.append({
                    "worker": worker.entity_label,
                    "worker_id": detections.index(worker),
                    "source": source.entity_label,
                    "source_id": detections.index(source),
                    "distance_px": float(dist),
                    "distance_pct": float(dist / diagonal * 100),
                    "is_close_proximity": is_close,
                    "relative_position": position_desc,
                    "proximity_description": proximity_desc,
                })
        
        # Also compute all pairwise relationships for general context
        for i, d1 in enumerate(detections):
            for j, d2 in enumerate(detections):
                if i >= j:
                    continue
                dist = np.sqrt(
                    (d1.bbox_center[0] - d2.bbox_center[0])**2 +
                    (d1.bbox_center[1] - d2.bbox_center[1])**2
                )
                
                # Only add general pairwise if not pair isn't already covered
                already_covered = any(
                    (r.get("worker_id") == i and r.get("source_id") == j) or
                    (r.get("worker_id") == j and r.get("source_id") == i)
                    for r in relations
                )
                
                if not already_covered:
                    dx = d2.bbox_center[0] - d1.bbox_center[0]
                    dy = d2.bbox_center[1] - d1.bbox_center[1]
                    position_desc = self._describe_relative_position(dx, dy, image_width, image_height)
                    
                    relations.append({
                        "entity1": d1.entity_label,
                        "entity1_id": i,
                        "entity2": d2.entity_label,
                        "entity2_id": j,
                        "distance_px": float(dist),
                        "distance_pct": float(dist / diagonal * 100),
                        "relative_position": position_desc,
                        "is_close_proximity": dist < proximity_threshold,
                    })
        
        return relations

    def _describe_relative_position(
        self,
        dx: float,
        dy: float,
        image_width: int,
        image_height: int,
    ) -> str:
        """Convert dx/dy offset to a natural language position description."""
        # Normalize offsets
        norm_dx = dx / image_width
        norm_dy = dy / image_height
        
        parts = []
        
        # Horizontal position
        if abs(norm_dx) > 0.05:  # 5% threshold for "significant" offset
            if norm_dx > 0:
                parts.append("to the right")
            else:
                parts.append("to the left")
        
        # Vertical position
        if abs(norm_dy) > 0.05:
            if norm_dy > 0:
                parts.append("below")
            else:
                parts.append("above")
        
        if not parts:
            return "at the same position"
        
        return " and ".join(parts)

    def _generate_detection_summary(self, detections: List[Detection]) -> str:
        """
        Generate a detection-to-text encoding: a natural language summary
        of detected objects for prompt conditioning.
        """
        if not detections:
            return "No construction-related objects detected in the scene."
        
        # Group by entity type
        type_counts = {}
        for d in detections:
            label = d.entity_label
            if label not in type_counts:
                type_counts[label] = []
            type_counts[label].append(d)
        
        lines = []
        for label, dets in type_counts.items():
            count = len(dets)
            if count == 1:
                d = dets[0]
                position = self._describe_position_in_image(
                    d.bbox_center, 640, 480
                )
                lines.append(
                    f"- {d.entity_label} (confidence: {d.confidence:.1%}) "
                    f"located in the {position} of the scene"
                )
            else:
                conf_str = ", ".join(f"{d.confidence:.1%}" for d in dets)
                lines.append(
                    f"- {count} {label}s detected in the scene (confidences: {conf_str})"
                )
        
        return "\n".join(lines)

    def _generate_spatial_relations_summary(
        self,
        relations: List[dict],
        detections: List[Detection],
    ) -> str:
        """
        Generate a natural language summary of spatial relationships
        for the structured prompt used by the sVLM.
        """
        if not relations:
            return "No significant spatial relationships between detected objects."
        
        lines = []
        
        # Focus on worker-hazard proximity relationships first
        worker_relations = [r for r in relations if "worker" in r]
        
        for r in worker_relations:
            if r.get("is_close_proximity", False):
                lines.append(
                    f"- WARNING: {r['worker']} is in close proximity to {r['source']} "
                    f"(positioned {r['relative_position']}, "
                    f"distance: {r['distance_pct']:.1f}% of scene)"
                )
            else:
                lines.append(
                    f"- {r['worker']} is {r['proximity_description']} from {r['source']} "
                    f"(positioned {r['relative_position']})"
                )
        
        # Add general pairwise relations
        general_relations = [r for r in relations if "worker" not in r]
        for r in general_relations:
            e1 = r.get("entity1", "object")
            e2 = r.get("entity2", "object")
            if r.get("is_close_proximity", False):
                lines.append(
                    f"- {e1} and {e2} are close together "
                    f"(positioned {r['relative_position']}, "
                    f"distance: {r['distance_pct']:.1f}% of scene)"
                )
            else:
                lines.append(
                    f"- {e1} is positioned {r['relative_position']} relative to {e2}"
                )
        
        return "\n".join(lines)

    def _describe_position_in_image(
        self,
        center: Tuple[float, float],
        image_width: int,
        image_height: int,
    ) -> str:
        """Describe an object's position in the image using natural language."""
        # Normalize center to 0-1 range
        norm_x = center[0] / max(image_width, 1)
        norm_y = center[1] / max(image_height, 1)
        
        # Horizontal quadrant
        if norm_x < 0.33:
            h_pos = "left"
        elif norm_x < 0.67:
            h_pos = "center"
        else:
            h_pos = "right"
        
        # Vertical quadrant
        if norm_y < 0.33:
            v_pos = "upper"
        elif norm_y < 0.67:
            v_pos = "middle"
        else:
            v_pos = "lower"
        
        # Combine
        if h_pos == "center" and v_pos == "middle":
            return "center"
        elif h_pos == "center":
            return f"{v_pos} center"
        elif v_pos == "middle":
            return f"{h_pos} side"
        else:
            return f"{v_pos}-{h_pos}"