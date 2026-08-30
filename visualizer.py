"""
Image Visualization and Annotation Module for Construction Safety AI.
Draws detection bounding boxes and hazard annotations on construction site images.

Visualization: Annotated output images with detection results and hazard zone highlighting.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple

from config import DETECTION_COLORS, SEVERITY_COLORS
from detector import Detection, DetectionResult
from response_parser import HazardAssessment


class Visualizer:
    """
    Creates annotated visualizations of construction site images.
    
    Features:
    - Draws YOLOv11n detection bounding boxes (color-coded by entity type)
    - Highlights hazard zones with semi-transparent overlays
    - Draws spatial relationship proximity lines
    - Adds severity color-coded hazard indicators
    """

    # Default font size for annotations
    DEFAULT_FONT_SIZE = 16
    
    # Line width for bounding boxes
    BBOX_LINE_WIDTH = 3
    
    # Hazard overlay transparency
    HAZARD_OVERLAY_ALPHA = 80  # 0-255 scale

    def __init__(self):
        """Initialize the visualizer."""
        self._font = None
        try:
            self._font = ImageFont.truetype("arial.ttf", self.DEFAULT_FONT_SIZE)
        except (IOError, OSError):
            # Fallback to default font
            self._font = ImageFont.load_default()

    def draw_detections(
        self,
        image: Image.Image,
        detection_result: DetectionResult,
        show_confidence: bool = True,
        show_labels: bool = True,
    ) -> Image.Image:
        """
        Draw YOLOv11n detection bounding boxes on the image.
        
        Args:
            image: Original PIL Image
            detection_result: Detection results with bounding boxes
            show_confidence: Whether to show confidence scores
            show_labels: Whether to show entity labels
            
        Returns:
            Annotated PIL Image with detection bounding boxes
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        for detection in detection_result.detections:
            color = self._get_detection_color(detection.class_name)
            
            # Draw bounding box
            bbox = detection.bbox_xyxy
            draw.rectangle(
                [bbox[0], bbox[1], bbox[2], bbox[3]],
                outline=color,
                width=self.BBOX_LINE_WIDTH,
            )
            
            # Draw label with confidence
            if show_labels:
                label_parts = [detection.entity_label]
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.0%}")
                label_text = " ".join(label_parts)
                
                # Calculate text position (above the bounding box)
                text_y = bbox[1] - self.DEFAULT_FONT_SIZE - 4
                if text_y < 0:
                    text_y = bbox[1] + 4  # Place below if at top of image
                
                # Draw text background
                text_bbox_size = draw.textbbox(
                    (bbox[0], text_y), label_text, font=self._font
                )
                draw.rectangle(
                    text_bbox_size,
                    fill=color,
                )
                draw.text(
                    (bbox[0], text_y),
                    label_text,
                    fill=(255, 255, 255),
                    font=self._font,
                )
        
        return annotated

    def draw_spatial_relations(
        self,
        image: Image.Image,
        detection_result: DetectionResult,
    ) -> Image.Image:
        """
        Draw spatial relationship lines between workers and hazard sources.
        
        Shows proximity lines between workers and nearby machinery/vehicles,
        highlighting close proximity relationships with warning colors.
        
        Args:
            image: PIL Image (possibly already annotated with detections)
            detection_result: Detection results with spatial relations
            
        Returns:
            Annotated PIL Image with spatial relationship lines
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        detections = detection_result.detections
        
        for relation in detection_result.spatial_relations:
            # Get the two entities involved
            if "worker_id" in relation and "source_id" in relation:
                idx1 = relation["worker_id"]
                idx2 = relation["source_id"]
            else:
                idx1 = relation.get("entity1_id", 0)
                idx2 = relation.get("entity2_id", 1)
            
            if idx1 < len(detections) and idx2 < len(detections):
                d1 = detections[idx1]
                d2 = detections[idx2]
                
                # Line color based on proximity
                is_close = relation.get("is_close_proximity", False)
                line_color = (255, 0, 0) if is_close else (200, 200, 200)  # Red for close, gray for far
                
                # Line thickness based on proximity
                line_width = 3 if is_close else 1
                
                # Draw line between centers
                draw.line(
                    [d1.bbox_center, d2.bbox_center],
                    fill=line_color,
                    width=line_width,
                )
                
                # Add distance label at midpoint
                if is_close:
                    mid_x = (d1.bbox_center[0] + d2.bbox_center[0]) / 2
                    mid_y = (d1.bbox_center[1] + d2.bbox_center[1]) / 2
                    dist_pct = relation.get("distance_pct", 0)
                    dist_label = f"CLOSE: {dist_pct:.1f}%"
                    draw.text(
                        (mid_x, mid_y),
                        dist_label,
                        fill=(255, 0, 0),
                        font=self._font,
                    )
        
        return annotated

    def draw_hazard_annotations(
        self,
        image: Image.Image,
        hazards: List[HazardAssessment],
        detection_result: DetectionResult,
    ) -> Image.Image:
        """
        Draw hazard zone annotations on the image.

        Adds semi-transparent overlays for hazard areas and severity indicators.
        Each hazard produces exactly one label, placed with collision avoidance
        so labels never stack on top of each other.
        """
        if not hazards:
            return image

        annotated = image.copy()

        # ── Semi-transparent zone overlays ──
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        for hazard in hazards:
            severity_hex = SEVERITY_COLORS.get(hazard.severity, SEVERITY_COLORS["medium"])
            severity_rgb = self._hex_to_rgb(severity_hex)
            relevant_detections = self._find_relevant_detections(
                hazard, detection_result.detections
            )
            for det in relevant_detections:
                expansion = 30
                zone_bbox = (
                    det.bbox_xyxy[0] - expansion,
                    det.bbox_xyxy[1] - expansion,
                    det.bbox_xyxy[2] + expansion,
                    det.bbox_xyxy[3] + expansion,
                )
                overlay_draw.rectangle(
                    zone_bbox,
                    fill=(*severity_rgb, self.HAZARD_OVERLAY_ALPHA),
                    outline=(*severity_rgb, 200),
                    width=2,
                )

        annotated = annotated.convert("RGBA")
        annotated = Image.alpha_composite(annotated, overlay)
        annotated = annotated.convert("RGB")

        # ── One label per hazard with collision avoidance ──
        draw = ImageDraw.Draw(annotated)
        img_w, img_h = annotated.size
        label_h = self.DEFAULT_FONT_SIZE + 8  # height of one label row
        used_y_positions: List[float] = []   # track occupied top-y values

        for hazard in hazards:
            severity_hex = SEVERITY_COLORS.get(hazard.severity, SEVERITY_COLORS["medium"])
            severity_rgb = self._hex_to_rgb(severity_hex)
            relevant_detections = self._find_relevant_detections(
                hazard, detection_result.detections
            )

            # Anchor: top-left of the first relevant detection
            if relevant_detections:
                det = relevant_detections[0]
                anchor_x = det.bbox_xyxy[0]
                preferred_y = det.bbox_xyxy[1] - label_h - 4
            else:
                anchor_x = 4
                preferred_y = 4

            # Clamp X so label stays within image
            anchor_x = max(2, min(anchor_x, img_w - 10))

            # Resolve Y collision: push down until clear of all previous labels
            label_y = max(2, preferred_y)
            for used_y in sorted(used_y_positions):
                if abs(label_y - used_y) < label_h:
                    label_y = used_y + label_h + 2
            label_y = min(label_y, img_h - label_h - 2)
            used_y_positions.append(label_y)

            label_text = f"⚠ {hazard.hazard_label} ({hazard.severity})"

            # Solid background pill for readability
            text_bbox = draw.textbbox((anchor_x, label_y), label_text, font=self._font)
            pad = 3
            draw.rectangle(
                (text_bbox[0] - pad, text_bbox[1] - pad,
                 text_bbox[2] + pad, text_bbox[3] + pad),
                fill=(20, 20, 20, 210) if hasattr(draw, 'rectangle') else (20, 20, 20),
            )
            draw.text(
                (anchor_x, label_y),
                label_text,
                fill=severity_rgb,
                font=self._font,
            )

        return annotated

    def create_full_annotation(
        self,
        image: Image.Image,
        detection_result: DetectionResult,
        hazards: List[HazardAssessment],
        show_detections: bool = True,
        show_relations: bool = True,
        show_hazard_zones: bool = True,
    ) -> Image.Image:
        """
        Create a fully annotated image with all visualization layers.
        
        Combines detection bounding boxes, spatial relations, and hazard zones.
        
        Args:
            image: Original PIL Image
            detection_result: Detection results
            hazards: List of parsed hazard assessments
            show_detections: Whether to show detection bounding boxes
            show_relations: Whether to show spatial relation lines
            show_hazard_zones: Whether to show hazard zone overlays
            
        Returns:
            Fully annotated PIL Image
        """
        annotated = image.copy()
        
        if show_detections and detection_result.detections:
            annotated = self.draw_detections(annotated, detection_result)
        
        if show_relations and detection_result.spatial_relations:
            annotated = self.draw_spatial_relations(annotated, detection_result)
        
        if show_hazard_zones and hazards:
            annotated = self.draw_hazard_annotations(annotated, hazards, detection_result)
        
        return annotated

    def _get_detection_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get the color for a detection class."""
        return DETECTION_COLORS.get(class_name, DETECTION_COLORS["default"])

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _find_relevant_detections(
        self,
        hazard: HazardAssessment,
        detections: List[Detection],
    ) -> List[Detection]:
        """
        Find detection results relevant to a specific hazard.
        
        Maps hazard types to relevant entity types for annotation.
        """
        relevant = []
        
        # Map hazard types to entity types
        hazard_entity_map = {
            "fall_from_height": ["worker"],
            "struck_by": ["worker", "vehicle", "machinery"],
            "caught_in_between": ["worker", "machinery"],
            "electrical": ["worker"],
            "excavation_trenching": ["worker"],
            "ppe_non_compliance": ["worker"],
            "unsafe_proximity": ["worker", "vehicle", "machinery"],
        }
        
        relevant_types = hazard_entity_map.get(hazard.hazard_type, ["worker"])
        
        for det in detections:
            if det.entity_type in relevant_types:
                relevant.append(det)
        
        # If no specific mapping, use all worker detections as default
        if not relevant:
            relevant = [d for d in detections if d.entity_type == "worker"]
        
        # If still no detections, use all detections
        if not relevant:
            relevant = list(detections)
        
        return relevant