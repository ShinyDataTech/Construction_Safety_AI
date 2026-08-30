"""
Prompt Engineering Module for Construction Safety AI.
Converts YOLOv11n detection results into structured prompts for sVLM conditioning.

Stage 2: Prompt Engineering Module - detected entities embedded into structured prompts for sVLM reasoning.
"""

from typing import Optional

from config import BASELINE_PROMPT, DETECTION_GUIDED_PROMPT, HAZARD_CATEGORIES
from detector import DetectionResult


class PromptEngineer:
    """
    Constructs structured prompts for small VLM inference.
    
    Implements the prompt engineering module from the paper:
    - Detection-to-Text Encoding: converts YOLOv11n detections to natural language
    - Structured Prompt Construction: embeds detection context into VLM prompts
    - Supports baseline (no detection context) and detection-guided modes
    """

    def build_prompt(
        self,
        detection_result: Optional[DetectionResult] = None,
        mode: str = "detection_guided",
        custom_context: Optional[str] = None,
    ) -> str:
        """
        Build a structured prompt for the sVLM based on detection results.
        
        Args:
            detection_result: Detection results from YOLOv11n (None for baseline mode)
            mode: Prompt mode - "baseline" or "detection_guided"
            custom_context: Optional additional context to include
            
        Returns:
            Structured prompt string for VLM inference
        """
        if mode == "baseline" or detection_result is None:
            prompt = BASELINE_PROMPT
            if custom_context:
                prompt = prompt + "\n\nAdditional context:\n" + custom_context
            return prompt
        
        elif mode == "detection_guided":
            detection_summary = detection_result.detection_summary_text
            spatial_summary = detection_result.spatial_relations_text
            
            # If no detections were found, fall back to baseline with a note
            if not detection_result.detections:
                prompt = BASELINE_PROMPT
                prompt += "\n\nNote: No specific objects were detected by the object detection model. Please analyze the image directly for any safety hazards."
                return prompt
            
            prompt = DETECTION_GUIDED_PROMPT.format(
                detection_summary=detection_summary,
                spatial_relations_summary=spatial_summary,
            )
            
            if custom_context:
                prompt = prompt + "\n\nAdditional context:\n" + custom_context
            
            return prompt
        
        else:
            raise ValueError(f"Unknown prompt mode: {mode}. Use 'baseline' or 'detection_guided'.")

    def build_hazard_focus_prompt(
        self,
        detection_result: DetectionResult,
        focus_categories: list[str] = None,
    ) -> str:
        """
        Build a prompt that focuses on specific hazard categories.
        
        Useful when the user wants to focus analysis on particular hazard types
        based on the detection results.
        
        Args:
            detection_result: Detection results from YOLOv11n
            focus_categories: List of hazard category keys to focus on
            
        Returns:
            Focused prompt string for VLM inference
        """
        detection_summary = detection_result.detection_summary_text
        spatial_summary = detection_result.spatial_relations_text
        
        # Build category focus section
        if focus_categories:
            focus_text = "Focus specifically on these hazard types:\n"
            for cat_key in focus_categories:
                if cat_key in HAZARD_CATEGORIES:
                    cat = HAZARD_CATEGORIES[cat_key]
                    focus_text += f"- {cat['label']}: {cat['description']}\n"
        else:
            focus_text = ""
        
        prompt = f"""You are a construction safety expert. Analyze this construction site image for safety hazards.

Detected objects in the scene:
{detection_summary}

Spatial relationships:
{spatial_summary}

{focus_text}

Based on the detected objects and their spatial arrangement, identify any safety hazards present in this image.

For each hazard you identify, provide:
1. Hazard type (fall from height, struck-by, caught-in/between, electrical, excavation/trenching, PPE non-compliance, unsafe worker-machinery proximity)
2. Severity level (low, medium, high, critical)
3. Description of the hazardous situation
4. Recommended corrective action

If no hazards of the specified types are present, respond with "No relevant hazards detected."

Format your response as a structured list."""
        
        return prompt

    def build_comparison_prompts(
        self,
        detection_result: DetectionResult,
    ) -> dict[str, str]:
        """
        Build both baseline and detection-guided prompts for comparison analysis.
        
        Returns a dictionary with both prompts so the user can compare
        VLM outputs with and without detection context.
        
        Args:
            detection_result: Detection results from YOLOv11n
            
        Returns:
            Dictionary with "baseline" and "detection_guided" prompt keys
        """
        return {
            "baseline": self.build_prompt(detection_result, mode="baseline"),
            "detection_guided": self.build_prompt(detection_result, mode="detection_guided"),
        }

    def get_hazard_category_list(self) -> str:
        """Get a formatted list of all hazard categories for display."""
        lines = []
        for key, cat in HAZARD_CATEGORIES.items():
            lines.append(f"- **{cat['label']}**: {cat['description']} (default severity: {cat['severity_default']})")
        return "\n".join(lines)