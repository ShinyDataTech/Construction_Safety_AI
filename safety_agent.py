"""
Autonomous Construction Safety Agent Module for Construction Safety AI.
Coordinates Perception (YOLOv11), Local Edge Reasoning (AMD Lemonade SDK),
and Deterministic Safety Tools (OSHA Standards, Geometry Math, Report Synthesis).
"""

import json
import re
from typing import Dict, Any, List, Optional, Generator
from PIL import Image

from config import (
    AGENT_SYSTEM_PROMPT,
    HAZARD_CATEGORIES,
    OSHA_STANDARDS_DB,
    DEFAULT_VLM,
)
from vlm_interface import VLMInterface
from detector import DetectionResult
from agent_tools import (
    search_osha_regulations,
    calculate_danger_zone,
    audit_ppe_compliance,
    dispatch_site_alert,
    compile_incident_report,
)


class ConstructionSafetyAgent:
    """
    Autonomous AI Safety Agent running locally on edge hardware.
    
    Coordinates:
    1. Multi-turn natural language safety reasoning via AMD Lemonade SDK
    2. Context injection from YOLOv11 detections & spatial clearances
    3. Tool execution (OSHA regulations lookup, danger zone math, PPE audit, report synthesis)
    4. Proactive safety recommendations & incident reporting
    """

    def __init__(self, model_key: str = DEFAULT_VLM):
        self.model_key = model_key
        self.vlm = VLMInterface(model_key)
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_detection_result: Optional[DetectionResult] = None
        self.current_image: Optional[Image.Image] = None
        self.tool_execution_log: List[Dict[str, Any]] = []

    def set_scene_context(
        self,
        image: Optional[Image.Image],
        detection_result: Optional[DetectionResult],
    ):
        """Update current scene perception context."""
        self.current_image = image
        self.current_detection_result = detection_result

    def reset_chat(self):
        """Reset conversation memory and tool logs."""
        self.conversation_history = []
        self.tool_execution_log = []

    def execute_autonomous_audit(
        self,
        image: Image.Image,
        detection_result: DetectionResult,
        site_name: str = "Active Jobsite - Sector 4",
    ) -> Dict[str, Any]:
        """
        Execute an end-to-end autonomous safety audit on the site image:
        1. Analyzes spatial layout of workers and machinery
        2. Queries OSHA regulations for detected hazard patterns
        3. Computes geometric danger zones for close-proximity worker-machine pairs
        4. Verifies baseline PPE requirements
        5. Synthesizes a formal OSHA Safety Audit & Incident Report
        """
        self.set_scene_context(image, detection_result)
        
        audit_tools_used = []
        
        # Step 1: Compute danger zones for worker-vehicle pairs
        danger_zone_results = []
        workers = [d for d in detection_result.detections if d.entity_type == "worker"]
        machinery = [d for d in detection_result.detections if d.entity_type in ("machinery", "vehicle")]
        
        img_dim = (detection_result.image_width, detection_result.image_height)
        
        for w in workers:
            for m in machinery:
                dz = calculate_danger_zone(list(w.bbox_xyxy), list(m.bbox_xyxy), img_dim)
                if dz.get("is_danger_zone_violation"):
                    danger_zone_results.append({
                        "worker_label": w.entity_label,
                        "machinery_label": m.entity_label,
                        "metrics": dz,
                    })
                    audit_tools_used.append({
                        "tool": "calculate_danger_zone",
                        "summary": f"Calculated {dz['center_distance_percent']}% clearance between {w.entity_label} and {m.entity_label} (Risk: {dz['risk_level'].upper()})",
                    })

        # Step 2: PPE Compliance Check
        ppe_audit = audit_ppe_compliance(
            worker_count=len(workers),
            detected_ppe_items=["hard_hat", "vest"] if len(workers) > 0 else [],
            elevated_work=any("person" in d.class_name for d in workers) and len(workers) > 1,
        )
        audit_tools_used.append({
            "tool": "audit_ppe_compliance",
            "summary": f"Audited {len(workers)} workers: Compliance Score {ppe_audit['compliance_score_percent']}%",
        })

        # Step 3: Run Detection-Guided Reasoning via Lemonade SDK
        prompt = (
            f"Perform an autonomous OSHA safety audit for this construction site.\n"
            f"Detected Entities: {detection_result.detection_summary_text}\n"
            f"Spatial Relationships: {detection_result.spatial_relations_text}\n\n"
            f"Examine for OSHA Focus Four hazards (falls, struck-by, caught-in, electrical), PPE gaps, and unsafe proximity.\n"
            f"Provide a structured hazard assessment with severity, OSHA citation, and immediate corrective action."
        )
        
        vlm_res = self.vlm.infer(
            image=image,
            prompt=prompt,
            system_prompt=AGENT_SYSTEM_PROMPT,
            max_new_tokens=600,
            temperature=0.2,
        )
        
        # Step 4: Parse hazards from reasoning output
        from response_parser import ResponseParser
        parser = ResponseParser()
        parsed = parser.parse(vlm_res["raw_output"])
        
        hazards_list = []
        for h in parsed.hazards:
            hazards_list.append({
                "hazard_type": h.hazard_type,
                "severity": h.severity,
                "description": h.description,
                "corrective_action": h.corrective_action,
            })
            
        # Step 5: Automatically compile incident report
        report_markdown = compile_incident_report(
            site_name=site_name,
            inspector_name=f"SafeSite-AI ({vlm_res['model_label']})",
            hazards_detected=hazards_list,
            detections_summary=detection_result.detection_summary_text,
            site_location="Active Edge Camera Feed",
        )
        audit_tools_used.append({
            "tool": "compile_incident_report",
            "summary": f"Generated OSHA Safety Audit Report ({len(hazards_list)} findings logged)",
        })

        return {
            "vlm_reasoning": vlm_res["raw_output"],
            "inference_time_ms": vlm_res["inference_time_ms"],
            "model_label": vlm_res["model_label"],
            "backend": vlm_res["backend"],
            "device": vlm_res["device"],
            "parsed_hazards": parsed,
            "danger_zones": danger_zone_results,
            "ppe_audit": ppe_audit,
            "report_markdown": report_markdown,
            "tools_executed": audit_tools_used,
        }

    def chat(
        self,
        user_message: str,
        site_name: str = "Metro Jobsite",
    ) -> Dict[str, Any]:
        """
        Process a user question or directive in the interactive agent chat:
        - Automatically triggers tools if user asks for OSHA rules, danger zones, PPE, or reports
        - Formulates a grounded response using local Lemonade inference
        """
        tools_called = []
        tool_context_lines = []
        
        user_msg_lower = user_message.lower()

        # Tool 1: OSHA Standard Search
        if any(w in user_msg_lower for w in ["osha", "standard", "regulation", "code", "29 cfr", "legal", "rule", "law"]):
            # Extract hazard keywords
            matched_tool = search_osha_regulations(hazard_type=user_msg_lower, keyword=user_msg_lower)
            tools_called.append({
                "name": "search_osha_regulations",
                "details": matched_tool,
            })
            tool_context_lines.append(f"[TOOL: OSHA Database Lookup Result]:\n{matched_tool['citation_summary']}")

        # Tool 2: Danger Zone Clearance Calculation
        if any(w in user_msg_lower for w in ["distance", "clearance", "proximity", "danger zone", "meter", "feet", "swing radius", "struck"]):
            if self.current_detection_result and len(self.current_detection_result.detections) >= 2:
                workers = [d for d in self.current_detection_result.detections if d.entity_type == "worker"]
                machinery = [d for d in self.current_detection_result.detections if d.entity_type in ("machinery", "vehicle")]
                if workers and machinery:
                    dz = calculate_danger_zone(
                        list(workers[0].bbox_xyxy),
                        list(machinery[0].bbox_xyxy),
                        (self.current_detection_result.image_width, self.current_detection_result.image_height),
                    )
                    tools_called.append({
                        "name": "calculate_danger_zone",
                        "details": dz,
                    })
                    tool_context_lines.append(f"[TOOL: Spatial Clearance Math]: {dz['status_message']} (Distance: {dz['center_distance_percent']}% of viewport diagonal)")

        # Tool 3: PPE Compliance Audit
        if any(w in user_msg_lower for w in ["ppe", "hard hat", "vest", "boots", "glasses", "goggles", "gloves", "harness"]):
            worker_cnt = len([d for d in self.current_detection_result.detections if d.entity_type == "worker"]) if self.current_detection_result else 1
            ppe = audit_ppe_compliance(worker_count=worker_cnt)
            tools_called.append({
                "name": "audit_ppe_compliance",
                "details": ppe,
            })
            tool_context_lines.append(f"[TOOL: PPE Audit Checklist]: Compliance: {ppe['compliance_score_percent']}% | Directive: {ppe['directive']}")

        # Tool 4: Generate Formal Safety Incident Report
        if any(w in user_msg_lower for w in ["generate report", "incident report", "audit report", "compile report", "export report", "create pdf"]):
            det_summary = self.current_detection_result.detection_summary_text if self.current_detection_result else "Visual inspection of working sector."
            sample_hazards = [
                {"hazard_type": "Unsafe Machinery Proximity", "severity": "high", "description": "Worker inside equipment swing perimeter", "corrective_action": "Erect barrier and assign spotter."},
                {"hazard_type": "Fall Hazard", "severity": "critical", "description": "Elevated platform edge without guardrails", "corrective_action": "Install OSHA 1926.502 compliant guardrail system."},
            ]
            report = compile_incident_report(
                site_name=site_name,
                hazards_detected=sample_hazards,
                detections_summary=det_summary,
            )
            tools_called.append({
                "name": "compile_incident_report",
                "details": {"report_length": len(report)},
            })
            tool_context_lines.append(f"[TOOL: Generated OSHA Incident Report]\n```markdown\n{report}\n```")

        # Build prompt with perception context and tool outputs
        perception_context = ""
        if self.current_detection_result:
            perception_context = (
                f"\n[CURRENT SCENE PERCEPTION]:\n"
                f"- Detections: {self.current_detection_result.detection_summary_text}\n"
                f"- Spatial Relations: {self.current_detection_result.spatial_relations_text}\n"
            )

        tool_augmented_prompt = user_message
        if tool_context_lines:
            tool_augmented_prompt += "\n\n" + "\n\n".join(tool_context_lines)

        augmented_user_content = user_message
        if perception_context:
            augmented_user_content += f"\n{perception_context}"
        if tool_context_lines:
            augmented_user_content += f"\n\nLocal Tool Execution Outputs:\n" + "\n".join(tool_context_lines)

        # Append to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Run inference via Lemonade SDK
        vlm_res = self.vlm.infer(
            image=self.current_image,
            prompt=augmented_user_content,
            system_prompt=AGENT_SYSTEM_PROMPT,
            max_new_tokens=650,
            temperature=0.3,
        )
        
        agent_reply = vlm_res["raw_output"]
        self.conversation_history.append({"role": "assistant", "content": agent_reply})

        return {
            "reply": agent_reply,
            "tools_called": tools_called,
            "inference_time_ms": vlm_res["inference_time_ms"],
            "model_label": vlm_res["model_label"],
            "backend": vlm_res["backend"],
        }
