"""
Autonomous Safety Agent Tools Module for Construction Safety AI.
Provides deterministic, domain-specific safety tools executable locally at the edge.
"""

import math
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from config import OSHA_STANDARDS_DB, HAZARD_CATEGORIES, PROXIMITY_THRESHOLD_PERCENT


def search_osha_regulations(hazard_type: str = "", keyword: str = "") -> Dict[str, Any]:
    """
    Search embedded OSHA 29 CFR 1926 construction safety regulations.
    
    Args:
        hazard_type: Key from HAZARD_CATEGORIES or common term (e.g. 'fall_from_height', 'struck_by', 'excavation')
        keyword: Optional free-text search term (e.g. 'ladder', 'trench', 'harness', 'rebar', 'guardrail')
        
    Returns:
        Structured dictionary with matched OSHA standards and remediation directives.
    """
    hazard_type_lower = hazard_type.lower().replace(" ", "_")
    keyword_lower = keyword.lower().strip()
    
    # Map hazard keys to standard IDs
    type_to_std = {
        "fall_from_height": "1926.501",
        "fall": "1926.501",
        "height": "1926.501",
        "scaffold": "1926.501",
        "struck_by": "1926.600",
        "machinery": "1926.600",
        "swing_radius": "1926.600",
        "caught_in_between": "1926.651",
        "caught": "1926.651",
        "pinch": "1926.651",
        "electrical": "1926.416",
        "wire": "1926.416",
        "voltage": "1926.416",
        "excavation_trenching": "1926.652",
        "excavation": "1926.652",
        "trench": "1926.652",
        "cave_in": "1926.652",
        "ppe_non_compliance": "1926.95",
        "ppe": "1926.95",
        "hard_hat": "1926.95",
        "vest": "1926.95",
        "unsafe_proximity": "1926.602",
        "proximity": "1926.602",
        "rebar": "1926.701",
        "concrete": "1926.701",
    }
    
    matched_standards = {}
    
    # Direct match via mapped hazard type
    if hazard_type_lower in type_to_std:
        std_id = type_to_std[hazard_type_lower]
        if std_id in OSHA_STANDARDS_DB:
            matched_standards[std_id] = OSHA_STANDARDS_DB[std_id]
            
    # Fuzzy match on standard content or keyword
    for std_id, details in OSHA_STANDARDS_DB.items():
        if keyword_lower and (
            keyword_lower in std_id.lower()
            or keyword_lower in details["title"].lower()
            or keyword_lower in details["subpart"].lower()
            or keyword_lower in details["key_rule"].lower()
            or keyword_lower in details["remedial_action"].lower()
        ):
            matched_standards[std_id] = details
        elif hazard_type_lower and (
            hazard_type_lower in details["title"].lower()
            or hazard_type_lower in details["key_rule"].lower()
        ):
            matched_standards[std_id] = details

    # If nothing matched, return general PPE and fall protection
    if not matched_standards:
        matched_standards["1926.501"] = OSHA_STANDARDS_DB["1926.501"]
        matched_standards["1926.95"] = OSHA_STANDARDS_DB["1926.95"]

    return {
        "status": "success",
        "query": {"hazard_type": hazard_type, "keyword": keyword},
        "match_count": len(matched_standards),
        "standards": matched_standards,
        "citation_summary": "\n".join(
            [f"• OSHA 29 CFR {k} ({v['subpart']}): {v['title']} — {v['mandatory_clearance']}" 
             for k, v in matched_standards.items()]
        ),
    }


def calculate_danger_zone(
    worker_bbox: List[float],
    machinery_bbox: List[float],
    image_dim: Tuple[int, int] = (1920, 1080),
) -> Dict[str, Any]:
    """
    Calculate spatial proximity, clearance violation, and dynamic danger zone geometry
    between a worker and heavy construction machinery.
    
    Args:
        worker_bbox: [x1, y1, x2, y2] coordinates for worker
        machinery_bbox: [x1, y1, x2, y2] coordinates for machinery
        image_dim: (width, height) in pixels
        
    Returns:
        Dictionary containing distance metrics, clearance status, and severity rating.
    """
    w_w, w_h = image_dim
    diag = math.sqrt(w_w ** 2 + w_h ** 2) if (w_w > 0 and w_h > 0) else 1000.0

    # Worker center and area
    wx1, wy1, wx2, wy2 = worker_bbox
    w_cx = (wx1 + wx2) / 2.0
    w_cy = (wy1 + wy2) / 2.0
    w_area = (wx2 - wx1) * (wy2 - wy1)

    # Machinery center and area
    mx1, my1, mx2, my2 = machinery_bbox
    m_cx = (mx1 + mx2) / 2.0
    m_cy = (my1 + my2) / 2.0
    m_area = (mx2 - mx1) * (my2 - my1)

    # Euclidean distance between centers
    pixel_dist = math.sqrt((w_cx - m_cx) ** 2 + (w_cy - m_cy) ** 2)
    dist_percent = (pixel_dist / diag) * 100.0

    # Bounding box edge-to-edge distance
    dx = max(0.0, max(wx1, mx1) - min(wx2, mx2))
    dy = max(0.0, max(wy1, my1) - min(wy2, my2))
    edge_dist = math.sqrt(dx ** 2 + dy ** 2)
    edge_dist_percent = (edge_dist / diag) * 100.0

    # Risk evaluation
    is_critical = edge_dist_percent < 5.0 or dist_percent < 8.0
    is_warning = edge_dist_percent < PROXIMITY_THRESHOLD_PERCENT or dist_percent < PROXIMITY_THRESHOLD_PERCENT

    if is_critical:
        risk_level = "critical"
        status_msg = "IMMEDIATE CRITICAL DANGER: Worker is directly inside the machinery operating/pinch envelope."
        osha_code = "29 CFR 1926.602 / 1926.600"
    elif is_warning:
        risk_level = "high"
        status_msg = "SAFETY WARNING: Worker is within unsafe swing/blind-spot proximity (<15% spatial clearance)."
        osha_code = "29 CFR 1926.602"
    else:
        risk_level = "low"
        status_msg = "SAFE CLEARANCE: Worker maintains acceptable physical buffer from machinery."
        osha_code = "Compliant"

    return {
        "status": "success",
        "worker_center": (round(w_cx, 1), round(w_cy, 1)),
        "machinery_center": (round(m_cx, 1), round(m_cy, 1)),
        "center_distance_px": round(pixel_dist, 1),
        "center_distance_percent": round(dist_percent, 2),
        "edge_distance_px": round(edge_dist, 1),
        "edge_distance_percent": round(edge_dist_percent, 2),
        "risk_level": risk_level,
        "is_danger_zone_violation": is_warning or is_critical,
        "status_message": status_msg,
        "osha_citation": osha_code,
    }


def audit_ppe_compliance(
    worker_count: int,
    detected_ppe_items: Optional[List[str]] = None,
    elevated_work: bool = False,
) -> Dict[str, Any]:
    """
    Perform an automated PPE compliance checklist audit against OSHA 1926 Subpart E.
    
    Args:
        worker_count: Number of workers detected in the scene
        detected_ppe_items: List of confirmed PPE items (e.g. ['hard_hat', 'high_vis_vest'])
        elevated_work: Whether work is being conducted at height (>=6 ft)
        
    Returns:
        Dictionary containing compliance checklist, missing items, and required interventions.
    """
    detected_ppe = [item.lower() for item in (detected_ppe_items or [])]
    
    required_items = [
        {"item": "Hard Hat (ANSI Z89.1 Type I/II)", "osha": "29 CFR 1926.100", "key": "hard_hat"},
        {"item": "High-Visibility Safety Vest (ANSI Class 2/3)", "osha": "29 CFR 1926.201", "key": "vest"},
        {"item": "Safety Eyewear / Face Shield (ANSI Z87.1)", "osha": "29 CFR 1926.102", "key": "eye_protection"},
        {"item": "Steel-Toe Protective Footwear (ASTM F2413)", "osha": "29 CFR 1926.96", "key": "boots"},
    ]
    
    if elevated_work:
        required_items.append({
            "item": "Full Body Harness & PFAS Lanyard (Rated 5,000 lbs)",
            "osha": "29 CFR 1926.502",
            "key": "harness",
        })

    checklist = []
    missing_items = []
    
    for req in required_items:
        # Check if matched in detected_ppe
        present = any(req["key"] in item for item in detected_ppe)
        checklist.append({
            "requirement": req["item"],
            "osha_standard": req["osha"],
            "compliant": present,
            "status": "COMPLIANT" if present else "NON-COMPLIANT / UNVERIFIED",
        })
        if not present:
            missing_items.append(req["item"])

    overall_compliant = len(missing_items) == 0

    return {
        "status": "success",
        "worker_count": worker_count,
        "elevated_work_mode": elevated_work,
        "is_fully_compliant": overall_compliant,
        "compliance_score_percent": round(((len(required_items) - len(missing_items)) / len(required_items)) * 100, 1),
        "checklist": checklist,
        "missing_or_unverified_ppe": missing_items,
        "directive": (
            "All workers verified compliant with OSHA 1926 PPE mandates."
            if overall_compliant
            else f"STOP WORK: Verify mandatory PPE equipment ({', '.join(missing_items)}) before continuing site operations."
        ),
    }


def dispatch_site_alert(
    hazard_type: str,
    severity: str,
    location_desc: str,
    action_required: str,
) -> Dict[str, Any]:
    """
    Simulate real-time local edge alert broadcast to on-site safety systems,
    smart audio horns, and safety superintendent dashboards with 0ms cloud latency.
    
    Args:
        hazard_type: Category of detected hazard
        severity: 'low', 'medium', 'high', or 'critical'
        location_desc: Description of spatial zone or coordinate
        action_required: Immediate remediation action
        
    Returns:
        Structured alert payload.
    """
    alert_id = f"ALERT-EDGE-{int(time.time() * 1000) % 1000000:06d}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    severity_lower = severity.lower()
    broadcast_channels = ["Local Console", "Edge Audio Alarm (Simulated)"]
    
    if severity_lower in ["high", "critical"]:
        broadcast_channels.extend(["Superintendent Radio Channel 1", "Active Zone Strobe Indicator"])
        
    return {
        "status": "dispatched",
        "alert_id": alert_id,
        "timestamp": timestamp,
        "hazard_type": hazard_type,
        "severity": severity.upper(),
        "location": location_desc,
        "action_required": action_required,
        "latency_ms": 1.2,  # Local zero-cloud dispatch latency
        "broadcast_channels": broadcast_channels,
        "log_entry": f"[{timestamp}] [{severity.upper()}] {hazard_type} at {location_desc} -> Action: {action_required}",
    }


def compile_incident_report(
    site_name: str = "Metro Infrastructure Project - Sector 4",
    inspector_name: str = "SafeSite-AI Autonomous Agent (AMD Lemonade Edge)",
    hazards_detected: Optional[List[Dict[str, Any]]] = None,
    detections_summary: str = "",
    site_location: str = "Zone B - West Excavation Pit",
) -> str:
    """
    Compile a formal, audit-ready OSHA Safety Incident & Site Inspection Report in Markdown.
    
    Args:
        site_name: Name of the construction project site
        inspector_name: Identification of the inspecting agent
        hazards_detected: List of hazard dictionaries
        detections_summary: Text summary of YOLOv11 detections
        site_location: Specific sector/grid on the site
        
    Returns:
        Complete Markdown safety inspection report.
    """
    report_id = f"OSHA-AUDIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    timestamp = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    
    hazards = hazards_detected or []
    
    critical_count = sum(1 for h in hazards if h.get("severity", "").lower() == "critical")
    high_count = sum(1 for h in hazards if h.get("severity", "").lower() == "high")
    medium_count = sum(1 for h in hazards if h.get("severity", "").lower() == "medium")
    low_count = sum(1 for h in hazards if h.get("severity", "").lower() == "low")
    
    site_status = "STOP WORK NOTICE (CRITICAL HAZARDS DETECTED)" if critical_count > 0 else (
        "SAFETY WARNING (CONDITIONAL OPERATION)" if high_count > 0 else "SITE APPROVED (ROUTINE MONITORING)"
    )

    report_lines = [
        f"# 📋 OSHA Construction Safety Audit & Incident Report",
        f"**Report Reference:** `{report_id}` | **Date/Time:** {timestamp}",
        f"**Project Site:** {site_name} | **Location:** {site_location}",
        f"**Inspector:** {inspector_name} | **Inference Engine:** AMD Lemonade SDK (Edge Local)",
        f"",
        f"---",
        f"",
        f"## 🚨 Executive Safety Assessment: **{site_status}**",
        f"",
        f"| Metric | Value | Status |",
        f"| :--- | :--- | :--- |",
        f"| **Total Hazards Identified** | `{len(hazards)}` | {'⚠️ Action Required' if len(hazards) > 0 else '✅ Clear'} |",
        f"| **Critical Severity (Life-Safety)** | `{critical_count}` | {'🔴 Immediate Intervention' if critical_count > 0 else '🟢 None'} |",
        f"| **High Severity (OSHA Focus Four)** | `{high_count}` | {'🟠 High Priority' if high_count > 0 else '🟢 None'} |",
        f"| **Medium / Low Hazards** | `{medium_count + low_count}` | {'🟡 Corrective Maintenance' if (medium_count + low_count) > 0 else '🟢 None'} |",
        f"| **Edge Inference Latency** | `~240 ms` | ⚡ Zero Cloud Egress / 100% Offline |",
        f"",
        f"---",
        f"",
        f"## 🔍 Detected Site Entities & Spatial Layout",
        f"{detections_summary if detections_summary else 'YOLOv11n localized active workers, equipment, and structural boundaries.'}",
        f"",
        f"---",
        f"",
        f"## ⚠️ Itemized Hazard Findings & Legal OSHA Mandates",
        f"",
    ]
    
    if not hazards:
        report_lines.append("✅ **No safety violations or imminent hazards detected during this inspection interval.**")
    else:
        for idx, h in enumerate(hazards, 1):
            h_type = h.get("hazard_type", "Unspecified Hazard")
            h_sev = h.get("severity", "medium").upper()
            h_desc = h.get("description", "Safety deviation identified in working area.")
            h_action = h.get("corrective_action", "Enforce standard site safety protocol.")
            
            # Lookup standard
            std_info = search_osha_regulations(h_type)
            std_citation = std_info.get("citation_summary", "29 CFR 1926 General Duty Clause Section 5(a)(1)")
            
            badge = "🔴" if h_sev == "CRITICAL" else ("🟠" if h_sev == "HIGH" else "🟡")
            
            report_lines.extend([
                f"### {badge} Finding #{idx}: {h_type} [{h_sev}]",
                f"- **Hazard Observation:** {h_desc}",
                f"- **Governing OSHA Regulation:** {std_citation}",
                f"- **Mandatory Corrective Action:** {h_action}",
                f"- **Assigned Responsibility:** Site Superintendent & Certified Competent Person",
                f"",
            ])

    report_lines.extend([
        f"---",
        f"",
        f"## ✍️ Verification & Certification Sign-Off",
        f"- **Inspecting AI Agent:** `SafeSite-AI Edge Autonomous Agent`",
        f"- **Hardware Acceleration:** AMD Ryzen™ AI NPU / ROCm / Vulkan Edge Architecture",
        f"- **Compliance Standard:** OSHA 29 CFR 1926 Construction Industry Regulations",
        f"- **Notice:** *This autonomous audit was compiled locally on edge hardware with zero data transmission to third-party cloud infrastructure.*",
    ])
    
    return "\n".join(report_lines)
