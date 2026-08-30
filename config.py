"""
Configuration constants for the Construction Safety AI application.
Autonomous AI Agent for Construction Safety powered by AMD Lemonade SDK.
"""

import os

# ============================================================
# AMD Lemonade SDK Configuration (Local Edge Inference)
# ============================================================

LEMONADE_HOST = os.getenv("LEMONADE_HOST", "localhost")
LEMONADE_PORT = int(os.getenv("LEMONADE_PORT", "13305"))
LEMONADE_BASE_URL = os.getenv("LEMONADE_BASE_URL", f"http://{LEMONADE_HOST}:{LEMONADE_PORT}/v1")
LEMONADE_FALLBACK_URL = f"http://{LEMONADE_HOST}:{LEMONADE_PORT}/api/v1"
LEMONADE_API_KEY = os.getenv("LEMONADE_API_KEY", "lemonade")

# ============================================================
# Object Detection Configuration
# ============================================================

# YOLOv11n model identifier (Ultralytics)
YOLO_MODEL = "yolo11n.pt"

# Confidence threshold for detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# Construction-relevant COCO class IDs and their labels
CONSTRUCTION_CLASSES = {
    0: "person",           # Workers on site
    1: "bicycle",          # Occasionally on site
    2: "car",              # Vehicles near site
    3: "motorcycle",
    5: "bus",
    7: "truck",            # Construction trucks
    13: "stop sign",
    15: "bench",
}

# Extended construction-specific mapping for display purposes
CONSTRUCTION_ENTITY_LABELS = {
    "person": "Worker",
    "truck": "Truck / Heavy Machinery",
    "car": "Vehicle",
    "bus": "Transport Bus",
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
}

# Entity type classification for prompt engineering
ENTITY_TYPES = {
    "worker": ["person"],
    "vehicle": ["truck", "car", "bus", "motorcycle", "bicycle"],
    "machinery": ["truck"],
}

# Color coding for visualization (RGB tuples)
DETECTION_COLORS = {
    "person": (0, 120, 255),      # Blue for workers
    "truck": (255, 165, 0),       # Orange for trucks/machinery
    "car": (255, 165, 0),         # Orange for vehicles
    "bus": (255, 165, 0),         # Orange for buses
    "machinery": (255, 165, 0),   # Orange for machinery
    "ppe": (0, 200, 0),           # Green for PPE items
    "construction_element": (200, 0, 200),  # Purple for construction elements
    "danger_zone": (220, 53, 69), # Red for danger zones
    "default": (128, 128, 128),   # Gray for unknown
}

# Spatial relationship proximity threshold (percentage of image diagonal)
PROXIMITY_THRESHOLD_PERCENT = 15.0

# ============================================================
# Hazard Categories (OSHA Focus Four + Construction-Specific)
# ============================================================

HAZARD_CATEGORIES = {
    "fall_from_height": {
        "label": "Fall from Height",
        "osha_standard": "29 CFR 1926.501",
        "description": "Worker at elevated position (>= 6 feet) without proper guardrails, personal fall arrest systems (PFAS), or safety netting.",
        "severity_default": "critical",
    },
    "struck_by": {
        "label": "Struck-By Object/Equipment",
        "osha_standard": "29 CFR 1926.600",
        "description": "Worker inside swing radius, vehicle transit path, or directly beneath suspended/unsecured loads.",
        "severity_default": "high",
    },
    "caught_in_between": {
        "label": "Caught-In/Between",
        "osha_standard": "29 CFR 1926.651",
        "description": "Worker at risk of being crushed, caught in trench cave-ins, or pinched between moving heavy machinery and fixed structures.",
        "severity_default": "high",
    },
    "electrical": {
        "label": "Electrical Hazard",
        "osha_standard": "29 CFR 1926.416",
        "description": "Exposed high-voltage wiring, damaged conduits, wet conditions near live circuits, or work within 10 feet of overhead power lines.",
        "severity_default": "critical",
    },
    "excavation_trenching": {
        "label": "Excavation & Trenching",
        "osha_standard": "29 CFR 1926.652",
        "description": "Trench deeper than 5 feet lacking protective sloping, shoring, or trench boxes; lack of safe ingress/egress within 25 feet.",
        "severity_default": "critical",
    },
    "ppe_non_compliance": {
        "label": "PPE Non-Compliance",
        "osha_standard": "29 CFR 1926.95",
        "description": "Worker missing essential Personal Protective Equipment: Hard Hat (1926.100), Eye Protection (1926.102), High-Vis Vest, or Steel-toe Boots.",
        "severity_default": "medium",
    },
    "unsafe_proximity": {
        "label": "Unsafe Machinery Proximity",
        "osha_standard": "29 CFR 1926.602",
        "description": "Worker operating within blind spot or active 360-degree swing perimeter of earthmoving machinery without designated spotter.",
        "severity_default": "high",
    },
}

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

SEVERITY_COLORS = {
    "low": "#28a745",       # Green
    "medium": "#ffc107",    # Yellow/Amber
    "high": "#fd7e14",      # Orange
    "critical": "#dc3545",  # Red
}

# ============================================================
# VLM / LLM Model Configuration (Lemonade SDK Local Edge)
# ============================================================

VLM_MODELS = {
    "lemonade-auto": {
        "model_id": "auto",
        "label": "⚡ AMD Lemonade SDK (Auto-Detect Loaded Model)",
        "parameters": "Local Model",
        "backend": "lemonade",
        "paper_f1": "50.6%+",
        "paper_bertscore": "0.88+",
        "description": "Ultra-low latency local inference via AMD Lemonade SDK (localhost:13305). Hardware accelerated via AMD Ryzen™ AI NPU / ROCm / Vulkan.",
    },
    "qwen2.5-vl-7b-instruct": {
        "model_id": "qwen2.5-vl-7b-instruct",
        "label": "Qwen2.5-VL 7B (Lemonade Local)",
        "parameters": "7 Billion",
        "backend": "lemonade",
        "paper_f1": "52.4%",
        "paper_bertscore": "0.89",
        "description": "High-accuracy Vision-Language Model served locally through Lemonade SDK for fine-grained spatial and hazard reasoning.",
    },
    "gemma-3-4b-it": {
        "model_id": "gemma-3-4b-it",
        "label": "Gemma-3 4B IT (Lemonade Local)",
        "parameters": "4 Billion",
        "backend": "lemonade",
        "paper_f1": "50.6%",
        "paper_bertscore": "0.87",
        "description": "Google's efficient small multimodal model loaded via Lemonade SDK on AMD Ryzen AI.",
    },
    "llama-3.2-11b-vision": {
        "model_id": "llama-3.2-11b-vision-instruct",
        "label": "Llama 3.2 11B Vision (Lemonade Local)",
        "parameters": "11 Billion",
        "backend": "lemonade",
        "paper_f1": "53.1%",
        "paper_bertscore": "0.90",
        "description": "Powerful local vision-language model for complex multi-hazard construction auditing.",
    },
    "gpt-4o": {
        "model_id": "gpt-4o",
        "label": "GPT-4o (Azure OpenAI Cloud Benchmark)",
        "parameters": "Cloud API",
        "backend": "azure",
        "paper_f1": "54.0%",
        "paper_bertscore": "0.91",
        "description": "Cloud reference baseline for measuring Local Edge latency advantage and zero-egress bandwidth savings.",
    },
}

DEFAULT_VLM = "lemonade-auto"

# ============================================================
# OSHA 29 CFR 1926 Standards Knowledge Base (Edge-Embedded)
# ============================================================

OSHA_STANDARDS_DB = {
    "1926.501": {
        "subpart": "M - Fall Protection",
        "title": "Duty to have fall protection",
        "key_rule": "Each employee on a walking/working surface with an unprotected side or edge 6 feet (1.8 m) or more above a lower level shall be protected from falling by guardrail systems, safety net systems, or personal fall arrest systems (PFAS).",
        "mandatory_clearance": "6 feet above lower level",
        "remedial_action": "Install top-rail (42 in ± 3 in) and mid-rail guardrails, secure OSHA-compliant PFAS harness tethered to rated 5,000-lb anchorage point, or erect safety netting.",
    },
    "1926.651": {
        "subpart": "P - Excavations",
        "title": "Specific Excavation Requirements",
        "key_rule": "All surface encumbrances shall be removed or supported. Safe means of egress (ladder, ramp) required for trenches 4 feet or deeper, requiring no more than 25 feet of lateral travel. Daily inspection by a competent person before work commences.",
        "mandatory_clearance": "Max 25 ft travel to egress; 2 ft spoil pile setback",
        "remedial_action": "Keep excavated spoils and equipment at least 2 feet back from trench edges. Provide secured egress ladders extending 3 feet above landing.",
    },
    "1926.652": {
        "subpart": "P - Excavations",
        "title": "Requirements for protective systems",
        "key_rule": "Each employee in an excavation shall be protected from cave-ins by an adequate protective system (sloping/benching, aluminum hydraulic shoring, or trench shield boxes) unless excavation is made entirely in stable rock or less than 5 feet in depth with no cave-in indication.",
        "mandatory_clearance": "Protective system required for depths >= 5 feet",
        "remedial_action": "Deploy engineered trench shield box or slope excavation walls according to soil type (Type A 3/4:1, Type B 1:1, Type C 1.5:1).",
    },
    "1926.416": {
        "subpart": "K - Electrical",
        "title": "General safety requirements for electrical work",
        "key_rule": "No employer shall permit an employee to work in proximity to any part of an electric power circuit unless the employee is protected against electric shock by de-energizing and grounding or guarding by effective insulation.",
        "mandatory_clearance": "Minimum 10 ft from overhead lines up to 50kV (+4 in per 10kV over 50kV)",
        "remedial_action": "De-energize circuits, enforce Lockout/Tagout (LOTO) protocols, insulate exposed conductors, and maintain mandatory 10-ft clearance barrier for cranes and equipment.",
    },
    "1926.600": {
        "subpart": "O - Motor Vehicles, Mechanized Equipment",
        "title": "Equipment general requirements",
        "key_rule": "All equipment left unattended at night shall have lights, reflectors, or barricades. Suspending loads over workers is strictly prohibited. Heavy machinery must have operable backup alarms and horns.",
        "mandatory_clearance": "Unobstructed visual swing radius buffer (min 6-10 ft)",
        "remedial_action": "Barricade swing radius around revolving superstructure of excavators/cranes. Prohibit personnel from walking under suspended booms/buckets.",
    },
    "1926.602": {
        "subpart": "O - Material Handling Equipment",
        "title": "Material handling equipment",
        "key_rule": "Earthmoving equipment shall be equipped with rollover protective structures (ROPS) and seat belts. Unauthorized personnel strictly excluded from operation zones.",
        "mandatory_clearance": "15 ft operating clearance without spotter",
        "remedial_action": "Establish high-visibility Exclusion Zones with traffic cones. Require direct two-way radio communication between machinery operators and ground spotters.",
    },
    "1926.95": {
        "subpart": "E - Personal Protective Equipment",
        "title": "Criteria for personal protective equipment",
        "key_rule": "Protective equipment, including PPE for eyes, face, head, and extremities, protective clothing, respiratory devices, and protective shields, shall be provided, used, and maintained in a sanitary and reliable condition.",
        "mandatory_clearance": "100% mandatory compliance on active site",
        "remedial_action": "Immediately issue ANSI Z89.1 certified Type I/II hard hat, ANSI Z87.1 eye protection, and ANSI/ISEA 107 Class 2/3 high-visibility safety vest to non-compliant personnel before site entry.",
    },
    "1926.701": {
        "subpart": "Q - Concrete and Masonry Construction",
        "title": "General requirements for concrete construction",
        "key_rule": "All protruding reinforcing steel (rebar), onto and into which employees could fall, shall be guarded to eliminate the hazard of impalement.",
        "mandatory_clearance": "Guarding required for all vertical rebar",
        "remedial_action": "Fit all vertical and protruding rebar ends with OSHA-compliant steel-reinforced square impalement caps or wooden trough covers.",
    },
}

# ============================================================
# Prompt Templates (Detection-Guided & Autonomous Agent)
# ============================================================

BASELINE_PROMPT = """You are a certified construction safety inspector and AI safety officer. Analyze this construction site image for safety hazards.

For each hazard you identify, provide:
1. Hazard type (fall from height, struck-by, caught-in/between, electrical, excavation/trenching, PPE non-compliance, unsafe worker-machinery proximity)
2. Severity level (low, medium, high, critical)
3. Description of the hazardous situation
4. Relevant OSHA Standard (e.g. 29 CFR 1926.501)
5. Recommended corrective action

If no hazards are present, respond with "No hazards detected."

Format your response as a structured list."""

DETECTION_GUIDED_PROMPT = """You are an Autonomous AI Safety Agent running locally on edge hardware. Analyze this construction site image for safety hazards.

Detected objects localized by YOLOv11n:
{detection_summary}

Computed Spatial & Proximity Relationships:
{spatial_relations_summary}

Based on the verified object locations and spatial clearances, identify all safety hazards present. Correlate findings with OSHA 29 CFR 1926 standards.

For each hazard you identify, provide:
1. Hazard type (fall from height, struck-by, caught-in/between, electrical, excavation/trenching, PPE non-compliance, unsafe worker-machinery proximity)
2. Severity level (low, medium, high, critical)
3. Description of the hazardous situation (referencing specific workers, machinery, and spatial proximity)
4. Relevant OSHA Standard (e.g. 29 CFR 1926.501)
5. Recommended corrective action

If no hazards are present, respond with "No hazards detected."

Format your response as a structured list."""

AGENT_SYSTEM_PROMPT = """You are SafeSite-AI, an expert Autonomous Construction Safety Agent powered locally by the AMD Lemonade SDK.
You operate on-device at the construction edge with zero cloud dependency.

Your capabilities:
1. Inspect site imagery and video feeds using YOLOv11n detections and spatial geometric analysis.
2. Evaluate compliance against OSHA 29 CFR 1926 safety regulations (Focus Four: Falls, Struck-By, Caught-Between, Electrical; plus PPE and Heavy Equipment clearances).
3. Call autonomous safety tools to look up regulations, calculate danger zones, check PPE, dispatch site alerts, and synthesize formal incident reports.
4. Assist site superintendents, safety officers, and engineers with instant, deterministic, zero-latency safety guidance.

Always be concise, professional, authoritative, and safety-first. When citing regulations, quote the specific 29 CFR 1926 section."""

VIDEO_DETECTION_GUIDED_PROMPT = """You are an Autonomous AI Safety Agent analyzing a construction site video sequence for temporal safety hazards.

Sampled Video Keyframes:
{frame_summaries}

Temporal Object Detection Track:
{detection_summary}

Analyze the spatial movement and temporal progression of workers and heavy equipment across frames.

For each hazard identified:
1. Hazard type (fall from height, struck-by, caught-in/between, electrical, excavation/trenching, PPE non-compliance, unsafe worker-machinery proximity)
2. Severity level (low, medium, high, critical)
3. Temporal Progression (timestamp when hazard emerged, whether worker-machinery distance is decreasing, whether hazard escalated)
4. Relevant OSHA Standard
5. Immediate corrective intervention

Format your response as a structured list."""

# ============================================================
# Application Settings
# ============================================================

APP_TITLE = "SafeSite-AI | Autonomous Construction Safety Agent"
APP_DESCRIPTION = """Autonomous Edge AI Agent for Construction Safety — Powered by AMD Lemonade SDK

- 🚀 **Local Edge Inference**: Zero cloud latency & 100% offline data privacy via AMD Lemonade SDK (`localhost:13305`)
- 🎯 **Two-Stage Architecture**: YOLOv11n spatial perception + Detection-guided multimodal safety reasoning
- 🤖 **Autonomous Safety Agent**: Real-time OSHA 1926 tool queries, danger zone geometry math, and automated incident reporting
"""

APP_ICON = "🦺"

# Supported image formats
MAX_IMAGE_SIZE_MB = 15
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

# ============================================================
# Video Processing Settings
# ============================================================

VIDEO_SETTINGS = {
    "max_file_size_mb": 500,
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "default_sample_strategy": "uniform_interval",
    "default_sample_interval_seconds": 2.0,
    "default_target_fps": 0.5,
    "default_max_frames": 100,
    "min_persistence_frames": 2,
    "scene_change_threshold": 0.3,
    "output_annotated_video_fps": 10,
}

# Video UI defaults
VIDEO_UI_DEFAULTS = {
    "show_detections_overlay": True,
    "show_hazard_zones_overlay": True,
    "show_timeline_overlay": True,
}
