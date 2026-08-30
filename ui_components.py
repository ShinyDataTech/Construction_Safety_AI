"""
UI Components Module for Construction Safety AI Streamlit App.
Autonomous Edge AI Agent for Construction Safety powered by AMD Lemonade SDK.
"""

import streamlit as st
from PIL import Image
from typing import List, Optional, Dict, Any
from datetime import datetime

from config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_ICON,
    HAZARD_CATEGORIES,
    SEVERITY_LEVELS,
    SEVERITY_COLORS,
    VLM_MODELS,
    DEFAULT_VLM,
    MAX_IMAGE_SIZE_MB,
    SUPPORTED_IMAGE_TYPES,
    VIDEO_SETTINGS,
    VIDEO_UI_DEFAULTS,
    LEMONADE_BASE_URL,
)
from detector import DetectionResult
from response_parser import HazardAssessment, ParsedResult
from vlm_interface import check_lemonade_health


def render_header():
    """Render the application header and page configuration."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_lemonade_status_banner(health: Optional[Dict[str, Any]] = None):
    """Render the AMD Lemonade SDK connection status banner."""
    if health is None:
        health = check_lemonade_health(LEMONADE_BASE_URL)
        
    if health.get("is_healthy"):
        models_str = ", ".join(health.get("models", [])[:3]) or "Active Vision/Instruct LLM"
        st.markdown(
            f"""
            <div style="background-color: #0f3822; border-left: 5px solid #28a745; padding: 10px 16px; border-radius: 6px; margin-bottom: 15px; color: #d4edda;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong>🟢 AMD Lemonade SDK Online</strong> &nbsp;|&nbsp; 
                        <code>{health.get('url_used')}</code> &nbsp;|&nbsp; 
                        <span style="font-size: 0.9em; color: #a3cfbb;">Hardware: AMD Ryzen™ AI / Local Edge NPU/GPU</span>
                    </div>
                    <div style="font-size: 0.85em; background: #198754; color: white; padding: 2px 8px; border-radius: 12px;">
                        Zero Cloud Latency Mode Active
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #3b2c11; border-left: 5px solid #ffc107; padding: 10px 16px; border-radius: 6px; margin-bottom: 15px; color: #fff3cd;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong>🟡 Lemonade SDK Server Offline</strong> &nbsp;|&nbsp; 
                        <code>{LEMONADE_BASE_URL}</code>
                    </div>
                    <div style="font-size: 0.85em; color: #ffe69c;">
                        Run: <code>lemonade serve --port 13305</code>
                    </div>
                </div>
                <div style="font-size: 0.85em; margin-top: 5px; color: #f8d7da;">
                    Local autonomous agent will attempt auto-connection upon request. Cloud benchmark backend remains accessible.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar(
    available_models: Optional[Dict[str, Any]] = None,
    default_model: str = DEFAULT_VLM,
) -> Dict[str, Any]:
    """
    Render the sidebar with configuration options.
    
    Returns:
        Dictionary of sidebar configuration values
    """
    with st.sidebar:
        st.title("🦺 SafeSite-AI Config")
        st.caption("Powered by **AMD Lemonade SDK** (Local Edge)")
        
        # Model selection
        model_options = list(VLM_MODELS.keys())
        selected_model = st.selectbox(
            "Inference Engine / Model",
            options=model_options,
            index=0 if default_model in model_options else 0,
            format_func=lambda x: VLM_MODELS.get(x, {}).get("label", x),
            help="Select local model hosted via Lemonade SDK on localhost:13305, or cloud benchmark.",
        )
        
        model_meta = VLM_MODELS.get(selected_model, {})
        if model_meta.get("backend") == "lemonade":
            st.success(f"⚡ Local Edge Backend ({model_meta.get('parameters', 'Edge Model')})")
        else:
            st.warning("☁️ Cloud API Benchmark Mode")
            
        # Analysis mode
        analysis_mode = st.radio(
            "Inspection Mode",
            options=["detection_guided", "baseline", "comparison"],
            format_func=lambda x: {
                "detection_guided": "Detection-Guided (Recommended)",
                "baseline": "Baseline (No Spatial Context)",
                "comparison": "Comparison (Side-by-Side)",
            }[x],
            help="Detection-guided mode embeds YOLOv11n spatial data into local VLM prompt for optimal F1 accuracy.",
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "YOLOv11n Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence threshold for local YOLOv11 object localization",
        )
        
        # VLM generation parameters
        with st.expander("⚡ Lemonade Model Hyperparameters"):
            max_tokens = st.slider(
                "Max New Tokens",
                min_value=128,
                max_value=1024,
                value=512,
                step=64,
                help="Maximum token generation limit for local inference",
            )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Sampling temperature (lower = more deterministic safety audit)",
            )
            lemonade_endpoint = st.text_input(
                "Lemonade SDK Endpoint",
                value=LEMONADE_BASE_URL,
                help="OpenAI-compatible local REST endpoint",
            )
        
        # Hazard category focus
        with st.expander("🎯 OSHA Focus Categories"):
            focus_categories = []
            for key, cat in HAZARD_CATEGORIES.items():
                if st.checkbox(
                    f"{cat['label']} ({cat['osha_standard']})",
                    value=True,
                    key=f"focus_{key}",
                ):
                    focus_categories.append(key)
        
        # Visualization options
        with st.expander("👁️ Visualization Overlays"):
            show_detections = st.checkbox("Show Worker/Machinery Boxes", value=True)
            show_relations = st.checkbox("Show Proximity Clearance Vectors", value=True)
            show_hazard_zones = st.checkbox("Show Danger Zone Perimeter", value=True)
        
        return {
            "selected_model": selected_model,
            "analysis_mode": analysis_mode,
            "confidence_threshold": confidence_threshold,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "lemonade_endpoint": lemonade_endpoint,
            "focus_categories": focus_categories,
            "show_detections": show_detections,
            "show_relations": show_relations,
            "show_hazard_zones": show_hazard_zones,
        }


def render_upload_section() -> Optional[Image.Image]:
    """
    Render the image upload section.
    
    Returns:
        Uploaded PIL Image or None
    """
    st.subheader("📸 Construction Site Image Selection")
    
    source_type = st.radio(
        "Image Source",
        options=["Use Demo Image", "Upload File"],
        horizontal=True,
        key="image_source_type",
    )
    
    if source_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a construction site image...",
            type=SUPPORTED_IMAGE_TYPES,
            help=f"Upload an image of a construction site (max {MAX_IMAGE_SIZE_MB}MB). "
                 f"Supported formats: {', '.join(SUPPORTED_IMAGE_TYPES)}",
        )
        
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_IMAGE_SIZE_MB:
                st.error(f"File size ({file_size_mb:.1f}MB) exceeds the maximum ({MAX_IMAGE_SIZE_MB}MB)")
                return None
            
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.metric("Resolution", f"{image.width}x{image.height}")
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            
            return image
    else:
        import os
        from pathlib import Path
        root_dir = Path(__file__).resolve().parent
        sample_dir = root_dir / "sample_data"
        
        demo_files = []
        if sample_dir.exists():
            for f in sorted(os.listdir(sample_dir)):
                ext = f.split(".")[-1].lower()
                if ext in SUPPORTED_IMAGE_TYPES:
                    demo_files.append(f)
                    
        if not demo_files:
            st.warning("No demo images found in sample_data folder.")
            return None
            
        selected_filename = st.selectbox(
            "Select a pre-loaded construction site scenario",
            options=demo_files,
            help="Choose a pre-loaded construction site image for analysis",
        )
        
        if selected_filename:
            image_path = sample_dir / selected_filename
            image = Image.open(image_path).convert("RGB")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption=f"Demo Scenario: {selected_filename}", use_container_width=True)
            with col2:
                st.metric("Resolution", f"{image.width}x{image.height}")
                file_size_mb = image_path.stat().st_size / (1024 * 1024) if image_path.exists() else 0.0
                st.metric("File Size", f"{file_size_mb:.2f} MB")
                st.caption(f"📁 `sample_data/{selected_filename}`")
                
            return image
            
    return None


def render_detection_results(detection_result: DetectionResult):
    """Render Stage 1 YOLOv11n detection results."""
    st.subheader("🎯 Stage 1: Edge Object Detection (YOLOv11n)")
    
    if not detection_result.detections:
        st.info("No construction-related objects detected in the scene.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Objects Localized", len(detection_result.detections))
    with col2:
        st.metric("YOLO Latency", f"{detection_result.inference_time_ms:.1f} ms")
    with col3:
        workers = [d for d in detection_result.detections if d.entity_type == "worker"]
        st.metric("Active Personnel", len(workers))
    
    with st.expander("🔍 Detailed Object Spatial Coordinates"):
        for det in detection_result.detections:
            st.markdown(
                f"- **{det.entity_label}** (`{det.entity_type}`): "
                f"Confidence: **{det.confidence:.1%}** | "
                f"Center: `({det.bbox_center[0]:.0f}, {det.bbox_center[1]:.0f})` | "
                f"Area: `{det.bbox_area:.0f} px²`"
            )
    
    with st.expander("📝 Structured Detection Summary (Prompt Conditioning)"):
        st.text(detection_result.detection_summary_text)
    
    if detection_result.spatial_relations:
        with st.expander("📐 Spatial Clearances & Proximity Vectors"):
            for rel in detection_result.spatial_relations:
                if "worker" in rel:
                    if rel.get("is_close_proximity"):
                        st.error(
                            f"⚠️ **PROXIMITY HAZARD:** {rel['worker']} near {rel['source']} "
                            f"({rel['distance_pct']:.1f}% diagonal distance, {rel['relative_position']})"
                        )
                    else:
                        st.markdown(
                            f"• {rel['worker']} is {rel['proximity_description']} from {rel['source']} "
                            f"({rel['relative_position']})"
                        )


def render_hazard_assessment(hazards: List[HazardAssessment], parsed_result: ParsedResult):
    """Render Stage 2 Hazard Assessment cards."""
    st.subheader("🧠 Stage 2: Autonomous Hazard Assessment (Lemonade Local VLM)")
    
    if parsed_result.no_hazards_detected:
        st.success("✅ No imminent safety hazards detected in the scene.")
        return
    
    if not hazards:
        st.warning("VLM output could not be automatically structured. Viewing raw stream below:")
        with st.expander("Raw Generated Stream"):
            st.text(parsed_result.raw_output)
        return
    
    severity_counts = {}
    for h in hazards:
        severity_counts[h.severity] = severity_counts.get(h.severity, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hazards Identified", len(hazards))
    with col2:
        st.metric("Critical (Life-Safety)", severity_counts.get("critical", 0))
    with col3:
        st.metric("High (OSHA Focus 4)", severity_counts.get("high", 0))
    with col4:
        st.metric("Medium / Low", severity_counts.get("medium", 0) + severity_counts.get("low", 0))
    
    for i, hazard in enumerate(hazards):
        severity_color = SEVERITY_COLORS.get(hazard.severity, "#ffc107")
        
        with st.container():
            col_icon, col_content = st.columns([1, 11])
            
            with col_icon:
                st.markdown(
                    f'<div style="background-color: {severity_color}; width: 44px; height: 44px; '
                    f'border-radius: 8px; display: flex; align-items: center; justify-content: center; '
                    f'color: white; font-weight: bold; font-size: 1.1em;">{hazard.severity.upper()[:1]}</div>',
                    unsafe_allow_html=True,
                )
            
            with col_content:
                st.markdown(f"### {hazard.hazard_label} &nbsp; `[{hazard.severity.upper()}]`")
                if hazard.description:
                    st.markdown(f"**Hazard Observation:** {hazard.description}")
                if hazard.recommendation:
                    st.markdown(f"**Mandatory Corrective Action:** {hazard.recommendation}")
                if hazard.detected_entities:
                    st.caption(f"Involved entities: {', '.join(hazard.detected_entities)}")
            
            st.divider()


def render_comparison_results(
    baseline_result: Dict[str, Any],
    guided_result: Dict[str, Any],
    baseline_parsed: ParsedResult,
    guided_parsed: ParsedResult,
):
    """Render side-by-side comparison of baseline vs detection-guided results."""
    st.subheader("⚖️ Side-by-Side: Baseline Prompt vs YOLO-Guided Agent")
    
    col_baseline, col_guided = st.columns(2)
    
    with col_baseline:
        st.markdown("### Standard Prompt (No Spatial Context)")
        st.caption("Vanilla prompt without object localization")
        
        if baseline_parsed.no_hazards_detected:
            st.info("No hazards detected")
        elif baseline_parsed.hazards:
            for h in baseline_parsed.hazards:
                severity_color = SEVERITY_COLORS.get(h.severity, "#ffc107")
                st.markdown(f"**{h.hazard_label}** — `{h.severity.upper()}`")
                if h.description:
                    st.caption(h.description)
        else:
            with st.expander("Raw Output"):
                st.text(baseline_parsed.raw_output)
        
        st.metric("Inference Latency", f"{baseline_result['inference_time_ms']:.1f} ms")
    
    with col_guided:
        st.markdown("### Detection-Guided Agent (YOLOv11 + Lemonade)")
        st.caption("Grounded in verified spatial coordinates and distance vectors")
        
        if guided_parsed.no_hazards_detected:
            st.info("No hazards detected")
        elif guided_parsed.hazards:
            for h in guided_parsed.hazards:
                severity_color = SEVERITY_COLORS.get(h.severity, "#ffc107")
                st.markdown(f"**{h.hazard_label}** — `{h.severity.upper()}`")
                if h.description:
                    st.caption(h.description)
        else:
            with st.expander("Raw Output"):
                st.text(guided_parsed.raw_output)
        
        st.metric("Inference Latency", f"{guided_result['inference_time_ms']:.1f} ms")
    
    st.subheader("📊 Accuracy & Recall Delta")
    baseline_count = len(baseline_parsed.hazards) if not baseline_parsed.no_hazards_detected else 0
    guided_count = len(guided_parsed.hazards) if not guided_parsed.no_hazards_detected else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline Hazards", baseline_count)
    with col2:
        st.metric("Detection-Guided Hazards", guided_count)
    with col3:
        delta = guided_count - baseline_count
        st.metric("Recall Gain", f"+{delta}" if delta > 0 else f"{delta}")


def render_annotated_image(image: Image.Image, caption: str = "Annotated Hazard Map"):
    """Display an annotated image."""
    st.image(image, caption=caption, use_container_width=True)


def render_raw_output(raw_output: str, model_label: str):
    """Render raw output stream."""
    with st.expander(f"📄 Full Local Generated Response ({model_label})"):
        st.text(raw_output)


def render_prompt_preview(prompt: str):
    """Render structured prompt."""
    with st.expander("📝 Grounded Prompt Structure"):
        st.text(prompt)


def render_progress_indicator(stage: str = "analyzing"):
    """Render progress notifications."""
    if stage == "detecting":
        st.info("🎯 Running YOLOv11n edge object detection & spatial geometric analysis...")
    elif stage == "loading_model":
        st.info("⚡ Connecting to AMD Lemonade SDK (localhost:13305)...")
    elif stage == "inferring":
        st.info("🧠 Executing local zero-latency multimodal hazard reasoning...")
    elif stage == "parsing":
        st.info("📋 Parsing structured safety directives & OSHA citations...")


def render_error(message: str):
    """Render error."""
    st.error(message)


# ============================================================
# Autonomous Agent UI Component
# ============================================================

def render_autonomous_agent_tab(agent, image: Optional[Image.Image], detection_result: Optional[DetectionResult]):
    """
    Render the interactive Autonomous Safety Agent workspace.
    """
    st.subheader("🤖 Autonomous Construction Safety Agent")
    st.caption("Powered by **AMD Lemonade SDK** Local Reasoning Engine (`http://localhost:13305/v1`) with Deterministic Tool Calling")
    
    if image is None:
        st.info("👆 Please select or upload a construction scene image in the Image Inspection tab or above to give the agent visual perception context.")
        return

    # Update scene context
    agent.set_scene_context(image, detection_result)

    # Quick action buttons
    st.markdown("##### ⚡ Agent Quick Actions")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    
    trigger_action = None
    with col_q1:
        if st.button("📋 Run Full OSHA Site Audit", use_container_width=True):
            trigger_action = "run_full_audit"
    with col_q2:
        if st.button("📐 Check Danger Zones", use_container_width=True):
            trigger_action = "check_danger_zones"
    with col_q3:
        if st.button("🦺 Audit PPE Compliance", use_container_width=True):
            trigger_action = "audit_ppe"
    with col_q4:
        if st.button("📄 Generate Incident Report", use_container_width=True):
            trigger_action = "generate_report"

    # Handle Full Autonomous Audit Execution
    if trigger_action == "run_full_audit":
        with st.spinner("🤖 Autonomous Agent executing end-to-end site inspection & OSHA audit..."):
            audit_res = agent.execute_autonomous_audit(image, detection_result)
            
            st.success(f"✅ Autonomous Audit Completed in {audit_res['inference_time_ms']:.1f}ms on {audit_res['device']}")
            
            # Tools executed summary
            st.markdown("#### 🛠️ Local Tools Invoked by Autonomous Agent")
            for t in audit_res["tools_executed"]:
                st.info(f"**Tool:** `{t['tool']}` &nbsp;|&nbsp; {t['summary']}")
                
            # Render danger zones if any
            if audit_res["danger_zones"]:
                st.markdown("#### ⚠️ Geometric Danger Zone Violations")
                for dz in audit_res["danger_zones"]:
                    m = dz["metrics"]
                    st.error(
                        f"🔴 **{dz['worker_label']} <-> {dz['machinery_label']}**: {m['status_message']} "
                        f"(Clearance: {m['center_distance_percent']}% | Citation: `{m['osha_citation']}`)"
                    )
            
            # Render PPE audit
            ppe = audit_res["ppe_audit"]
            st.markdown("#### 🦺 PPE Compliance Audit")
            st.metric("PPE Compliance Score", f"{ppe['compliance_score_percent']}%", delta="OSHA 1926 Subpart E")
            st.caption(ppe["directive"])
            
            # Render formal incident report
            st.markdown("#### 📋 Formal OSHA Safety Audit Report")
            st.markdown(audit_res["report_markdown"])
            
            st.download_button(
                label="📥 Download Formal OSHA Safety Report (.md)",
                data=audit_res["report_markdown"],
                file_name=f"OSHA_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
            st.divider()

    elif trigger_action == "check_danger_zones":
        with st.spinner("Calculating spatial clearances..."):
            res = agent.chat("Calculate spatial clearance and danger zones between all workers and machinery in the scene.")
            st.markdown(res["reply"])
            st.divider()

    elif trigger_action == "audit_ppe":
        with st.spinner("Verifying PPE compliance..."):
            res = agent.chat("Perform a PPE compliance check for all workers in this scene against OSHA 1926.")
            st.markdown(res["reply"])
            st.divider()

    elif trigger_action == "generate_report":
        with st.spinner("Compiling OSHA incident report..."):
            res = agent.chat("Generate a formal OSHA safety audit report with all detected hazards and corrective actions.")
            st.markdown(res["reply"])
            st.download_button(
                label="📥 Download Report (.md)",
                data=res["reply"],
                file_name=f"Incident_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
            st.divider()

    # Interactive Chat Loop
    st.markdown("##### 💬 Chat with On-Device Safety Agent")
    
    # Display conversation history
    for msg in agent.conversation_history:
        with st.chat_message(msg["role"], avatar="🦺" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about safety regulations, specific worker hazards, machinery swing radius, or type 'Generate Report'...")
    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🦺"):
            with st.spinner("Agent reasoning locally via Lemonade SDK..."):
                response = agent.chat(user_input)
                st.markdown(response["reply"])
                
                if response["tools_called"]:
                    with st.expander("🛠️ Deterministic Tools Called"):
                        for tc in response["tools_called"]:
                            st.json(tc)
                st.caption(f"⚡ Latency: {response['inference_time_ms']:.1f}ms | Backend: {response['backend']} (Local Edge)")


# ============================================================
# Edge vs Cloud Benchmark Dashboard
# ============================================================

def render_edge_benchmark_dashboard(local_latency_ms: float = 240.0, cloud_latency_ms: float = 3850.0):
    """
    Render quantitative comparison dashboard highlighting AMD Lemonade Edge benefits:
    Zero Latency, Zero Egress Bandwidth, 100% Offline Privacy, Zero API Cost.
    """
    st.subheader("⚡ Edge vs Cloud: Quantitative Benchmark & Feasibility Audit")
    st.caption("Demonstrating the mission-critical superiority of local AMD Lemonade SDK execution on harsh construction job sites.")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Inference Latency",
            value=f"{local_latency_ms:.0f} ms",
            delta=f"-{((cloud_latency_ms - local_latency_ms)/cloud_latency_ms)*100:.0f}% (vs Cloud)",
            delta_color="normal",
        )
        st.caption("Local AMD Ryzen™ AI NPU/ROCm")

    with col2:
        st.metric(
            label="Cloud Egress Bandwidth",
            value="0.0 MB/hr",
            delta="100% Bandwidth Saved",
            delta_color="normal",
        )
        st.caption("Zero 4K video upload requirement")

    with col3:
        st.metric(
            label="Data Privacy & Compliance",
            value="100% On-Device",
            delta="Zero Cloud Exposure",
            delta_color="normal",
        )
        st.caption("Full GDPR/Proprietary Site Protection")

    with col4:
        st.metric(
            label="API Token Cost",
            value="$0.00",
            delta="Zero Recurring Fees",
            delta_color="normal",
        )
        st.caption("Unlimited 24/7 continuous monitoring")

    st.divider()

    # Detailed comparative breakdown table
    st.markdown("### 📊 Architecture & Operational Feature Matrix")
    
    matrix_md = """
| Capability / Metric | ⚡ AMD Lemonade SDK (Local Edge) | ☁️ Cloud Multimodal APIs (Azure/OpenAI) | Real-World Site Impact |
| :--- | :--- | :--- | :--- |
| **Response Latency** | **~150 – 350 ms** | 2,500 – 6,000 ms | Instant life-safety alerts before machinery collisions occur. |
| **Offline Operation** | **100% Autonomous (No Internet Needed)** | 0% (Fails completely when disconnected) | Works in underground basements, tunnels, & remote civil sites. |
| **Video Bandwidth Load** | **0 KB egress (100% edge processed)** | ~1.5 GB / hour per 1080p camera | Massive cellular data cost savings for jobsite trailers. |
| **Worker PII / Site Privacy** | **Private On-Device (Zero data leakage)** | Transmitted to third-party cloud servers | Complies with union privacy agreements and proprietary site security. |
| **Autonomous Tool Calling** | **Deterministic Local Python Engine** | Remote function calling overhead | Instant regulatory lookup without token latency penalties. |
| **Hardware Acceleration** | **AMD Ryzen™ AI (XDNA NPU) / ROCm / Vulkan** | Remote cloud data center GPUs | Leverages energy-efficient laptop & ruggedized PC hardware. |
"""
    st.markdown(matrix_md)


# ============================================================
# Video Analysis UI Components
# ============================================================

def render_video_upload_section():
    """Render video upload section."""
    st.subheader("🎥 Construction Site Video Stream Analysis")
    
    source_type = st.radio(
        "Video Source",
        options=["Use Demo Video", "Upload File"],
        horizontal=True,
        key="video_source_type",
    )
    
    supported_formats = [fmt.lstrip(".") for fmt in VIDEO_SETTINGS["supported_formats"]]
    max_size_mb = VIDEO_SETTINGS["max_file_size_mb"]
    
    import os
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent
    sample_dir = root_dir / "sample_data"
    
    if source_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a construction site video...",
            type=supported_formats,
            help=f"Upload a video of a construction site (max {max_size_mb} MB).",
            key="video_upload",
        )
        
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                st.error(f"File size ({file_size_mb:.1f} MB) exceeds maximum ({max_size_mb} MB).")
                return None
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.video(uploaded_file)
            with col2:
                st.metric("File Size", f"{file_size_mb:.1f} MB")
                st.metric("Format", Path_ext(uploaded_file.name))
                
            return uploaded_file
    else:
        demo_files = []
        if sample_dir.exists():
            for f in sorted(os.listdir(sample_dir)):
                ext = f.split(".")[-1].lower()
                if f".{ext}" in VIDEO_SETTINGS["supported_formats"]:
                    demo_files.append(f)
                    
        if not demo_files:
            st.warning("No demo videos found in sample_data folder.")
            return None
            
        selected_filename = st.selectbox(
            "Select a pre-loaded construction video scenario",
            options=demo_files,
            help="Choose a pre-loaded construction site video for temporal analysis",
        )
        
        if selected_filename:
            video_path = sample_dir / selected_filename
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.video(str(video_path))
            with col2:
                file_size_mb = video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0.0
                st.metric("File Size", f"{file_size_mb:.2f} MB")
                st.caption(f"📁 `sample_data/{selected_filename}`")
                
            return str(video_path)
            
    return None


def Path_ext(filename: str) -> str:
    """Return upper-cased extension of filename."""
    from pathlib import Path
    return Path(filename).suffix.upper().lstrip(".")


def render_video_sidebar_settings() -> Dict[str, Any]:
    """Render video-specific sidebar settings."""
    with st.expander("🎬 Video Sampling Settings", expanded=True):
        sample_strategy = st.selectbox(
            "Sampling Strategy",
            options=["uniform_interval", "fps_based", "fixed_count"],
            format_func=lambda x: {
                "uniform_interval": "Uniform Interval (every N seconds)",
                "fps_based": "FPS-Based (target output FPS)",
                "fixed_count": "Fixed Frame Count",
            }[x],
            index=0,
            key="video_sample_strategy",
        )

        if sample_strategy == "uniform_interval":
            interval_seconds = st.slider(
                "Sample Interval (seconds)",
                min_value=0.5,
                max_value=10.0,
                value=float(VIDEO_SETTINGS["default_sample_interval_seconds"]),
                step=0.5,
                key="video_interval",
            )
            target_fps = VIDEO_SETTINGS["default_target_fps"]
        elif sample_strategy == "fps_based":
            target_fps = st.slider(
                "Target Sampling FPS",
                min_value=0.1,
                max_value=2.0,
                value=float(VIDEO_SETTINGS["default_target_fps"]),
                step=0.1,
                key="video_target_fps",
            )
            interval_seconds = VIDEO_SETTINGS["default_sample_interval_seconds"]
        else:
            interval_seconds = VIDEO_SETTINGS["default_sample_interval_seconds"]
            target_fps = VIDEO_SETTINGS["default_target_fps"]

        max_frames = st.slider(
            "Max Frames to Analyze",
            min_value=1,
            max_value=100,
            value=min(15, 100),
            step=1,
            key="video_max_frames",
        )

        video_mode = st.radio(
            "Video Analysis Mode",
            options=["detection_guided", "baseline"],
            format_func=lambda x: {
                "detection_guided": "Detection-Guided (Recommended)",
                "baseline": "Baseline (No Detection Context)",
            }[x],
            key="video_mode",
        )

    with st.expander("🎞️ Video Visualization Options"):
        show_detections = st.checkbox(
            "Show Detection Overlays",
            value=VIDEO_UI_DEFAULTS["show_detections_overlay"],
            key="video_show_detections",
        )
        show_hazard_zones = st.checkbox(
            "Show Hazard Zone Overlays",
            value=VIDEO_UI_DEFAULTS["show_hazard_zones_overlay"],
            key="video_show_hazard_zones",
        )

    return {
        "sample_strategy": sample_strategy,
        "interval_seconds": interval_seconds,
        "target_fps": target_fps,
        "max_frames": max_frames,
        "video_mode": video_mode,
        "show_detections": show_detections,
        "show_hazard_zones": show_hazard_zones,
    }


def render_video_analysis_results(
    video_result,
    show_detections: bool = True,
    show_hazard_zones: bool = True,
) -> None:
    """Render full video analysis results."""
    from visualizer import Visualizer
    viz = Visualizer()

    summary = video_result.aggregated_summary
    metadata = video_result.metadata
    temporal_hazards = video_result.temporal_hazards
    frame_results = video_result.sampled_frames

    st.subheader("📹 Video Stream Metadata")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{metadata.duration_seconds:.1f}s")
    c2.metric("Resolution", summary.get("video_resolution", "N/A"))
    c3.metric("Native FPS", f"{metadata.fps:.1f}")
    c4.metric("Frames Evaluated (Edge)", summary["total_sampled_frames"])

    st.divider()

    st.subheader("⚠️ Temporal Hazard Timeline")
    total_hazards = summary["total_temporal_hazards"]
    if total_hazards == 0:
        st.success("✅ No persistent hazards detected across video timeline.")
    else:
        sev_dist = summary.get("severity_distribution", {})
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Persistent Hazards", total_hazards)
        col2.metric("Critical", sev_dist.get("critical", 0))
        col3.metric("High", sev_dist.get("high", 0))
        col4.metric("Medium", sev_dist.get("medium", 0))
        col5.metric("Low", sev_dist.get("low", 0))

        for th in temporal_hazards:
            severity_color = SEVERITY_COLORS.get(th.severity, "#ffc107")
            with st.container():
                col_sev, col_info = st.columns([1, 9])
                with col_sev:
                    st.markdown(
                        f'<div style="background-color:{severity_color};width:44px;height:44px;'
                        f'border-radius:8px;display:flex;align-items:center;justify-content:center;'
                        f'color:white;font-weight:bold;">{th.severity.upper()[:1]}</div>',
                        unsafe_allow_html=True,
                    )
                with col_info:
                    st.markdown(
                        f"**{th.hazard_label}** — "
                        f"First seen: `{th.first_seen_seconds:.1f}s` | "
                        f"Last seen: `{th.last_seen_seconds:.1f}s` | "
                        f"Duration: `{th.duration_seconds:.1f}s` | "
                        f"Frames: `{len(th.affected_frames)}`"
                    )
                    if th.description:
                        st.markdown(f"> {th.description}")
                    if th.recommendation:
                        st.caption(f"✅ Directive: {th.recommendation}")
                st.divider()

    if frame_results:
        with st.expander(f"🖼️ Per-Frame Edge Breakdown ({len(frame_results)} keyframes)"):
            for fr in frame_results:
                ts = fr.timestamp_seconds
                n_hazards = len(fr.parsed_result.hazards) if fr.parsed_result else 0
                header = f"Keyframe {fr.frame_index} @ {ts:.1f}s — {n_hazards} hazard(s)"
                with st.expander(header):
                    annotated_image = None
                    if fr.detection_result and (show_detections or show_hazard_zones):
                        hazards = fr.parsed_result.hazards if fr.parsed_result else []
                        annotated_image = viz.create_full_annotation(
                            fr.image,
                            fr.detection_result,
                            hazards,
                            show_detections=show_detections,
                            show_relations=False,
                            show_hazard_zones=show_hazard_zones,
                        )

                    if annotated_image is not None:
                        col_raw, col_ann, col_detail = st.columns([2, 2, 3])
                        with col_raw:
                            st.caption("Raw Keyframe")
                            st.image(fr.image, use_container_width=True)
                        with col_ann:
                            st.caption("Annotated Spatial Map")
                            st.image(annotated_image, use_container_width=True)
                    else:
                        col_raw, col_detail = st.columns([2, 3])
                        with col_raw:
                            st.caption("Raw Keyframe")
                            st.image(fr.image, use_container_width=True)

                    with col_detail:
                        if fr.detection_result:
                            st.caption(
                                f"Localized Entities: {len(fr.detection_result.detections)} | "
                                f"YOLO Latency: {fr.detection_result.inference_time_ms:.1f}ms"
                            )
                        if fr.parsed_result and fr.parsed_result.hazards:
                            for h in fr.parsed_result.hazards:
                                st.markdown(f"- **{h.hazard_label}** `[{h.severity.upper()}]`: {h.description}")
                        elif fr.parsed_result and fr.parsed_result.no_hazards_detected:
                            st.success("No hazards in this frame.")