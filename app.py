"""
SafeSite-AI: Autonomous Construction Safety Agent.
Powered by AMD Lemonade SDK (Local Edge Inference).

Implements a two-stage detection-guided sVLM & autonomous agent framework:
1. YOLOv11n object detection for entity localization & spatial proximity
2. Structured prompt-conditioned local VLM reasoning via AMD Lemonade SDK (localhost:13305/v1)
3. Autonomous Safety Agent with deterministic OSHA 1926 tool calling & report synthesis
4. Temporal video stream analysis & edge vs cloud benchmarking

Usage:
    streamlit run app.py
"""

import asyncio
import sys

# Suppress benign WinError 10054 noise from asyncio ProactorEventLoop on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from PIL import Image
from typing import Optional, Dict, Any

# Application modules
from config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_ICON,
    VLM_MODELS,
    DEFAULT_VLM,
    SUPPORTED_IMAGE_TYPES,
    HAZARD_CATEGORIES,
    VIDEO_SETTINGS,
)
from detector import ConstructionDetector, DetectionResult
from prompt_engineer import PromptEngineer
from vlm_interface import VLMInterface, check_lemonade_health
from response_parser import ResponseParser, ParsedResult
from visualizer import Visualizer
from video_processor import VideoProcessor, FrameSampler, save_uploaded_video
from safety_agent import ConstructionSafetyAgent
from ui_components import (
    render_header,
    render_lemonade_status_banner,
    render_sidebar,
    render_upload_section,
    render_detection_results,
    render_hazard_assessment,
    render_comparison_results,
    render_annotated_image,
    render_raw_output,
    render_prompt_preview,
    render_progress_indicator,
    render_error,
    render_autonomous_agent_tab,
    render_edge_benchmark_dashboard,
    render_video_upload_section,
    render_video_sidebar_settings,
    render_video_analysis_results,
)


# ============================================================
# Cached Resource Loading (Streamlit caching for performance)
# ============================================================

@st.cache_resource
def get_detector() -> ConstructionDetector:
    """Load and cache the YOLOv11n detector."""
    return ConstructionDetector()


@st.cache_resource
def get_vlm(model_key: str) -> VLMInterface:
    """Load and cache the VLM model."""
    return VLMInterface(model_key)


@st.cache_resource
def get_prompt_engineer() -> PromptEngineer:
    """Get the prompt engineer."""
    return PromptEngineer()


@st.cache_resource
def get_response_parser() -> ResponseParser:
    """Get the response parser."""
    return ResponseParser()


@st.cache_resource
def get_visualizer() -> Visualizer:
    """Get the visualizer."""
    return Visualizer()


def get_or_create_safety_agent(model_key: str) -> ConstructionSafetyAgent:
    """Get or initialize the autonomous safety agent in session state."""
    if "safety_agent" not in st.session_state or st.session_state.get("safety_agent_model") != model_key:
        st.session_state["safety_agent"] = ConstructionSafetyAgent(model_key)
        st.session_state["safety_agent_model"] = model_key
    return st.session_state["safety_agent"]


# ============================================================
# Main Pipeline
# ============================================================

def run_detection_guided_pipeline(
    image: Image.Image,
    model_key: str,
    confidence_threshold: float,
    max_tokens: int,
    temperature: float,
    focus_categories: list,
) -> Dict[str, Any]:
    """
    Run the full detection-guided pipeline.
    Stage 1: YOLOv11n detection & spatial relationship analysis
    Stage 2: Structured prompt conditioning + Lemonade sVLM inference
    Stage 3: Response parsing
    """
    detector = get_detector()
    prompt_eng = get_prompt_engineer()
    parser = get_response_parser()
    
    # Stage 1: Object Detection
    render_progress_indicator("detecting")
    detection_result = detector.detect(image, confidence_threshold)
    
    # Stage 2: Prompt Engineering + Lemonade Local VLM Inference
    render_progress_indicator("loading_model")
    vlm = get_vlm(model_key)
    
    render_progress_indicator("inferring")
    if focus_categories and len(focus_categories) < len(HAZARD_CATEGORIES):
        prompt = prompt_eng.build_hazard_focus_prompt(detection_result, focus_categories)
    else:
        prompt = prompt_eng.build_prompt(detection_result, mode="detection_guided")
    
    vlm_result = vlm.infer(image, prompt, max_new_tokens=max_tokens, temperature=temperature)
    
    # Stage 3: Response Parsing
    render_progress_indicator("parsing")
    parsed_result = parser.parse(vlm_result["raw_output"])
    
    return {
        "detection_result": detection_result,
        "prompt": prompt,
        "vlm_result": vlm_result,
        "parsed_result": parsed_result,
    }


def run_baseline_pipeline(
    image: Image.Image,
    model_key: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Run baseline pipeline without detection conditioning."""
    prompt_eng = get_prompt_engineer()
    parser = get_response_parser()
    
    prompt = prompt_eng.build_prompt(None, mode="baseline")
    
    render_progress_indicator("loading_model")
    vlm = get_vlm(model_key)
    render_progress_indicator("inferring")
    vlm_result = vlm.infer(image, prompt, max_new_tokens=max_tokens, temperature=temperature)
    
    render_progress_indicator("parsing")
    parsed_result = parser.parse(vlm_result["raw_output"])
    
    return {
        "prompt": prompt,
        "vlm_result": vlm_result,
        "parsed_result": parsed_result,
    }


def run_comparison_pipeline(
    image: Image.Image,
    model_key: str,
    confidence_threshold: float,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Run both baseline and detection-guided pipelines for side-by-side comparison."""
    detector = get_detector()
    prompt_eng = get_prompt_engineer()
    parser = get_response_parser()
    
    render_progress_indicator("detecting")
    detection_result = detector.detect(image, confidence_threshold)
    
    prompts = prompt_eng.build_comparison_prompts(detection_result)
    
    render_progress_indicator("loading_model")
    vlm = get_vlm(model_key)
    
    render_progress_indicator("inferring")
    baseline_result = vlm.infer(image, prompts["baseline"], max_new_tokens=max_tokens, temperature=temperature)
    guided_result = vlm.infer(image, prompts["detection_guided"], max_new_tokens=max_tokens, temperature=temperature)
    
    render_progress_indicator("parsing")
    baseline_parsed = parser.parse(baseline_result["raw_output"])
    guided_parsed = parser.parse(guided_result["raw_output"])
    
    return {
        "detection_result": detection_result,
        "baseline_result": baseline_result,
        "guided_result": guided_result,
        "baseline_parsed": baseline_parsed,
        "guided_parsed": guided_parsed,
        "baseline_prompt": prompts["baseline"],
        "guided_prompt": prompts["detection_guided"],
    }


# ============================================================
# Main Application
# ============================================================

def main():
    """Main Streamlit application entry point."""
    render_header()
    
    # Title and description
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown(APP_DESCRIPTION)
    
    # Real-time Lemonade Connection Banner
    lemonade_health = check_lemonade_health()
    render_lemonade_status_banner(lemonade_health)
    
    st.divider()
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Top-level tabs
    tab_image, tab_agent, tab_video, tab_benchmark = st.tabs([
        "📸 Site Image Inspection",
        "🤖 Autonomous Safety Agent",
        "🎥 Temporal Video Monitoring",
        "⚡ Lemonade Edge Benchmark",
    ])

    # Shared image state in session
    if "current_image" not in st.session_state:
        st.session_state["current_image"] = None
    if "current_detection_result" not in st.session_state:
        st.session_state["current_detection_result"] = None

    # ============================================================
    # Tab 1: Image Analysis
    # ============================================================
    with tab_image:
        uploaded_image = render_upload_section()
        if uploaded_image is not None:
            st.session_state["current_image"] = uploaded_image
            
        if uploaded_image is None:
            st.info("Upload or select a construction site image to begin inspection.")
        else:
            if st.button("🚀 Run Hazard Inspection", type="primary", use_container_width=True):
                try:
                    if config["analysis_mode"] == "detection_guided":
                        result = run_detection_guided_pipeline(
                            image=uploaded_image,
                            model_key=config["selected_model"],
                            confidence_threshold=config["confidence_threshold"],
                            max_tokens=config["max_tokens"],
                            temperature=config["temperature"],
                            focus_categories=config["focus_categories"],
                        )
                        
                        st.session_state["current_detection_result"] = result["detection_result"]
                        
                        render_detection_results(result["detection_result"])
                        
                        viz = get_visualizer()
                        annotated_img = viz.create_full_annotation(
                            uploaded_image,
                            result["detection_result"],
                            result["parsed_result"].hazards,
                            show_detections=config["show_detections"],
                            show_relations=config["show_relations"],
                            show_hazard_zones=config["show_hazard_zones"],
                        )
                        render_annotated_image(annotated_img, "YOLOv11 Detection + Hazard Spatial Map")
                        
                        render_hazard_assessment(
                            result["parsed_result"].hazards,
                            result["parsed_result"],
                        )
                        
                        st.subheader("⚡ Edge Latency Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("YOLOv11 Detection", f"{result['detection_result'].inference_time_ms:.1f} ms")
                        with col2:
                            st.metric("Lemonade Local VLM", f"{result['vlm_result']['inference_time_ms']:.1f} ms")
                        with col3:
                            total_time = (
                                result['detection_result'].inference_time_ms +
                                result['vlm_result']['inference_time_ms']
                            )
                            st.metric("Total Edge Turnaround", f"{total_time:.1f} ms", delta="Zero Cloud Latency")
                        
                    elif config["analysis_mode"] == "baseline":
                        result = run_baseline_pipeline(
                            image=uploaded_image,
                            model_key=config["selected_model"],
                            max_tokens=config["max_tokens"],
                            temperature=config["temperature"],
                        )
                        render_hazard_assessment(
                            result["parsed_result"].hazards,
                            result["parsed_result"],
                        )
                        st.metric("Lemonade Inference Time", f"{result['vlm_result']['inference_time_ms']:.1f} ms")
                        
                    elif config["analysis_mode"] == "comparison":
                        result = run_comparison_pipeline(
                            image=uploaded_image,
                            model_key=config["selected_model"],
                            confidence_threshold=config["confidence_threshold"],
                            max_tokens=config["max_tokens"],
                            temperature=config["temperature"],
                        )
                        st.session_state["current_detection_result"] = result["detection_result"]
                        render_detection_results(result["detection_result"])
                        
                        viz = get_visualizer()
                        annotated_img = viz.create_full_annotation(
                            uploaded_image,
                            result["detection_result"],
                            result["guided_parsed"].hazards,
                            show_detections=config["show_detections"],
                            show_relations=config["show_relations"],
                            show_hazard_zones=config["show_hazard_zones"],
                        )
                        render_annotated_image(annotated_img, "Detection-Guided Annotation")
                        
                        render_comparison_results(
                            result["baseline_result"],
                            result["guided_result"],
                            result["baseline_parsed"],
                            result["guided_parsed"],
                        )
                
                except Exception as e:
                    render_error(f"Analysis failed: {str(e)}")
                    st.error(
                        "Troubleshooting tips:\n"
                        "1. Ensure AMD Lemonade SDK is running (`lemonade serve --port 13305`)\n"
                        "2. Check if the model is pulled (`lemonade pull qwen2.5-vl-7b-instruct`)\n"
                        "3. Or select 'GPT-4o (Azure OpenAI)' in the sidebar if comparing against cloud baseline."
                    )

    # ============================================================
    # Tab 2: Autonomous Safety Agent
    # ============================================================
    with tab_agent:
        safety_agent = get_or_create_safety_agent(config["selected_model"])
        current_img = st.session_state.get("current_image")
        current_det = st.session_state.get("current_detection_result")
        
        # If no detection was run yet, run YOLO automatically on image if present
        if current_img is not None and current_det is None:
            detector = get_detector()
            current_det = detector.detect(current_img, config["confidence_threshold"])
            st.session_state["current_detection_result"] = current_det

        render_autonomous_agent_tab(safety_agent, current_img, current_det)

    # ============================================================
    # Tab 3: Video Analysis
    # ============================================================
    with tab_video:
        with st.sidebar:
            st.divider()
            video_config = render_video_sidebar_settings()

        uploaded_video = render_video_upload_section()

        if uploaded_video is None:
            st.info("Upload or select a construction site video to begin temporal analysis.")
        else:
            if st.button("🚀 Analyze Video Stream (Local Edge)", type="primary", use_container_width=True, key="analyze_video_btn"):
                video_path = None
                is_temp_video = False
                try:
                    if isinstance(uploaded_video, str):
                        video_path = uploaded_video
                        is_temp_video = False
                    else:
                        with st.spinner("Saving video stream to local edge buffer..."):
                            video_path = save_uploaded_video(uploaded_video)
                            is_temp_video = True

                    detector = get_detector()
                    vlm = get_vlm(config["selected_model"])
                    prompt_eng = get_prompt_engineer()
                    parser = get_response_parser()

                    sampler = FrameSampler(
                        strategy=video_config["sample_strategy"],
                        interval_seconds=video_config["interval_seconds"],
                        target_fps=video_config["target_fps"],
                        max_frames=video_config["max_frames"],
                    )

                    processor = VideoProcessor(
                        detector=detector,
                        vlm=vlm,
                        prompt_engineer=prompt_eng,
                        response_parser=parser,
                        sampler=sampler,
                        confidence_threshold=config["confidence_threshold"],
                        max_frames=video_config["max_frames"],
                        sample_strategy=video_config["sample_strategy"],
                        sample_interval=video_config["interval_seconds"],
                    )

                    progress_bar = st.progress(0, text="Processing frames locally...")

                    def update_progress(current: int, total: int):
                        pct = int(current / total * 100) if total > 0 else 100
                        progress_bar.progress(pct, text=f"Analyzing keyframe {current}/{total} via Lemonade SDK...")

                    video_result = processor.process_video(
                        video_path=video_path,
                        mode=video_config["video_mode"],
                        progress_callback=update_progress,
                    )

                    progress_bar.empty()
                    render_video_analysis_results(
                        video_result,
                        show_detections=video_config["show_detections"],
                        show_hazard_zones=video_config["show_hazard_zones"],
                    )

                except Exception as e:
                    render_error(f"Video analysis failed: {str(e)}")
                finally:
                    if video_path and is_temp_video:
                        try:
                            import os
                            os.unlink(video_path)
                        except Exception:
                            pass

    # ============================================================
    # Tab 4: Edge Benchmark
    # ============================================================
    with tab_benchmark:
        render_edge_benchmark_dashboard()


if __name__ == "__main__":
    main()