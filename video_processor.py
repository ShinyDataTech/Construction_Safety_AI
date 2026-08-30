"""
video_processor.py — Video decoding, frame sampling, temporal aggregation
and orchestration of the detection + VLM pipeline for video analysis.
"""

import io
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio

import numpy as np
from PIL import Image

from config import (
    CONSTRUCTION_ENTITY_LABELS,
    VIDEO_SETTINGS,
    SEVERITY_LEVELS,
)
from detector import ConstructionDetector, DetectionResult
from prompt_engineer import PromptEngineer
from response_parser import ResponseParser, ParsedResult, HazardAssessment


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """Result for a single sampled frame."""
    frame_index: int              # Original video frame number
    timestamp_seconds: float      # Time in video
    image: Image.Image            # Extracted frame (PIL)
    detection_result: Optional[DetectionResult] = None
    prompt: str = ""
    raw_response: str = ""
    parsed_result: Optional[ParsedResult] = None
    annotated_image: Optional[Image.Image] = None


@dataclass
class TemporalHazard:
    """Hazard aggregated across multiple frames."""
    hazard_type: str
    hazard_label: str
    severity: str
    severity_timeline: List[Tuple[float, str]] = field(default_factory=list)
    description: str = ""
    recommendation: str = ""
    first_seen_seconds: float = 0.0
    last_seen_seconds: float = 0.0
    duration_seconds: float = 0.0
    affected_frames: List[int] = field(default_factory=list)
    confidence: float = 0.0
    detected_entities: List[str] = field(default_factory=list)


@dataclass
class VideoMetadata:
    """Basic metadata extracted from a video file."""
    duration_seconds: float
    fps: float
    width: int
    height: int
    total_frames: int
    codec_hint: str = ""


@dataclass
class VideoAnalysisResult:
    """Complete result from analyzing a video."""
    video_path: str
    metadata: VideoMetadata
    sampled_frames: List[FrameResult] = field(default_factory=list)
    temporal_hazards: List[TemporalHazard] = field(default_factory=list)
    aggregated_summary: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Frame Sampler
# ──────────────────────────────────────────────────────────────

class FrameSampler:
    """
    Samples frames from a video using configurable strategies.
    
    Strategies:
    - uniform_interval : sample every N seconds
    - fps_based        : sample at target FPS
    - fixed_count      : extract exactly N evenly distributed frames
    - keyframe_only    : use iio to read at specific indices only
    """

    def __init__(
        self,
        strategy: str = VIDEO_SETTINGS["default_sample_strategy"],
        interval_seconds: float = VIDEO_SETTINGS["default_sample_interval_seconds"],
        target_fps: float = VIDEO_SETTINGS["default_target_fps"],
        max_frames: int = VIDEO_SETTINGS["default_max_frames"],
        scene_change_threshold: float = VIDEO_SETTINGS["scene_change_threshold"],
    ):
        self.strategy = strategy
        self.interval_seconds = interval_seconds
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.scene_change_threshold = scene_change_threshold

    def sample_indices(self, metadata: VideoMetadata) -> List[int]:
        """Return list of frame indices to sample."""
        total = metadata.total_frames
        if total <= 0:
            return []

        if self.strategy == "uniform_interval":
            step = max(1, int(self.interval_seconds * metadata.fps))
            indices = list(range(0, total, step))

        elif self.strategy == "fps_based":
            step = max(1, int(metadata.fps / self.target_fps))
            indices = list(range(0, total, step))

        elif self.strategy == "fixed_count":
            count = min(self.max_frames, total)
            if count <= 1:
                indices = [0]
            else:
                indices = [int(i * (total - 1) / (count - 1)) for i in range(count)]

        elif self.strategy == "keyframe_only":
            # Simple uniform fallback; true keyframe detection needs container parsing
            count = min(self.max_frames, total)
            if count <= 1:
                indices = [0]
            else:
                indices = [int(i * (total - 1) / (count - 1)) for i in range(count)]
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

        # Hard cap for safety
        if len(indices) > self.max_frames:
            subset_count = self.max_frames
            if subset_count <= 1:
                indices = [indices[0]]
            else:
                step = max(1, len(indices) // subset_count)
                indices = indices[::step][:subset_count]

        return sorted(set(indices))


# ──────────────────────────────────────────────────────────────
# Video Decoder
# ──────────────────────────────────────────────────────────────

class VideoDecoder:
    """Decode video metadata and sample frames using imageio."""

    @staticmethod
    def read_metadata(video_path: str) -> VideoMetadata:
        """Extract metadata without loading all frames."""
        try:
            meta = iio.immeta(video_path, plugin="pyav")
        except Exception:
            # Fallback: try to read first few frames to determine properties
            reader = iio.imiter(video_path)
            first_frame = next(reader)
            height, width = first_frame.shape[:2]
            # Estimate total frames by reading all
            count = 1
            for _ in reader:
                count += 1
            fps = 30.0  # default guess
            duration = count / fps
            return VideoMetadata(
                duration_seconds=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=count,
            )

        fps = meta.get("fps", 30.0)
        if isinstance(fps, (list, tuple)):
            fps = float(fps[0]) if fps else 30.0
        fps = float(fps) if fps else 30.0

        duration = meta.get("duration", None)
        if duration is None:
            duration = 0.0
        elif isinstance(duration, (list, tuple)):
            duration = float(duration[0]) if duration else 0.0
        else:
            duration = float(duration)

        # For pyav, duration might be in seconds already; otherwise convert from stream
        if duration > 100000:
            # likely in time-base units, convert
            duration = duration / (meta.get("time_base", 1.0) or 1.0)

        size = meta.get("size", (1920, 1080))
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            width, height = int(size[0]), int(size[1])
        else:
            width, height = 1920, 1080

        # Estimate total frames
        total_frames = meta.get("nframes", 0)
        if total_frames == 0 and duration > 0 and fps > 0:
            total_frames = int(duration * fps)

        return VideoMetadata(
            duration_seconds=duration if duration > 0 else total_frames / fps,
            fps=fps,
            width=width,
            height=height,
            total_frames=max(total_frames, 1),
            codec_hint=meta.get("codec", ""),
        )

    @staticmethod
    def read_frames_at_indices(
        video_path: str, indices: List[int]
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Lazily yield (frame_index, frame_array) for requested indices.
        Uses imageio with index-based reading for memory efficiency.
        """
        if not indices:
            return

        # Sort to allow sequential access optimization
        sorted_indices = sorted(set(indices))

        for idx in sorted_indices:
            try:
                frame = iio.imread(video_path, index=idx)
                yield idx, frame
            except Exception:
                # If index-based reading fails, skip this frame
                continue


# ──────────────────────────────────────────────────────────────
# Temporal Aggregator
# ──────────────────────────────────────────────────────────────

class TemporalAggregator:
    """
    Aggregates per-frame hazards into temporally consistent findings.
    """

    def __init__(
        self,
        min_persistence_frames: int = VIDEO_SETTINGS["min_persistence_frames"],
    ):
        self.min_persistence_frames = min_persistence_frames

    def aggregate(
        self, frame_results: List[FrameResult]
    ) -> List[TemporalHazard]:
        """
        Convert a list of FrameResult hazards into TemporalHazard objects.
        
        Matching logic:
        - Hazards match if they share the same hazard_type AND
          share at least one detected entity.
        """
        if not frame_results:
            return []

        # Collect all frame-level hazards keyed by frame index
        frame_hazards: Dict[int, List[Tuple[float, HazardAssessment]]] = {}
        for fr in frame_results:
            if fr.parsed_result and fr.parsed_result.hazards:
                frame_hazards[fr.frame_index] = [
                    (fr.timestamp_seconds, h) for h in fr.parsed_result.hazards
                ]

        # Track raw runs: list of (frame_index, timestamp, hazard)
        runs: Dict[str, List[Tuple[int, float, HazardAssessment]]] = {}

        for frame_idx, hazards in frame_hazards.items():
            for ts, h in hazards:
                key = self._hazard_key(h)
                runs.setdefault(key, []).append((frame_idx, ts, h))

        temporal: List[TemporalHazard] = []
        for key, entries in runs.items():
            if len(entries) < self.min_persistence_frames:
                continue

            entries.sort(key=lambda x: x[0])
            # Take the most common severity as base; build timeline
            severities = [e[2].severity for e in entries]
            severity_counts = {s: severities.count(s) for s in set(severities)}
            base_severity = max(severity_counts, key=severity_counts.get)

            # Peak severity based on ordering
            severity_order = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
            peak_severity = max(
                severities, key=lambda s: severity_order.get(s, 0)
            )

            # Choose the longest description
            descriptions = [e[2].description.strip() for e in entries if e[2].description]
            best_description = max(descriptions, key=len) if descriptions else ""

            recommendations = [e[2].recommendation.strip() for e in entries if e[2].recommendation]
            best_recommendation = recommendations[0] if recommendations else ""

            # Entities
            all_entities = []
            for e in entries:
                all_entities.extend(e[2].detected_entities or [])
            unique_entities = sorted(set(all_entities))

            # Confidence average
            confidences = [e[2].confidence for e in entries if e[2].confidence]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            t = TemporalHazard(
                hazard_type=entries[0][2].hazard_type,
                hazard_label=entries[0][2].hazard_label,
                severity=peak_severity,
                severity_timeline=[(e[1], e[2].severity) for e in entries],
                description=best_description,
                recommendation=best_recommendation,
                first_seen_seconds=entries[0][1],
                last_seen_seconds=entries[-1][1],
                duration_seconds=entries[-1][1] - entries[0][1],
                affected_frames=[e[0] for e in entries],
                confidence=avg_conf,
                detected_entities=unique_entities,
            )
            temporal.append(t)

        # Sort by first appearance
        temporal.sort(key=lambda x: x.first_seen_seconds)
        return temporal

    @staticmethod
    def _hazard_key(hazard: HazardAssessment) -> str:
        """Create a canonical key for matching hazards across frames."""
        entities = sorted(set(hazard.detected_entities or []))
        return f"{hazard.hazard_type}|{','.join(entities)}"


# ──────────────────────────────────────────────────────────────
# Video Processor (Orchestrator)
# ──────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    End-to-end video analysis orchestrator.
    """

    def __init__(
        self,
        detector: ConstructionDetector,
        vlm,
        prompt_engineer: PromptEngineer,
        response_parser: ResponseParser,
        sampler: Optional[FrameSampler] = None,
        aggregator: Optional[TemporalAggregator] = None,
        confidence_threshold: float = 0.25,
        max_frames: int = VIDEO_SETTINGS["default_max_frames"],
        sample_strategy: str = VIDEO_SETTINGS["default_sample_strategy"],
        sample_interval: float = VIDEO_SETTINGS["default_sample_interval_seconds"],
    ):
        self.detector = detector
        self.vlm = vlm
        self.prompt_engineer = prompt_engineer
        self.response_parser = response_parser
        self.sampler = sampler or FrameSampler(
            strategy=sample_strategy,
            interval_seconds=sample_interval,
            max_frames=max_frames,
        )
        self.aggregator = aggregator or TemporalAggregator()
        self.confidence_threshold = confidence_threshold
        self.max_frames = max_frames

    def process_video(
        self,
        video_path: str,
        mode: str = "detection_guided",
        progress_callback=None,
        cancelled_flag=None,
    ) -> VideoAnalysisResult:
        """
        Full pipeline: decode → sample → detect → prompt → VLM → parse → aggregate.

        Args:
            video_path: Path to video file.
            mode: "baseline" or "detection_guided".
            progress_callback: Callable(current, total) for progress updates.
            cancelled_flag: Callable() -> bool; if True, abort early.
        """
        # ── Step 1: Metadata ──
        metadata = VideoDecoder.read_metadata(video_path)

        # ── Step 2: Sample frames ──
        indices = self.sampler.sample_indices(metadata)
        total = len(indices)

        frame_results: List[FrameResult] = []

        # ── Step 3: Process each sampled frame ──
        for i, (frame_idx, frame_arr) in enumerate(
            VideoDecoder.read_frames_at_indices(video_path, indices)
        ):
            if cancelled_flag and cancelled_flag():
                break

            timestamp = frame_idx / metadata.fps if metadata.fps > 0 else 0.0

            # Convert numpy array to PIL Image
            if frame_arr.shape[-1] == 4:
                # RGBA
                pil_image = Image.fromarray(frame_arr, mode="RGBA").convert("RGB")
            elif len(frame_arr.shape) == 2:
                # Grayscale
                pil_image = Image.fromarray(frame_arr).convert("RGB")
            else:
                pil_image = Image.fromarray(frame_arr)

            # ── Detection ──
            if mode == "detection_guided":
                detection_result = self.detector.detect(
                    pil_image, confidence_threshold=self.confidence_threshold
                )
            else:
                detection_result = None

            # ── Prompt ──
            if mode == "detection_guided":
                prompt = self.prompt_engineer.build_prompt(
                    detection_result, mode="detection_guided"
                )
            else:
                prompt = self.prompt_engineer.build_prompt(None, mode="baseline")

            # ── VLM ──
            try:
                vlm_result = self.vlm.infer(pil_image, prompt)
                raw_response = vlm_result["raw_output"]
            except Exception as e:
                raw_response = f"Error during VLM inference: {e}"

            # ── Parse ──
            parsed = self.response_parser.parse(raw_response)

            # ── Pack result ──
            frame_results.append(
                FrameResult(
                    frame_index=frame_idx,
                    timestamp_seconds=timestamp,
                    image=pil_image,
                    detection_result=detection_result,
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_result=parsed,
                )
            )

            if progress_callback:
                progress_callback(i + 1, total)

        # ── Step 4: Temporal aggregation ──
        temporal_hazards = self.aggregator.aggregate(frame_results)

        # ── Step 5: Build summary ──
        summary = self._build_summary(frame_results, temporal_hazards, metadata)

        return VideoAnalysisResult(
            video_path=video_path,
            metadata=metadata,
            sampled_frames=frame_results,
            temporal_hazards=temporal_hazards,
            aggregated_summary=summary,
        )

    @staticmethod
    def _build_summary(
        frame_results: List[FrameResult],
        temporal_hazards: List[TemporalHazard],
        metadata: VideoMetadata,
    ) -> Dict[str, Any]:
        """Build aggregated statistics for the video."""
        severities = [th.severity for th in temporal_hazards]
        severity_counts = {s: severities.count(s) for s in set(severities)}

        hazard_types = [th.hazard_type for th in temporal_hazards]
        type_counts = {t: hazard_types.count(t) for t in set(hazard_types)}

        # Max simultaneous hazards at any sampled frame
        max_simultaneous = 0
        for fr in frame_results:
            if fr.parsed_result and fr.parsed_result.hazards:
                max_simultaneous = max(
                    max_simultaneous, len(fr.parsed_result.hazards)
                )

        longest = None
        if temporal_hazards:
            longest = max(temporal_hazards, key=lambda x: x.duration_seconds)

        return {
            "total_sampled_frames": len(frame_results),
            "total_temporal_hazards": len(temporal_hazards),
            "severity_distribution": severity_counts,
            "hazard_type_distribution": type_counts,
            "max_simultaneous_hazards": max_simultaneous,
            "longest_hazard_duration_seconds": (
                longest.duration_seconds if longest else 0.0
            ),
            "longest_hazard_type": longest.hazard_type if longest else "",
            "video_duration_seconds": metadata.duration_seconds,
            "video_fps": metadata.fps,
            "video_resolution": f"{metadata.width}x{metadata.height}",
        }


# ──────────────────────────────────────────────────────────────
# Utility: save uploaded video to temp file
# ──────────────────────────────────────────────────────────────

def save_uploaded_video(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temporary path and return the path."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name
