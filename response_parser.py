"""
Response Parser Module for Construction Safety AI.
Parses raw VLM output into structured HazardAssessment objects.

Post-processing: Parse structured VLM output for hazard identification results.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from config import HAZARD_CATEGORIES, SEVERITY_LEVELS


@dataclass
class HazardAssessment:
    """Represents a single identified safety hazard."""
    hazard_type: str           # Category key (e.g., "fall_from_height")
    hazard_label: str          # Display label (e.g., "Fall from Height")
    severity: str              # Severity level: low, medium, high, critical
    description: str           # Natural language description of the hazard
    recommendation: str        # Recommended corrective action
    confidence: Optional[float] = None  # Detection-guided confidence (if available)
    detected_entities: List[str] = field(default_factory=list)  # Related entity labels


@dataclass
class ParsedResult:
    """Represents the parsed result from a VLM inference."""
    hazards: List[HazardAssessment] = field(default_factory=list)
    no_hazards_detected: bool = False
    raw_output: str = ""
    parse_success: bool = True
    parse_warnings: List[str] = field(default_factory=list)


class ResponseParser:
    """
    Parses raw VLM output text into structured HazardAssessment objects.
    
    The VLM generates hazard identification text following a structured prompt.
    This parser extracts:
    - Hazard type and maps to standard categories
    - Severity level
    - Description and recommendation
    - Handles both well-structured and semi-structured VLM outputs
    """

    # Regex patterns for parsing structured hazard output
    HAZARD_TYPE_PATTERNS = {
        "fall_from_height": [
            r"fall\s*from\s*height",
            r"fall\s*hazard",
            r"elevation\s*hazard",
            r"height\s*risk",
            r"working\s*at\s*height",
            r"fall\s*risk",
        ],
        "struck_by": [
            r"struck\s*by",
            r"struck.*object",
            r"struck.*equipment",
            r"hit\s*by",
            r"impact\s*hazard",
        ],
        "caught_in_between": [
            r"caught\s*in",
            r"caught\s*between",
            r"pinch\s*point",
            r"crush\s*hazard",
            r"entanglement",
        ],
        "electrical": [
            r"electrical\s*hazard",
            r"electrocution",
            r"electrical\s*risk",
            r"power\s*line",
            r"wiring\s*hazard",
        ],
        "excavation_trenching": [
            r"excavation",
            r"trench",
            r"trenching\s*hazard",
            r"cave\s*in",
            r"excavation\s*risk",
        ],
        "ppe_non_compliance": [
            r"ppe\s*non\s*compliance",
            r"missing\s*helmet",
            r"no\s*hard\s*hat",
            r"missing\s*protective",
            r"ppe\s*violation",
            r"no\s*safety\s*equipment",
            r"without\s*helmet",
            r"without\s*ppe",
        ],
        "unsafe_proximity": [
            r"unsafe\s*proximity",
            r"worker.*machinery",
            r"close.*equipment",
            r"proximity.*hazard",
            r"too\s*close",
            r"near.*machinery",
            r"near.*equipment",
            r"near.*vehicle",
        ],
    }

    SEVERITY_PATTERNS = {
        "critical": [r"critical", r"severe", r"extreme", r"imminent\s*danger"],
        "high": [r"high", r"serious", r"significant", r"dangerous"],
        "medium": [r"medium", r"moderate", r"moderate\s*risk"],
        "low": [r"low", r"minor", r"minimal", r"slight"],
    }

    def parse(self, raw_output: str) -> ParsedResult:
        """
        Parse raw VLM output into structured hazard assessments.
        
        Args:
            raw_output: Raw text output from the VLM
            
        Returns:
            ParsedResult containing list of HazardAssessment objects
        """
        if not raw_output or raw_output.strip() == "":
            return ParsedResult(
                hazards=[],
                no_hazards_detected=True,
                raw_output=raw_output,
                parse_success=False,
                parse_warnings=["Empty output from VLM"],
            )
        
        # Check for "no hazards" response
        no_hazard_patterns = [
            r"no\s*hazards?\s*detected",
            r"no\s*safety\s*hazards?",
            r"safe\s*environment",
            r"no\s*risk\s*identified",
        ]
        
        for pattern in no_hazard_patterns:
            if re.search(pattern, raw_output.lower()):
                return ParsedResult(
                    hazards=[],
                    no_hazards_detected=True,
                    raw_output=raw_output,
                    parse_success=True,
                )
        
        # Attempt structured parsing first
        hazards = self._parse_structured_output(raw_output)
        
        # If structured parsing fails, fall back to unstructured parsing
        if not hazards:
            hazards = self._parse_unstructured_output(raw_output)
        
        if not hazards:
            # Last resort: extract any hazard-like mentions
            hazards = self._extract_hazard_mentions(raw_output)
        
        warnings = []
        if not hazards and not self._is_no_hazard_response(raw_output):
            warnings.append("Could not parse any hazards from VLM output. Raw text may not follow expected format.")
        
        return ParsedResult(
            hazards=hazards,
            no_hazards_detected=len(hazards) == 0,
            raw_output=raw_output,
            parse_success=len(hazards) > 0 or self._is_no_hazard_response(raw_output),
            parse_warnings=warnings,
        )

    def _is_no_hazard_response(self, text: str) -> bool:
        """Check if the text indicates no hazards were found."""
        lower = text.lower().strip()
        no_hazard_phrases = [
            "no hazards detected",
            "no safety hazards",
            "no hazard",
            "safe environment",
            "no risk",
            "no hazards present",
            "no relevant hazards",
        ]
        return any(phrase in lower for phrase in no_hazard_phrases)

    def _parse_structured_output(self, raw_output: str) -> List[HazardAssessment]:
        """
        Parse well-structured VLM output that follows the expected format.
        
        Expected format per hazard:
        1. Hazard type: ...
        2. Severity level: ...
        3. Description: ...
        4. Recommended corrective action: ...
        """
        hazards = []
        
        # Split into hazard blocks (separated by numbered items or blank lines)
        blocks = self._split_into_blocks(raw_output)
        
        for block in blocks:
            hazard = self._parse_hazard_block(block)
            if hazard:
                hazards.append(hazard)
        
        return hazards

    def _split_into_blocks(self, text: str) -> List[str]:
        """Split the VLM output into individual hazard blocks."""
        # Try splitting by numbered hazard indicators
        # Pattern: "Hazard 1", "1.", "Hazard:", etc.
        patterns = [
            r"(?:hazard\s*\d+|hazard\s*:)",           # "Hazard 1:", "Hazard:"
            r"(?:\d+\.\s*hazard\s*type)",               # "1. Hazard type"
            r"(?:\n\s*\d+\.\s*\n)",                     # Numbered sections
        ]
        
        blocks = []
        
        # Try splitting by "Hazard" keyword occurrences
        hazard_starts = []
        for match in re.finditer(r"(?:hazard\s*\d+|hazard\s*:|\d+\.\s*hazard)", text.lower()):
            hazard_starts.append(match.start())
        
        if hazard_starts:
            for i, start in enumerate(hazard_starts):
                end = hazard_starts[i + 1] if i + 1 < len(hazard_starts) else len(text)
                blocks.append(text[start:end])
        else:
            # Try splitting by numbered items (1., 2., 3., etc.)
            numbered_splits = re.split(r"\n\s*(?=\d+\.\s)", text)
            if len(numbered_splits) > 1:
                blocks = numbered_splits
            else:
                # Treat entire text as one block
                blocks = [text]
        
        return blocks

    def _parse_hazard_block(self, block: str) -> Optional[HazardAssessment]:
        """Parse a single hazard block into a HazardAssessment."""
        # Extract hazard type
        hazard_type, hazard_label = self._extract_hazard_type(block)
        
        # Extract severity
        severity = self._extract_severity(block)
        
        # Extract description
        description = self._extract_description(block)
        
        # Extract recommendation
        recommendation = self._extract_recommendation(block)
        
        if hazard_type or description:
            # If we couldn't classify the hazard type but have a description,
            # use "other" as type
            if not hazard_type:
                hazard_type = "other"
                hazard_label = "Other Hazard"
            
            if not severity:
                severity = "medium"  # Default severity
            
            return HazardAssessment(
                hazard_type=hazard_type,
                hazard_label=hazard_label,
                severity=severity,
                description=description or "Hazard detected (details not parseable)",
                recommendation=recommendation or "Follow standard safety procedures",
            )
        
        return None

    def _extract_hazard_type(self, text: str) -> tuple[str, str]:
        """Extract and classify the hazard type from text."""
        text_lower = text.lower()
        
        for category_key, patterns in self.HAZARD_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    label = HAZARD_CATEGORIES[category_key]["label"]
                    return category_key, label
        
        # Check if any hazard category label appears directly
        for category_key, cat_info in HAZARD_CATEGORIES.items():
            if cat_info["label"].lower() in text_lower:
                return category_key, cat_info["label"]
        
        return "", ""

    def _extract_severity(self, text: str) -> str:
        """Extract severity level from text."""
        text_lower = text.lower()
        
        # Check for explicit severity mentions
        for level, patterns in self.SEVERITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return level
        
        # Check for severity keywords in context
        severity_context = re.search(
            r"severity\s*(?:level|:)\s*(\w+)", text_lower
        )
        if severity_context:
            level = severity_context.group(1)
            if level in SEVERITY_LEVELS:
                return level
        
        return ""

    def _extract_description(self, text: str) -> str:
        """Extract hazard description from text."""
        # Look for "Description:" or "3." prefix patterns
        patterns = [
            r"(?:description|3\.)\s*(?:of\s*the\s*hazardous?\s*situation)?\s*[:\.]?\s*(.+?)(?:\n|$)",
            r"description\s*[:\.]\s*(.+?)(?:\n(?:4\.|recommendation)|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                desc = match.group(1).strip()
                if desc:
                    return desc
        
        # If no structured description found, use the hazard context
        # Remove known sections and use remaining text
        cleaned = text
        for section in ["hazard type", "severity", "recommendation", "corrective action"]:
            cleaned = re.sub(
                rf"{section}\s*[:\.]?\s*.+?(?:\n|$)",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 10:
            return cleaned
        
        return ""

    def _extract_recommendation(self, text: str) -> str:
        """Extract recommended corrective action from text."""
        patterns = [
            r"(?:recommended\s*corrective\s*action|recommendation|4\.)\s*[:\.]?\s*(.+?)$",
            r"(?:corrective\s*action|recommend)\s*[:\.]?\s*(.+?)$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                rec = match.group(1).strip()
                if rec:
                    return rec
        
        return ""

    def _parse_unstructured_output(self, raw_output: str) -> List[HazardAssessment]:
        """
        Parse less structured VLM output that may not follow the expected format.
        
        Attempts to identify hazard mentions even when the output format is irregular.
        """
        hazards = []
        lines = raw_output.split("\n")
        
        current_hazard_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line starts a new hazard description
            is_new_hazard = bool(
                re.search(
                    r"(?:hazard|risk|danger|unsafe|violation|safety\s*issue)",
                    line_stripped.lower(),
                )
            )
            
            if is_new_hazard and current_hazard_lines:
                # Parse accumulated lines as a hazard
                hazard = self._parse_hazard_block("\n".join(current_hazard_lines))
                if hazard:
                    hazards.append(hazard)
                current_hazard_lines = [line_stripped]
            elif line_stripped:
                current_hazard_lines.append(line_stripped)
        
        # Parse remaining lines
        if current_hazard_lines:
            hazard = self._parse_hazard_block("\n".join(current_hazard_lines))
            if hazard:
                hazards.append(hazard)
        
        return hazards

    def _extract_hazard_mentions(self, raw_output: str) -> List[HazardAssessment]:
        """
        Last resort: extract any hazard-like mentions from unstructured text.
        
        Creates basic HazardAssessment objects from keyword matches.
        """
        hazards = []
        text_lower = raw_output.lower()
        
        for category_key, patterns in self.HAZARD_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    cat_info = HAZARD_CATEGORIES[category_key]
                    # Extract surrounding context as description
                    match = re.search(pattern, text_lower)
                    start = max(0, match.start() - 50)
                    end = min(len(raw_output), match.end() + 100)
                    context = raw_output[start:end].strip()
                    
                    hazards.append(HazardAssessment(
                        hazard_type=category_key,
                        hazard_label=cat_info["label"],
                        severity=cat_info["severity_default"],
                        description=context,
                        recommendation="Follow standard safety procedures for " + cat_info["label"],
                    ))
                    break  # Only add one assessment per category
        
        return hazards