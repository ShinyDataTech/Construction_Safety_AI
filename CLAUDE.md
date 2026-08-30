# Construction Safety AI — Project Rules for Claude

## Architecture
- Two-stage pipeline: YOLOv11n detection → structured prompt + sVLM inference
- Flat structure — all Python modules live in the project root (no subpackages)
- Streamlit web app (`app.py`) is the sole entry point — run with `streamlit run app.py`
- `VLMInterface` supports two backends: Azure OpenAI (GPT-4o, cloud) and HuggingFace transformers (local sVLMs)
- `ResponseParser` uses regex-based parsing with three fallback levels (structured → unstructured → keyword extraction)

## Coding Conventions
- All modules are standalone files (no `__init__.py`, no packages)
- Use `@st.cache_resource` for heavy resource loaders (detector, VLM, prompt engineer, parser, visualizer)
- Core data types: `Detection`/`DetectionResult` in `detector.py`, `HazardAssessment`/`ParsedResult` in `response_parser.py`
- All configuration lives in `config.py` — do not hardcode constants elsewhere

## VLM Models
- Default is `gpt-4o` (requires `.env` with Azure OpenAI credentials)
- Local models require GPU (CUDA) for reasonable speed; CPU inference is extremely slow
- Gemma-3 4B is the paper's best performer (F1=50.6% with detection guidance)
- Local models are lazy-loaded on first inference call

## Key Files
| File | Role |
|------|------|
| `app.py` | Main Streamlit entry, pipeline orchestration |
| `config.py` | All constants, model configs, hazard categories, prompt templates |
| `detector.py` | YOLOv11n wrapper, spatial relationship computation |
| `vlm_interface.py` | Model loading and inference (Azure OpenAI + HuggingFace) |
| `prompt_engineer.py` | Detection-to-text prompt construction |
| `response_parser.py` | Regex-based VLM output parsing |
| `visualizer.py` | Bounding box and hazard zone image annotation |
| `ui_components.py` | Reusable Streamlit UI components |

## Environment
- Virtual environment: `construction_safety_env/` (do not commit)
- Azure OpenAI config: `.env` file — **DO NOT commit secrets**
- YOLO model weight: `yolo11n.pt` (already committed)
- No automated tests — manual validation via Streamlit UI only

## Git Branches
- `main` — stable application code
- `feature/job-scraper` — active branch (job listing scraper, not merged)
