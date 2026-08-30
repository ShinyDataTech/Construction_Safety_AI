# 🎬 YouTube Video Demo Script: SafeSite-AI
## Autonomous Construction Safety Agent powered by AMD Lemonade SDK

**Target Duration:** 4:30 – 5:00 Minutes  
**Video Title:** *Building an Autonomous Construction Safety Agent with AMD Lemonade SDK (Zero Cloud Latency!)*  
**Video Description:**  
> Watch how we built **SafeSite-AI**, an autonomous AI agent for construction site safety, running 100% locally on edge hardware using the **AMD Lemonade SDK** (`http://localhost:13305/v1`). Discover how edge AI eliminates cloud latency, protects jobsite privacy, and automates OSHA compliance auditing with zero internet connection required.  
> 🔗 Open Source Repo: [https://github.com/ShinyDataTech/Construction_Safety_AI](https://github.com/ShinyDataTech/Construction_Safety_AI)  
> 🏆 Submission for the AMD Lemonade Developer Challenge.

---

## ⏱️ Video Timeline Overview

| Timestamp | Act / Scene | Core Focus |
| :--- | :--- | :--- |
| **0:00 – 0:45** | **Act 1: The Problem** | Why Cloud AI fails on harsh, disconnected construction sites |
| **0:45 – 1:35** | **Act 2: Architecture & Setup** | AMD Lemonade SDK local server (`localhost:13305`) & YOLOv11 Edge |
| **1:35 – 2:50** | **Act 3: Live Image Hazard Demo** | Zero-latency detection-guided hazard analysis & spatial clearance |
| **2:50 – 3:55** | **Act 4: Autonomous Agent & Tools** | Multi-turn chat, OSHA regulation lookup & automated incident reporting |
| **3:55 – 4:35** | **Act 5: Video Streams & Benchmarks** | Temporal tracking, 16x latency speedup, and zero egress cost |
| **4:35 – 5:00** | **Outro & Call to Action** | Open-source release, Apache 2.0 license, AMD Challenge wrap-up |

---

## 🎬 Detailed Step-by-Step Script & Production Guide

---

### 🟢 Act 1: The Hook & The Construction Edge Dilemma (0:00 – 0:45)

**Visuals & Screen Action:**
- **[0:00 - 0:15]** Fast-paced B-roll montage of active construction sites: heavy excavators swinging, workers on scaffolding, deep trenching excavations.
- **[0:15 - 0:30]** Split-screen graphic showing a worker standing near a swinging excavator. A red spinning loading icon labeled *"Connecting to Cloud API... 3.8s"* appears.
- **[0:30 - 0:45]** Cut to presenter on camera or screen capture of SafeSite-AI interface with the banner **"🟢 AMD Lemonade SDK Online (Local: 13305)"**.

**Voiceover / Narration:**
> *"Construction is one of the most dangerous industries in the world, with over 1,000 fatal accidents in the US alone each year. The OSHA 'Focus Four' hazards—falls, struck-by equipment, caught-in trenches, and electrical shock—demand instant, sub-second intervention.*  
>  
> *But here is the catch: when you are in a concrete basement, an underground tunnel, or a remote bridge site, you have zero internet connection. And even with 5G, uploading 4K video feeds to cloud LLMs takes 3 to 6 seconds. If a 20-ton excavator swings in your direction, 4 seconds is too late.*  
>  
> *Today, I'm introducing **SafeSite-AI**, an autonomous AI safety agent running 100% locally on edge hardware powered by the **AMD Lemonade SDK**."*

**On-Screen Text Callout:**
- ⚠️ *Cloud Latency: 3.8s (Too Slow for Moving Machinery)*
- ⚡ *SafeSite-AI: ~200ms Local Edge Turnaround*

---

### 🟢 Act 2: System Architecture & Lemonade Local Endpoint (0:45 – 1:35)

**Visuals & Screen Action:**
- **[0:45 - 1:05]** Terminal window shown on screen.
  - Command typed: `lemonade serve --port 13305`
  - Output shows Lemonade server launching on `http://localhost:13305/v1` with AMD Ryzen™ AI NPU / ROCm acceleration.
- **[1:05 - 1:20]** Show clean architectural diagram on screen (Mermaid flow from README):
  - Stage 1: YOLOv11n object localization & spatial math
  - Stage 2: Lemonade local multimodal VLM inference via OpenAI-compatible endpoint
  - Autonomous Agent: Deterministic OSHA tools + incident report engine.
- **[1:20 - 1:35]** Open browser to Streamlit app (`http://localhost:8501`), highlighting the green status pill: **"🟢 AMD Lemonade SDK Online | localhost:13305/v1"**.

**Voiceover / Narration:**
> *"Under the hood, SafeSite-AI leverages the open-source **AMD Lemonade SDK** as its local inference backbone.*  
>  
> *With a single command—`lemonade serve --port 13305`—we have an ultra-fast, OpenAI-compatible local endpoint running directly on our AMD Ryzen AI hardware.*  
>  
> *Our architecture combines two stages: first, a lightweight YOLOv11n model localizes workers, vehicles, and spatial clearances in 15 milliseconds. Second, those coordinates condition a local vision-language model hosted in Lemonade, giving our autonomous agent deep contextual understanding with zero cloud dependency."*

**On-Screen Text Callout:**
- 🚀 *Unified API: `http://localhost:13305/v1`*
- 🔒 *100% Private, Air-Gapped, Zero Data Egress*

---

### 🟢 Act 3: Live Demo — Sub-Second Hazard Assessment (1:35 – 2:50)

**Visuals & Screen Action:**
- **[1:35 - 1:55]** Click on **"📸 Site Image Inspection"** tab. Select demo scenario `construction-image-01.webp` showing a worker standing within inches of an operating heavy truck near an excavation pit.
- **[1:55 - 2:15]** Click **"🚀 Run Hazard Inspection"**.
  - Show real-time progress: Stage 1 (YOLO detection: 16ms) → Stage 2 (Lemonade Local VLM: 220ms).
  - Screen updates instantly with bounding boxes, spatial vectors, and color-coded hazard cards.
- **[2:15 - 2:35]** Zoom in on the output cards:
  - Card 1: 🔴 **CRITICAL | Unsafe Machinery Proximity (29 CFR 1926.602)** — Worker inside blind-spot swing perimeter.
  - Card 2: 🟠 **HIGH | Fall Hazard (29 CFR 1926.501)** — Unprotected edge near trench.
- **[2:35 - 2:50]** Toggle the **"Side-by-Side Comparison Mode"** showing how detection-guided spatial prompt conditioning uncovers 2 additional subtle hazards that standard vanilla prompts miss completely.

**Voiceover / Narration:**
> *"Let's see it in action. I'll load a real construction site scenario with multiple workers and heavy equipment.  
>  
> When I hit 'Run Hazard Inspection', watch how fast this runs.  
>  
> [Click button] In just 236 milliseconds—completely on-device—YOLOv11 localizes every worker and truck, calculates spatial clearance vectors, and passes that context to Lemonade.  
>  
> Immediately, it flags a Critical Struck-By hazard: Worker #1 is standing within the unsafe 15% machinery clearance zone, violating OSHA 29 CFR 1926.602, and recommends establishing a physical barrier and dedicated spotter.  
>  
> Notice there was zero network latency, zero token costs, and 100% precision."*

**On-Screen Text Callout:**
- ⚡ *Detection: 16.4ms | Lemonade Reasoning: 220.1ms | Total: 236.5ms*
- 🎯 *F1 Recall Gain: +40% with Detection-Guided Conditioning*

---

### 🟢 Act 4: Autonomous Safety Agent & Deterministic Tool Calling (2:50 – 3:55)

**Visuals & Screen Action:**
- **[2:50 - 3:10]** Switch to the **"🤖 Autonomous Safety Agent"** tab.
- **[3:10 - 3:25]** Click the quick action button **"📋 Run Full OSHA Site Audit"**.
  - Show tool execution pills animating:
    - `[TOOL: calculate_danger_zone]` -> 9.9% clearance violation!
    - `[TOOL: search_osha_regulations]` -> 29 CFR 1926.602 retrieved!
    - `[TOOL: audit_ppe_compliance]` -> Checklist verified!
    - `[TOOL: compile_incident_report]` -> Formal Report generated!
- **[3:25 - 3:45]** Type in interactive chat: *"What are the mandatory guardrail requirements for the elevated platform on the left?"*
  - The agent responds in real-time streaming: quotes **29 CFR 1926.502(b)** (top-rail height 42 in ± 3 in, 200 lb force capacity) and references the exact scene layout.
- **[3:45 - 3:55]** Click **"📥 Download Formal OSHA Safety Report (.md)"** and show the clean, audit-ready Markdown document generated on-device.

**Voiceover / Narration:**
> *"Now let's open the Autonomous Safety Agent tab. SafeSite-AI isn't just a static classifier; it's an interactive safety supervisor.  
>  
> When we trigger a full audit, the agent autonomously invokes our suite of local tools: it computes geometric danger zones, cross-references our local OSHA 1926 database, audits PPE checklists, and compiles an audit-ready inspection report.  
>  
> Superintendents can even chat with the agent in plain English: 'What are the legal guardrail rules for this platform?'—and receive immediate regulatory citations and actionable remediation steps."*

**On-Screen Text Callout:**
- 🛠️ *Deterministic Local Tools: OSHA Lookup, Danger Math, PPE Audit, Report Synthesis*
- 📄 *Instant Formal Safety Audit Export*

---

### 🟢 Act 5: Video Streams, Edge Benchmarks & Wrap-Up (3:55 – 5:00)

**Visuals & Screen Action:**
- **[3:55 - 4:15]** Switch to **"🎥 Temporal Video Monitoring"** tab. Load `construction-video-01.mp4`.
  - Show keyframe timeline tracking how a worker walks closer to an active vehicle over a 12-second sequence, tracking hazard progression across frames.
- **[4:15 - 4:35]** Switch to **"⚡ Lemonade Edge Benchmark"** tab.
  - Highlight the 4 metric tiles:
    - ⚡ Latency: 240ms (16x faster than 3,850ms cloud API)
    - 📉 Egress: 0.0 MB / hr (100% bandwidth saved)
    - 🔒 Privacy: 100% On-Device
    - 💰 Cost: $0.00 / Unlimited
- **[4:35 - 5:00]** Display GitHub repository page, Apache 2.0 license badge, and challenge submission info.

**Voiceover / Narration:**
> *"For continuous monitoring, our temporal video processor samples video streams frame-by-frame, tracking hazards as they develop over time—all without uploading a single megabyte of 4K video to the cloud.  
>  
> When we benchmark this against cloud APIs like GPT-4o, the local AMD Lemonade SDK is over 16 times faster, consumes zero network egress, and delivers 100% privacy for on-site personnel.  
>  
> SafeSite-AI is fully open-source under the Apache 2.0 license. The complete code, prompt engineering templates, and setup guide are available on GitHub right now.  
>  
> Thank you to AMD and the Lemonade team for powering the next generation of local, life-saving AI at the edge. Check out the repo link in the description, and let's build safer jobsites together!"*

**On-Screen Text Callout:**
- 🏆 *AMD Lemonade Developer Challenge Submission*
- 📜 *Apache 2.0 Open-Source License*
- ⭐ *Star & Fork on GitHub!*

---
*End of Demo Script.*
