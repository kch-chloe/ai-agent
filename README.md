# IFTE0001 Group D — AI Analyst Agents in Asset Management

This repository contains Group D’s **AI Analyst Agents** (Fundamental + Technical) for IFTE0001.  
The **main runnable demo** for the group submission is the **Technical Agent A (Main)** located in:

- `Ma_Suet_Nam_Technical_Agent(Agent_A)/`

`run_demo.py` (in the repo root) executes this main agent end-to-end.

---

## Repository Structure

High-level folders (each member’s contribution is separated):

- `Ma_Suet_Nam_Technical_Agent(Agent_A)/` ✅ **Main Technical Agent (used by run_demo.py)**
- `Ching_Hei_Chloe_Keung_Technical_Agent(Agent_B)/`
- `Ying_Shek_Isaac_Ho_Technical_Agent(Agent_C)/`
- `GUYU ZHANG Fundemental analysis/`
- `Jingjing Zhang Fundamental Analyst Agent/`
- `Wanyue Yao AI Agent/`
- `Xinyan Chen fundamental AI agent/`
- `Zhaoyu Wang fundamnetal analysis/`

Root-level files:
- `run_demo.py` — **main demo runner**
- `requirements.txt` — Python dependencies
- `.gitignore`

---

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Azure OpenAI Setup (Required)

run_demo.py requires an Azure OpenAI connection via environment variables.

Create a file named .env in the repo root (same folder as run_demo.py) with:

```bash
AZURE_ENDPOINT=...
AZURE_API_KEY=...
AZURE_API_VERSION=...
AZURE_DEPLOYMENT_NAME=...
```

### 3) Run the Main Demo

From the repo root:

```bash
python run_demo.py
```

run_demo.py will run the Technical Agent A (Main) implementation under: `Ma_Suet_Nam_Technical_Agent(Agent_A)/`

### 4) Outputs
Outputs will be located in `OUTPUT/` in repo root

Expected outputs include:
- downloaded/cleaned OHLCV data
- computed indicators
- backtest results (equity curve, drawdown, trade markers, metrics)
- an LLM-generated trade note (using the Azure deployment)




