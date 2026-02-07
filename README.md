# 🧬 LargeR: Interactive RNA-Ligand Modeling Agent

<div align="center">
    <img src="docs/fig_sum.png" width="800" alt="LargeR Architecture">
    <br><br>
    <a href="[https://platform.deepseek.com/](https://platform.deepseek.com/)">
        <img src="[https://img.shields.io/badge/Powered%20by-DeepSeek-blue?logo=openai&logoColor=white](https://img.shields.io/badge/Powered%20by-DeepSeek-blue?logo=openai&logoColor=white)" alt="DeepSeek">
    </a>
    <img src="[https://img.shields.io/badge/Python-3.9-green.svg](https://img.shields.io/badge/Python-3.9-green.svg)" alt="Python 3.9">
    <img src="[https://img.shields.io/badge/License-MIT-yellow.svg](https://img.shields.io/badge/License-MIT-yellow.svg)" alt="License">
</div>

**LargeR** is an interactive, natural language-based agent for RNA-Ligand interaction modeling. It seamlessly integrates data preparation, feature encoding, model training, validation, and visualization into a fully automated workflow driven by Large Language Models (LLMs).

It simplifies the complex deep learning process into a conversation, allowing researchers to focus on biological insights rather than coding details.

## ✨ Key Features

* **🤖 Interactive Agent**: Just type your requirements, and the agent handles the rest.
* **⚡ Mamba-Ready**: Optimized for fast environment setup using Mamba/Miniforge.
* **🧠 LLM-Driven Coding**: Automatically generates, debugs, and optimizes PyTorch model code.
* **📉 Smart Training Pipeline**: 
    * Automated **Train / Val / Test** splitting.
    * Real-time monitoring and **Best Checkpoint** saving strategy.
* **🛠️ Self-Correction**: Automatically detects execution errors and retries with fixed code.

---

## 🚀 Installation

We strongly recommend using **Mamba** (via Miniforge) for faster dependency resolution, although standard Conda works too.

### 1. Install Mamba (Recommended)
If you don't have Mamba installed, download **Miniforge** here:
👉 **[Download Miniforge (Mamba)](https://github.com/conda-forge/miniforge#download)**

*Already have Conda?* You can install mamba in your base environment:
```bash
conda install -n base -c conda-forge mamba
# 1. Create environment (Python 3.9 recommended)
mamba create -n LargeR python=3.9
mamba activate LargeR

# 2. Clone the repository
git clone https://github.com/623680841/LargeR-main.git
cd LargeR-main
pip install -r requirements.txt
# Install the package in editable mode
pip install -e .

🔑 DeepSeek API Configuration
LargeR is powered by DeepSeek-V3 (or DeepSeek-R1). You need an API Key to run the agent.

Step 1: Get your API Key
Register at the DeepSeek Open Platform.

Navigate to API Keys in the left menu.

Click Create API Key and copy the string starting with sk-....

Step 2: Configure the Key
You can configure the key in one of the two ways below:

Option A: Using Environment Variable (Recommended & Secure)
Run this in your terminal before starting the agent. This keeps your key safe and out of the code.

Bash
export DEEPSEEK_API_KEY="sk-your_actual_key_here"
Option B: Edit the Config File
Alternatively, you can paste your key directly into the configuration file.

Open larger/llm_tools.py.

Find the API_KEY variable and update it:

Python
# larger/llm_tools.py
# ⚠️ WARNING: Do not push this file to GitHub if you paste your real key here!
API_KEY = "sk-your_actual_key_here" 
BASE_URL = "https://api.deepseek.com"
💻 Usage
Once the installation and API setup are complete, start the interactive agent:

Bash
python run_agent.py
📝 Data Preparation
The agent will guide you to prepare a training dataset. You need a CSV file (e.g., datasets/train.csv) with the following columns:

ligand: Name of the ligand.

label: 1 (positive) or 0 (negative).

rna_sequence: Full RNA sequence string (e.g., "AUGCC...").

region_mask: A string representation of a list marking the region (e.g., "[0, 0, 1, 1, 0]").

🔄 Workflow Example
Interaction: Type yes when your data is ready.

Auto-Coding: The agent generates the PyTorch model architecture.

Training: Training starts automatically. The agent monitors the loss and saves the best model as best_model.pth.

Evaluation: The agent loads the best model and reports AUC/Accuracy on the test set.

<div align="center"> Designed for RNA Research | Powered by LLM </div>
