# 🧬 LargeR: Interactive RNA-Ligand Modeling Agent

<div align="center">
    <img src="docs/fig_sum.png" width="800" alt="LargeR Architecture">
    <br><br>
    <a href="https://platform.deepseek.com/">
        <img src="https://img.shields.io/badge/Powered%20by-DeepSeek-blue?logo=openai&logoColor=white" alt="DeepSeek">
    </a>
    <img src="https://img.shields.io/badge/Python-3.9-green.svg" alt="Python 3.9">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

**LargeR** is an interactive, natural language-based agent for RNA-Ligand interaction modeling. It automates the entire deep learning workflow—from data preparation to model training and self-correcting code generation—allowing researchers to focus on biological insights.

## 🚀 Installation

We strongly recommend using **Mamba** instead of Conda for faster dependency resolution.

### 1. Install Mamba (Optional but Recommended)
If you don't have Mamba installed, you can download **Miniforge** (which includes Mamba) here:
👉 **[Download Miniforge / Mamba](https://github.com/conda-forge/miniforge#download)**

*Already have Conda?* You can just install mamba in your base environment:
```bash
conda install -n base -c conda-forge mamba
```
### 2. Set up Environment
# Create environment using Mamba (faster)
mamba create -n LargeR python=3.9
mamba activate LargeR

# Clone the repository
git clone [https://github.com/623680841/LargeR-main.git](https://github.com/623680841/LargeR-main.git)
cd LargeR-main

### 3. Install Dependencies
```
pip install -r requirements.txt
# Install the package in editable mode
pip install -e .
```
### 🔑 DeepSeek API Configuration
LargeR is powered by DeepSeek-V3. You need an API Key to run the agent.
Step 1: Get Your Key
Register at DeepSeek Open Platform.

Go to API Keys and create a new key.

Copy the key (starts with sk-...).

Step 2: Where to put the Key?
You have two options to configure the key. Option A is recommended for security.

Option A: Environment Variable (Safe & Recommended)
Run this command in your terminal before starting the agent:
# Replace with your actual key
```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

Option B: Edit Config File (Quick Start)
If you prefer, you can paste your key directly into the code.

Open larger/llm_tools.py.

Find the API_KEY variable and paste your key:

```bash
# larger/llm_tools.py


API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
BASE_URL = "[https://api.deepseek.com](https://api.deepseek.com)"
```
💻 Usage
Once installed, simply run the agent:

```bash
python run_agent.py
```
Interactive Workflow
Data Prep: The agent will ask you to prepare a csv file (e.g., datasets/train.csv).

Auto-Coding: The agent writes the PyTorch model for you.

Training: It automatically trains the model and saves the best version as best_model.pth.

Result: Performance metrics (AUC, Accuracy) are reported automatically.

