🧬 LargeR: Interactive RNA-Ligand Modeling Agent
<div align="center"> <img src="docs/fig_sum.png" width="800" alt="LargeR Architecture">



<a href="https://platform.deepseek.com/"> <img src="https://img.shields.io/badge/Powered%20by-DeepSeek-blue?logo=openai&logoColor=white" alt="DeepSeek"> </a> <img src="https://img.shields.io/badge/Python-3.9-green.svg" alt="Python 3.9"> <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"> </div>

LargeR provides an interactive, natural language-based pipeline that enables region-level RNA-ligand modeling. It seamlessly integrates data preparation, feature encoding, model training, validation, and visualization into a fully automated workflow driven by Large Language Models (LLMs).

It simplifies the complex deep learning process into a conversation, allowing researchers to focus on biological insights rather than coding details.

✨ Key Features
🤖 Interactive Agent: Just type your requirements, and the agent handles the rest.

🧠 LLM-Driven Coding: Automatically generates, debugs, and optimizes PyTorch model code.

📉 Smart Training Pipeline:

Automated Train (70%) / Val (15%) / Test (15%) splitting.

Real-time monitoring and Best Checkpoint saving strategy.

🛠️ Self-Correction: Automatically detects execution errors and retries with fixed code.

🚀 Installation
1. Set up the Environment
We recommend using Conda to manage dependencies.

Bash
conda create -n LargeR python=3.9
conda activate LargeR
git clone git@github.com:kai3171/LargeR.git
cd LargeR
2. Install Dependencies
Bash
pip install -r requirements.txt
# Install the package in editable mode
pip install -e .
🔑 DeepSeek API Configuration (Crucial!)
LargeR is powered by DeepSeek (DeepSeek-V3 or DeepSeek-R1). To run the agent, you need to configure your API Key.

Step 1: Get your API Key
Go to the DeepSeek Open Platform.

Log in and navigate to API Keys.

Click Create API Key and copy the string starting with sk-....

Step 2: Configure the Key
You can configure the key in one of the two ways below:

Option A: Using Environment Variable (Recommended) Run this in your terminal before starting the agent:

Bash
export DEEPSEEK_API_KEY="your_sk_key_here"
Option B: Edit the Config File Open the file larger/llm_tools.py (or your specific config file), find the API_KEY variable, and paste your key there:

Python
# larger/llm_tools.py
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
BASE_URL = "https://api.deepseek.com"
💻 Usage
Once the installation and API setup are complete, start the interactive agent:

Bash
python run_agent_demo.py
Workflow Example
Data Prep: The agent will guide you to prepare a CSV file (datasets/train.csv).

Confirmation: Type yes when data is ready.

Training: The agent automatically generates the model and starts training.

Result: The best model is saved as best_model.pth, and performance metrics (AUC, Accuracy) are reported automatically.

<div align="center"> Designed for RNA Research | Powered by LLM </div>