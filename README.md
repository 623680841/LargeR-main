# 🧬 LargeR: Interactive RNA-Ligand Modeling Agent

<div align="center">
    <img src="docs/fig_sum.png" width="800" alt="LargeR Architecture">
    <br><br>
    <a href="[https://platform.deepseek.com/](https://platform.deepseek.com/)">


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
