2. Set up Environment
3. Install Dependencies
🔑 DeepSeek API Configuration
LargeR is powered by DeepSeek-V3 (or DeepSeek-R1). You need an API Key to run the agent.

Step 1: Get your API Key
Register at the .

Navigate to API Keys in the left menu.

Click Create API Key and copy the string starting with sk-....

Step 2: Configure the Key
You can configure the key in one of the two ways below:

Option A: Using Environment Variable (Recommended & Secure)
Run this in your terminal before starting the agent. This keeps your key safe and out of the code.

Option B: Edit the Config File
Alternatively, you can paste your key directly into the configuration file.

Open larger/llm_tools.py.

Find the API_KEY variable and update it:

💻 Usage
Once the installation and API setup are complete, start the interactive agent:

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
