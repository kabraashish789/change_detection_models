# Change Detection Models

This repository contains implementations of several **deep learning–based change detection algorithms** for remote sensing and computer vision tasks.  
Each model is organized into its own folder with dataset handling, training, and utility scripts.

---

## 📂 Repository Structure

change_detection_models/

attunet/   #Attention U-Net

raunet/   #Residual Attention U-Net
unetpp/   #U-Net++
changeformer/   #ChangeFormer (Transformer-based model)
bit/   #BIT (Bitemporal Image Transformer)

Each subfolder contains:
- `dataset.py` – Dataset loading and preprocessing  
- `model.py` – Model architecture  
- `train.py` – Training pipeline  
- `utils.py` – Helper functions  
- `Dockerfile` – For reproducible builds  
- `requirements.txt` – Python dependencies  

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/kabraashish789/change_detection_models.git
cd change_detection_models
2. Build the Docker image
Each model folder has its own Dockerfile. Example for Attention U-Net:

bash
Copy code
cd attunet
docker build -t attunet .
3. Run the container
bash
Copy code
docker run -it --rm attunet
4. Train the model
Inside the container:

bash
Copy code
python train.py
🧩 Available Models
Attention U-Net (attunet/) – U-Net with attention gates for improved feature selection

Residual Attention U-Net (raunet/) – Combines residual connections and attention

U-Net++ (unetpp/) – Nested U-Net with dense skip connections

ChangeFormer (changeformer/) – Transformer-based architecture for change detection

BIT (bit/) – Bitemporal Image Transformer

📊 Model Comparison
The following table compares the performance of different change detection models implemented in this repository:

Model	Dice Score	Accuracy (%)
Attention U-Net	0.9483	99.94
Residual Attention U-Net	0.9525	99.95
U-Net++	0.9238	99.91
ChangeFormer	0.9106	99.89
BIT	0.8393	99.81

⚡ Residual Attention U-Net achieved the highest Dice Score and Accuracy among the tested models.

📦 Requirements
Each model has a requirements.txt with its dependencies. To install locally (without Docker):

bash
Copy code
pip install -r requirements.txt
📊 Datasets
You can plug in your own change detection dataset (bi-temporal image pairs + labels).
Update dataset.py in the respective model folder to point to your dataset.

📧 Contact
Created by Ashish Kabra – feel free to reach out for questions or collaborations.
