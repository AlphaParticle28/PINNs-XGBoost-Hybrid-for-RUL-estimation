# PINNs-XGBoost-Hybrid-for-RUL-estimation

A physics-informed deep learning framework for battery life prediction, leveraging a **hybrid Physics-Informed Neural Network (PINN) and XGBoost model** to integrate electrochemical dynamics with data-driven patterns.

Achieves end-to-end **Remaining Useful Life (RUL)** forecasting by learning battery-specific degradation parameters from the NASA Prognostics dataset.

---

## Project Overview
This system provides high-fidelity predictions of battery capacity degradation by enforcing the **Arrhenius degradation law** within a neural network, creating a robust model that generalizes well. The hybrid approach uses the PINN as an intelligent feature extractor to enhance a powerful XGBoost model.

### **Key Capabilities**
- **Hybrid Model Prediction** – Combines a PINN with an XGBoost model to achieve **99.8% R² accuracy**
- **Physics-Informed Learning** – Embeds the Arrhenius degradation equation directly into the model's loss function to ensure physically plausible predictions.
- **Battery-Specific Embeddings** – Learns a unique "fingerprint" for each battery to capture individual manufacturing and chemical variations.
- **Dynamic Parameter Estimation** – A ParameterLearner network estimates physical constants (k, n, Ea) in real-time based on the battery's state.
- **Comprehensive Evaluation** – Validated using R², MAPE, and NRMSE metrics for a complete performance overview.

---

## Directory Structure
```plaintext
PINNs-XGBoost-Hybrid-for-RUL-estimation/
│
├── Dataset/
│   ├── B0005.mat              # NASA battery dataset for battery B0005
│   ├── B0006.mat              # NASA battery dataset for battery B0006
│   ├── B0007.mat              # NASA battery dataset for battery B0007
│   ├── B0018.mat              # NASA battery dataset for battery B0018
|   └── Dataset_Info.txt       # Information about the dataset used
│
├── Scripts/
|   ├── Hybrid.ipynb           # Combined structure of the PINNs + XGBoost        
|   ├── NN_Learner.py          # Defines the neural network for learning
|   ├── Train.ipynb            # For model training
|   ├── ann_model.pth          # Saved weights of the neural network for reuseability
|   ├── emb_model.pth          # Saved battery enbeddings for future reference
|   ├── param_model.pth        # Saved physical parameters (activation energy, rate constant and reaction order)  
|   ├── parseData.ipynb        # For parsing raw data for feature engineering    
|   ├── physics.py             # For defining the physical constraints and physics losses
|   ├── RUL.ipynb              # For estimating teh RUL using all the models trained
|   ├── test_data.csv          # Test split of the cleaned dataset
|   ├── train_data.csv         # Train split of the cleaned dataset
|   └── xgb_hybrid_model.json  # Sved CGBoost weights
└── README.md                  # Project documentation
```

---

## Workflow Description

### **1. Data Processing**
**File:** `parseData.ipynb`
- Extracts dicharge cycle data from the raw NASA (`.mat`) files.
- Computes key features like average and normalized temperature and current.
- Splits data into training and testing sets, stratified by battery ID.

---

### **2. PINN Model Development**
**File:** `NN_Learner.ipynb`, `physics.py` and `Train.ipynb`
- A core ANN predicts the capacity (`C`) as a function of time (`t`), current (`I`) and Temperature (`T`), as these are the explicit features required for the Arrhenius Equation.
- The `BatteryEmbedding` layer creates a 6-d-dimensional vector for each unique battery
- A `ParameterLearner` network uses the predicted capacity and battery embedding to estimate physical degradation parameters (Ea, k, and n).
- The model is trained using a combination of Adam and L-BFGS optimizers for stability and precision.

---

### **3. Hybrid XGBoost Model**
**File:** `Hybrid.ipynb`
- Loads the trained PINN model weights
- Uses the PINN to create battery embeddinng and the physical parameters for the dataset.
- Combines the original features with these new physics-informed features.
- Trains an XGBRegressor on this newly formed dataset to make final, highly accurate capacity predictions, followed by saving the model.

### **4. RUL Estimation**
**File:** `RUL.ipynb`
- Loads the trained PINN model and XGBoost weights.
- Iterates through the number of cycles the capacity remains above a certain threshold capacity
- Compares the predictied and the ground truth RUL values, followed by plotting the forecast.
---

## Data Management

### **Primary Datasets**
- **NASA Prognostics Data (`B0005.mat`, etc.)** – Contains voltage, current, temperature, and capacity measurements for multiple discharge cycles across four batteries.

### **Model Artifacts**
- `ann_model.pth` – Trained weights for the main capacity prediction network.
- `emb_model.pth` – Trained weights for the battery embedding layer.
- `param_model.pth` – Trained weights for the parameter learner network.
- `xgb_hybrid_model.json` - Trained XGBoost model for the hybrid.
---

## Technical Architecture

### **Neural Network Specifications**
| Property        | Details |
|-----------------|---------|
| **Architecture** | 3 hidden layers with 32 neurons each |
| **Input Features** | `t`, `I`, `T`, `battery_embedding` |
| **Activations** | Tanh |
| **Optimizer** | Adam(initial) → L-BFGS(fine-tuning) |
| **Physics Law** | Arrhenius Degradation Equation |
| **Performance** | R² = 0.997, MAPE = 0.46% |

---

## Prerequisites

### **System Requirements**
- Python **3.10+**
- Jupyter Notebook
- PyTorch
- scikit-learn, pandas, numpy, scipy, xgboost

### **Required Libraries**
```bash
pip install torch pandas numpy scipy scikit-learn xgboost
```

---

## Installation Instructions

### **1. Environment Setup**
```bash
# Clone repository
git clone https://github.com/AlphaParticle28/PINNs-XGBoost-Hybrid-for-RUL-estimation
cd PINNs-XGBoost-Hybrid-for-RUL-estimation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch pandas numpy scipy scikit-learn xgboost
```

### **2. Data Preparation**
- Create a `Dataset/` folder in the root directory.
- Place all four `.mat` files (`B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`) inside the `Dataset/` folder.
---

## How to Run

### **Step 1 – Preprocess Data**
```bash
jupyter notebook parseData.ipynb
```

### **Step 2 – Train PINN Model**
```bash
jupyter notebook Train.ipynb
```

### **Step 3 – Train and Evaluate the Hybrid XGBoost Model**
```bash
jupyter notebook XGBoost_Hybrid.ipynb
```
### **Step 4 – Evaluate RUL**
```python
from RUL_Estimator import predict_rul

battery_to_predict = 3 # Or the battery of your choice
start_cycle_to_predict = 50 # Or the start cycle of your choice
EOL_Threshold = 1.4 # Or the threshold capacity of your choice
full_df = your_production_dataset

predicted_rul, forecast_curve = predict_rul(
    battery_id=battery_to_predict,
    start_cycle=start_cycle_to_predict,
    eol_threshold=EOL_Threshold
)

print(f"Prediction for Battery ID {battery_to_predict} starting from cycle {start_cycle_to_predict}:")
print(f"Predicted RUL: {predicted_rul} cycles")
```
---

## Key Features
| Feature | Description | Benefit |
|---------|-------------|---------|
| **Hybrid Model** | PINN feature extraction + XGBoost prediction. | Achieves 99.8% R² accuracy. |
| **Physics-Informed** | Embeds Arrhenius Law into the loss function. | Ensures physically realistic predictions and better generalization. |
| **Battery Embeddings** | Learns a unique vector for each battery. | Captures individual battery characteristics and manufacturing variations. |
| **Dynamic Parameters** | Estimates physical constants in real-time. | Provides insight into the battery's current degradation state. |
| **Two-Stage Optimizer** | Uses Adam and L-BFGS for robust training. | Combines fast convergence with high-precision optimization. |

---

## Model Performance
- **PINN Performance**: R² = 0.997, MAPE = 0.46%
- **Final Hybrid Performance**: R² = 0.998, MAPE = 0.39%
- **Data Source**: NASA Prognostics Battery Dataset (636+ cycles)

---

## Technical Implementation

### PINN Composite Loss Function
The model is trained by minimizing a combination of data error and the physics residual:

$$
L_{\text{total}} = \lambda_{\text{data}} \cdot L_{\text{data}} + \lambda_{\text{physics}} \cdot L_{\text{physics}}
$$

---

### Arrhenius Degradation Equation
The physics loss  $$\( L_{\text{physics}} \)$$ is the residual of this differential equation, where  
$$\( \frac{dC}{dt} \)$$  is computed via automatic differentiation:

$$
\frac{dC}{dt} = -k \cdot I^n \cdot e^{-\frac{E_a}{R T}}
$$

---

### RUL Calculation
The function predicts a battery’s **Remaining Useful Life (RUL)** by forecasting its capacity $$\( C \)$$ across cycles until it drops below the **End of Life (EOL)** threshold.

#### 1. Initialization
At start cycle $$\( c_s \)$$:

$$
C_{\text{start}} = C(c_s)
$$

PINN models generate physics-based features:

$$
(k, n, E_a, \text{embeddings}) = f_{\text{PINN}}(x_{\text{start}})
$$

#### 2. Normalization
For each future cycle $$\( c_f \)$$:

$$
t_{\text{norm}} = \frac{c_f - c_{\min}}{c_{\max} - c_{\min}}
$$

Form hybrid input:

$$
X = [t_{\text{norm}}, T_{\text{norm}}, I_{\text{norm}}, \text{embeddings}, k, n, E_a]
$$

#### 3. Capacity Forecast
Predict next capacity using XGBoost:

$$
C_{\text{new}} = f_{\text{XGB}}(X)
$$

Repeat until:

$$
C_{\text{new}} \leq C_{\text{EOL}}
$$

#### 4. Remaining Useful Life

$$
\text{RUL} = c_f - c_s
$$


---
