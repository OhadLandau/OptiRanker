
# **OptiRanker: A Framework for Optimizing Pre-Clinical Drug Prioritization**

## **Overview**
OptiRanker is a Python-based tool designed to simulate, rank, and optimize pre-clinical drug prioritization algorithms. This framework systematically evaluates the performance of prioritization algorithms under various experimental conditions. By identifying the smallest subsets of drugs and individuals required to differentiate between algorithms, OptiRanker minimizes experimental costs while maximizing statistical power.

### **Key Features**
- Simulate the ranking process of drugs and predictors with varying levels of noise.
- Optimize subsets of drugs and individuals to reproduce accurate algorithm rankings.
- Evaluate and compare the robustness of WINTHER, SIMS, and DDPP drug prioritization algorithms.
- Visualize results with detailed heatmaps, PCA plots, and statistical analyses.

---

## **Applications**
- **Simulated Trial Optimization**: Reduce experimental costs for in vivo validation of drug prioritization algorithms.
- **Empirical Evaluation**: Analyze accuracy decay in noisy predictors.
- **Real-World Validation**: Use CCLE datasets for IC50 predictions and evaluate against WINTHER, SIMS, and DDPP algorithms.

---

## **Technical Details**

### **Languages and Libraries**
- **Core Language**: Python (version 3.8+)
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly, Dash, PySimpleGUI

### **Dependencies**
- **Datasets**: Cancer Cell-Line Encyclopedia (CCLE) transcriptomics and IC50 data
- **Predictor Scores**: Algorithm outputs from WINTHER, SIMS, and DDPP

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/OptiRanker.git
cd OptiRanker
```

### **2. Set Up a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### **3. Install Required Libraries**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **4. Download CCLE Dataset**
- Download transcriptomics and IC50 datasets from the official [CCLE website](https://depmap.org/portal/).
- Save the datasets in a folder named `data/` within the repository.

---

## **Repository Structure**
```
OptiRanker/
│
├── data/                       # Place for CCLE datasets (IC50 values, transcriptomics)
├── scripts/
│   ├── FullSimulation.py       # Simulation-based pipeline
│   ├── InputData.py            # Real-world dataset-based pipeline
│   └── utils.py                # Utility functions shared between scripts
│
├── results/                    # Results folder for outputs, heatmaps, etc.
├── README.md                   # Documentation
├── requirements.txt            # Dependencies list
└── LICENSE                     # License information
```

---

## **Usage**

### **1. Full Simulation Pipeline**
This script generates predictors with varying levels of noise and evaluates algorithm rankings based on simulated data.

```bash
python scripts/FullSimulation.py
```

### **2. Real-World Dataset Pipeline**
This script processes user-provided CCLE datasets, evaluates algorithm performance, and visualizes the results.

```bash
python scripts/InputData.py
```

---

## **Workflow**
1. **Simulate or Import Data**:
   - Full simulation: Generate randomized rankings for individuals and drugs.
   - Real-world input: Use CCLE data for predictions and IC50 comparisons.

2. **Noise Injection**:
   - Inject varying levels of noise into the rankings to simulate predictor variability.

3. **Subset Optimization**:
   - Identify the smallest subsets of drugs and individuals with high correlation to full dataset rankings.
   - Use standard deviation metrics to prioritize subsets.

4. **Algorithm Evaluation**:
   - Compare WINTHER, SIMS, and DDPP outputs against IC50 ground truth.

5. **Visualization**:
   - Generate heatmaps, boxplots, and PCA plots to summarize results.

---

## **Example Outputs**
- **Heatmaps**: Visualize correlations for varying subsets of drugs and individuals.
- **Boxplots**: Show the impact of noise on predictor accuracy.
- **PCA Clustering**: Evaluate algorithm separation in feature space.

---

## **Visualization Example**
Example output from a PCA plot comparing WINTHER, SIMS, and DDPP algorithms:

![PCA Clustering Example](results/pca_plot_example.png)

---

## **Contributing**

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

### **Linting**
```bash
# Run linting to ensure code quality
flake8 scripts/
```
