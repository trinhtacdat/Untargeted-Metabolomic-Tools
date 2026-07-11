# 🌿 Herbal Metabolomics Analyzer

A comprehensive desktop GUI application for the untargeted analysis of metabolomics data, specifically designed for natural product and herbal medicine research. This tool provides an end-to-end workflow, from raw MZmine quantification data to biomarker discovery and visualization.

---

## ✨ Key Features

- **Data Import & Management**:
  - Load quantification data directly from MZmine CSV files.
  - Automatically detect sample structure, including technical replicates and QC samples.
  - Flexible sample renaming and group management with customizable color-coding.

- **Advanced Preprocessing**:
  - Automatic averaging of technical replicates while preserving QC samples.
  - Multiple feature filtering methods: QC-RSD, detection rate, intensity, and variance (IQR).
  - Missing value imputation (LOD).
  - A suite of normalization options: TIC, Median, Log Transform, Pareto, and Auto-scaling.

- **Multivariate Analysis**:
  - **Principal Component Analysis (PCA)**: Interactive 2D score plots with confidence ellipses, sample labels, and statistical validation (PERMANOVA, PERMDISP).
  - **Partial Least Squares Discriminant Analysis (PLS-DA)**: Supervised modeling with score plots, VIP scores for feature importance, and robust model validation (R²X, R²Y, Q²).
  - **Permutation Testing**: Rigorously validate your PLS-DA model to prevent overfitting.

- **Biomarker Discovery**:
  - **Univariate Screening**: Interactive Volcano plots to identify statistically significant features (p-value, fold-change). Supports FDR correction.
  - **Random Forest**: Powerful machine learning for feature selection, with OOB error, cross-validation accuracy, and ROC curve analysis.
  - **Consensus Analysis**: Export features that are identified as significant by Volcano, PLS-DA, and Random Forest simultaneously.

- **Rich Visualization**:
  - **Clustered Heatmaps**: Visualize the abundance of top biomarkers across samples.
  - **Comparative Analysis**: Generate quantitative Venn diagrams and UpSet plots to find common and unique features between groups.
  - **Feature Viewer**: Inspect the distribution of any individual feature with box plots, violin plots, and data points.
  - **Spectrum Viewer**: Load MGF files to search and visualize MS/MS spectra for features of interest.

- **Session & Export**:
  - **Save/Load Session**: Save your entire analysis state—including data, groups, and results—to a single file and resume later.
  - **Comprehensive Export**: Export all results, tables, and high-resolution plots (PNG, PDF report) for publication and further analysis.

---

## 📦 Installation

This application is built with Python and requires several packages to run. Using a virtual environment is highly recommended to avoid conflicts with other projects.

### 1. Prerequisites
- [Python 3.8+](https://www.python.org/downloads/)

### 2. Set Up a Virtual Environment (Recommended)

Open your terminal or command prompt and run:

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Packages

The easiest way to install all dependencies is by using the provided `requirements.txt` file.

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn openpyxl scikit-bio matplotlib-venn
```

**Package Roles:**
- `pandas`, `numpy`: Core data handling and numerical operations.
- `scikit-learn`, `scipy`: Statistical analysis and machine learning (PCA, PLS-DA, RF, t-tests).
- `matplotlib`, `seaborn`, `matplotlib-venn`: Data visualization and plotting.
- `openpyxl`: Required for exporting data to Excel (`.xlsx`) files.
- `scikit-bio`: Required for PERMANOVA and PERMDISP p-value calculations in the PCA tab.

---

## 🚀 Quick Start

1.  **Activate your virtual environment** (if you created one).
    ```bash
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

2.  **Launch the application.**
    ```bash
    python main_gui.py
    ```

3.  **Follow the workflow:**
    - **📂 Data Tab**: Click `Load CSV` to import your MZmine data. Use `Configure Groups` to assign samples to experimental groups (e.g., 'Control', 'Treatment_A').
    - **🔧 Preprocessing Tab**: Apply filters (like QC-RSD) and normalization (like TIC or Log). Click `Run Preprocessing`.
    - **📊 PCA Tab**: Run PCA to get an initial overview of your data and check for outliers.
    - **📊 Univariate Screening Tab**: Create a Volcano Plot to find significantly different features between two groups.
    - **📈 PLS-DA & ✅ PLS Validation Tabs**: Run PLS-DA to find features that best separate groups and validate the model with permutation testing.
    - **🌲 Random Forest Tab**: Use another method to rank important features.
    - **🔥 Heatmap Tab**: Visualize the top features from your models.
    - **File Menu**: Use the `Export` menu to save your results or `Save Session` to save your progress.

---

## ⚙️ Dependencies

This tool is built on the following open-source libraries:
- **GUI**: `tkinter` (Python Standard Library)
- **Data Analysis**: `pandas`, `numpy`, `scikit-learn`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `matplotlib-venn`
- **Utilities**: `openpyxl` (for Excel export), `scikit-bio` (for PERMANOVA)

---

## 🐛 Troubleshooting

- **GUI fails to start with a "ModuleNotFoundError"**:
  - Ensure you have activated your virtual environment.
  - Run `pip install -r requirements.txt` again to make sure all packages are installed.

- **Error loading CSV file**:
  - Make sure the file is a valid CSV generated by MZmine.
  - Check that the file is not corrupted and has standard UTF-8 encoding.

- **PERMANOVA/PERMDISP fails**:
  - This feature requires the `scikit-bio` package. Install it with `pip install scikit-bio`.

- **Excel export fails**:
  - This feature requires the `openpyxl` package. Install it with `pip install openpyxl`.

---

## 📞 Contact & Support

For questions, bug reports, or feature requests, please contact the author or open an issue on the project repository.

---

## 📝 License
This software is intended for educational and research use.
