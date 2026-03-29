# 🚀 Quick Start Guide
## Herbal Metabolomics Analyzer

---

## 📥 Installation

### 1. Install Python Requirements
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python demo_workflow.py
```

This should run the analysis demo on your JW-119-Herba_quant.csv file.

---

## 🖥️ Launch GUI Application

```bash
python main_gui.py
```

---

## 📊 Analysis Workflow

### Step 1: Load Data
1. Click **"📂 Load CSV File"** button
2. Select your MZmine quantification CSV file
3. Review data summary and preview

### Step 2: Preprocessing
1. Go to **"🔧 Preprocessing"** tab
2. Check **"Average Technical Replicates"** (default: ON)
   - Automatically averages 2 injections → 1 value
   - QC samples (>2 injections) are preserved
3. Select **Normalization Method**:
   - **TIC** (recommended for herbal samples)
   - Median
   - Log
   - Pareto
   - Auto
4. Click **"▶️ Run Preprocessing"**

### Step 3: Exploratory Analysis (PCA)
1. Go to **"📊 PCA"** tab
2. Set number of components (default: 2)
3. Click **"▶️ Run PCA"**
4. Review score plot and variance explained

### Step 4: PLS-DA Analysis
1. Go to **"📈 PLS-DA"** tab
2. Run analysis with cross-validation
3. Review VIP scores (features with VIP > 1.0 are important)

### Step 5: Volcano Plot (Statistical Significance)
1. Go to **"🌋 Volcano Plot"** tab
2. Set p-value threshold (default: 0.05)
3. Set fold-change threshold (default: 2.0)
4. Identify significant features

### Step 6: Random Forest Feature Selection
1. Go to **"🌲 Random Forest"** tab
2. Set number of trees (default: 500)
3. Run analysis
4. Review top N important features
5. Export feature list

---

## 💡 Key Concepts

### Technical Replicate Averaging
- **What**: Each sample has 2 injections
- **Why**: Reduce technical variation
- **How**: Automatically calculates mean of 2 injections
- **Exception**: QC samples (>2 injections) kept separate

### Sample vs QC Detection
- **Regular samples**: Exactly 2 injections (e.g., JW1-119-1_1, JW1-119-1_2)
- **QC samples**: More than 2 injections (e.g., JW1-119-46_1, _2, _3, _4)
- **Auto-detected**: No manual specification needed!

### Group Assignment
- Assign samples to groups (e.g., "Herbal_A", "Herbal_B", "Control")
- Required for:
  - PLS-DA
  - Volcano plot (two-group comparison)
  - Random Forest classification

### Feature Selection Strategy
To find **key features distinguishing two herbal samples**, use **consensus approach**:

1. **VIP scores > 1.0** (from PLS-DA) → Features driving separation
2. **Volcano plot significant** (p < 0.05, FC > 2) → Statistically different
3. **RF top-ranked** → Important for classification

**Features appearing in ALL THREE = Robust biomarkers!**

---

## 📤 Export Results

### Available Exports:
- Feature lists (Excel/CSV)
- Statistical results (Excel/CSV)
- Plots (PNG/SVG/PDF)
- Full analysis report (PDF)

---

## 🐛 Troubleshooting

### "Error loading data"
- Ensure file is MZmine CSV format
- Check file is not corrupted
- Verify column names match expected format

### "Preprocessing failed"
- Check sample naming follows pattern: `JW1-119-{number}_{injection}.mzML Peak area`
- Ensure at least 2 samples per group

### "Not enough samples for analysis"
- PLS-DA requires at least 3 samples per group
- Random Forest requires at least 5 samples per group

---

## 💬 Example: Analyzing Your Data

Your file `JW-119-Herba_quant.csv` contains:
- **19 regular samples** (samples 1-19, each with 2 injections)
- **1 QC sample** (sample 46, with 4 injections)

Suggested grouping (adjust based on your experimental design):
- **Group A**: Samples 1-9 (e.g., one herbal medicine type)
- **Group B**: Samples 10-19 (e.g., different herbal medicine or treatment)
- **QC**: Sample 46 (quality control, excluded from analysis)

Expected results:
- **PCA**: Should show clustering by group
- **PLS-DA**: 70-80% CV accuracy indicates good separation
- **VIP > 1**: 50-100 features expected
- **Volcano**: Significant features depend on biological difference
- **RF**: Top 20-50 features for biomarker discovery

---

## 📚 Further Reading

### Metabolomics Methods:
- **PCA**: Unsupervised, detects outliers and batch effects
- **PLS-DA**: Supervised, maximizes group separation
- **VIP scores**: Variable importance in projection
- **Volcano plot**: Combines statistical significance + effect size
- **Random Forest**: Non-linear classification, robust to noise

### Recommended Workflow:
1. PCA → Check data quality
2. PLS-DA → Assess group separation
3. Volcano → Identify significant features
4. Random Forest → Select robust biomarkers
5. Consensus → Features in all methods = key markers

---

## 📞 Need Help?

For questions or issues:
- Check README.md for detailed documentation
- Review demo_workflow.py for code examples
- Contact: [Your email/institution]

---

**Happy Analyzing! 🌿**
