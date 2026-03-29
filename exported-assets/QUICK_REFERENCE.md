# QUICK_REFERENCE.md
# Quick Reference: Key Changes and Usage

## Files You Need

Place these 4 files in the same directory as your main_gui-2.py:

1. **multivariate_analysis.py** - Fixed PLS-DA engine
2. **mgf_handler.py** - MGF spectrum search
3. **export_utils.py** - Export functions
4. **INTEGRATION_GUIDE.md** - Detailed integration instructions

## Critical Fixes

### 1. PLS-DA R²/Q² Now Correct ✅

**Before (WRONG):**
- Used Random Forest for cross-validation
- No proper Q² calculation
- Missing R²X metric

**After (CORRECT):**
- Uses proper LOOCV for Q² = 1 - PRESS/TSS
- Calculates R²X (variance in X)
- Calculates R²Y (variance in Y)
- True predictive ability assessment

### 2. New Metrics Display

```
R²X (variance in X): 0.7234  → How well X is modeled
R²Y (variance in Y): 0.8912  → How well Y is predicted (training)
Q² (predictive ability): 0.6543  → True predictive power (LOOCV)
CV Accuracy: 0.9474  → Classification accuracy

Interpretation guide:
- R²X > 0.5 = GOOD model fit
- Q² > 0.5 = GOOD predictive power
- Q² > 0 = Valid model (Q² < 0 = invalid!)
```

### 3. Export Functions Now Available

**From Menu → File → Export:**
- Export Preprocessed Data (CSV)
- Export PCA Results (CSV with variance explained)
- Export Volcano Results (significant features only)
- Export PLS-DA Results (scores + VIP + metrics as Excel/CSV)
- Export RF Results (feature importance with model metrics)

**From Spectrum Search Tab:**
- Export annotated important features with spectrum info

## New Spectrum Search Tab

### Load MGF Files
1. Click "Load MGF File(s)"
2. Select one or more .mgf files from MZmine
3. Status shows: "Loaded 1234 spectra from 2 file(s)"

### Search Options

**By Feature ID:**
```
Search type: Feature ID
Value: 123_456.78_12.34 124_457.22_13.45
(Space-separated for multiple IDs)
```

**By m/z:**
```
Search type: m/z
Value: 456.7812
Tolerance: 0.01
(Finds all spectra containing peaks at 456.7812 ± 0.01)
```

**By RT Range:**
```
Search type: RT range
Value: 5-10
(RT in minutes)
```

### Annotate Important Features

1. Run PLS-DA and/or Random Forest first
2. Load MGF file(s)
3. Click "Annotate Important Features"
4. Exports CSV with:
   - FeatureID
   - HasSpectrum (True/False)
   - NumPeaks
   - PrecursorMZ
   - RT_minutes

## Validation Best Practices

### Before Publishing Results

1. **Check Q²:**
   - Q² > 0.5 → Excellent predictive model
   - Q² > 0 → Valid model
   - Q² < 0 → Model is overfitted (invalid!)

2. **Run Permutation Test:**
   - Go to "PLS Validation" tab
   - Set n_permutations = 100 (minimum)
   - Check p-value < 0.05 for significance
   - Q² intercept should be < 0.3

3. **Cross-reference Features:**
   - Features with high VIP (>1.5) from PLS-DA
   - Features with high importance from RF
   - Overlap = most robust biomarkers

4. **Verify with Spectra:**
   - Load MGF files
   - Check that important features have actual MS/MS spectra
   - Export annotated list for manual inspection

## Common Issues & Solutions

### Issue: "Q² is negative"
**Solution:** Model is overfitted. Try:
- Reduce number of features (stricter volcano filtering)
- Check for batch effects in PCA
- Ensure groups are truly different

### Issue: "Permutation test p-value > 0.05"
**Solution:** Model may not be valid. Try:
- More samples (n < 20 is risky)
- Check data quality (QC RSD filtering)
- Re-examine group assignments

### Issue: "Can't find spectra for important features"
**Solution:** 
- Check that MGF file matches the quantification CSV
- Feature IDs must match between files
- Use m/z search as fallback

### Issue: "Export PLS-DA creates 3 files instead of Excel"
**Solution:**
- Change file extension to .xlsx in save dialog
- If only .csv is available, 3 separate CSVs will be created:
  - *_scores.csv
  - *_vip.csv
  - *_metrics.csv

## Workflow Recommendation

### For Publication-Ready Analysis:

```
1. Load Data → Configure Groups → QC Selection
2. Preprocessing (Normalize + Filter by QC RSD)
3. PCA (check for outliers/batch effects)
4. Univariate Screening (volcano plot)
   - Set p < 0.05, FC > 2
   - Click "Filter Data for PLS-DA/RF"
5. PLS-DA
   - Check R²X, R²Y, Q²
   - Export results
6. PLS Validation (permutation test)
   - Verify p-value < 0.05
7. Random Forest
   - Cross-validate feature importance
   - Export results
8. Load MGF Files
   - Annotate important features
   - Export for manual curation
9. Heatmap (visual verification)
10. Write manuscript with validated biomarkers!
```

## Code Changes Summary

### In main_gui-2.py, you need to:

1. **Add imports** (top of file):
```python
from multivariate_analysis import PLSDAAnalyzer
from mgf_handler import MGFSearchEngine
from export_utils import ExportManager
```

2. **Replace `run_plsda()` method** (~line 1500)
   - See INTEGRATION_GUIDE.md for full code

3. **Update `display_plsda_results()` method**
   - Display new metrics: R²X, R²Y, Q²

4. **Update `run_plsval()` method**
   - Use `plsda_analyzer.permutation_test()`

5. **Add `create_spectrum_tab()` method**
   - New tab for MGF search
   - See INTEGRATION_GUIDE.md for full code

6. **Update `create_menu_bar()` method**
   - Add Export submenu with all export options

7. **Add 3 new methods:**
   - `load_mgf_files()`
   - `search_spectra()`
   - `annotate_features_with_spectra()`

8. **In `__init__()`, add:**
```python
self.mgf_engine = MGFSearchEngine()
self.plsda_analyzer = None
```

9. **Update tab creation** (~line 2800):
```python
self.create_spectrum_tab()  # Add after heatmap tab
```

## Dependencies

Make sure you have:
```bash
pip install numpy pandas scikit-learn matplotlib scipy openpyxl
```

`openpyxl` is needed for Excel export (.xlsx files).

## Testing Your Integration

### Quick Test Checklist:

1. [ ] Load your CSV data
2. [ ] Run preprocessing
3. [ ] Run PLS-DA → Check metrics display shows R²X, R²Y, Q²
4. [ ] Run PLS Validation → Permutation plot appears with p-value
5. [ ] File → Export → Export PLS-DA Results → Creates Excel/CSV
6. [ ] Go to Spectrum Search tab
7. [ ] Load MGF file → Status updates
8. [ ] Search by Feature ID → Results appear
9. [ ] Run "Annotate Important Features" → CSV exports

If all 9 steps work, integration is successful! ✅

## Support

If you encounter issues:
1. Check Python version (3.8+ required)
2. Verify all dependencies installed
3. Check file paths are correct
4. Review error messages in terminal
5. Consult INTEGRATION_GUIDE.md for detailed instructions

## Version History

- **v2.14.1** → Original code with incorrect Q² calculation
- **v2.15.0** → Fixed PLS-DA, added exports, added MGF search (this update)
