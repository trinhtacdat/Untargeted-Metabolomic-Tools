
# Create comprehensive modular structure with MGF integration and export functions

structure = """
COMPREHENSIVE METABOLOMICS GUI - MODULAR STRUCTURE
================================================================

FILE STRUCTURE:
├── main_gui.py (main window, tabs, orchestration)
├── data_handler.py (CSV loading, sample management)
├── preprocessing.py (normalization, filtering, QC)
├── univariate_stats.py (volcano plot, t-tests)
├── multivariate_analysis.py (PCA, PLS-DA with R²/Q² validation)
├── random_forest.py (RF classification, feature importance)
├── mgf_handler.py (NEW - MGF parsing, spectrum search)
├── export_utils.py (NEW - unified export functions)
└── visualization.py (plotting utilities)

KEY ADDITIONS REQUESTED:
========================

1. FIX PLS-DA R² and Q² CALCULATION
   - Current: uses cross_val_score (incorrect for Q²)
   - Fixed: proper LOOCV with Q² = 1 - PRESS/TSS
   - Add R²X, R²Y, Q² metrics to validation

2. ADD EXPORT FUNCTIONS FOR EACH ANALYSIS
   ✓ Export Preprocessed Data (CSV)
   ✓ Export PCA Results (CSV)
   ✓ Export Volcano Plot Results (CSV with significant features)
   → NEW: Export PLS-DA Results (scores, VIP, metrics)
   → NEW: Export RF Results (feature importance, predictions)
   → NEW: Export Heatmap Data (clustered feature matrix)

3. ADD MGF SPECTRUM SEARCH
   - Import parse_mgf() from MGF_Viewer_GUI.py
   - New tab: "Spectrum Search"
   - Input: Load MGF file(s)
   - Search by: Feature ID, m/z, RT
   - Display matched spectra for important features
   - Export: Annotated feature list with spectrum info

"""

print(structure)
print("\n" + "="*60)
print("GENERATING MODULAR CODE FILES...")
print("="*60)
