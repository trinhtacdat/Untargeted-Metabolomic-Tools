# INTEGRATION_GUIDE.md
# Comprehensive Metabolomics GUI - Integration Guide

## Overview
This guide explains how to integrate the improved modules into your existing main_gui-2.py

## Files Created
1. **multivariate_analysis.py** - Fixed PLS-DA with proper R²/Q² calculation
2. **mgf_handler.py** - MGF spectrum parsing and search
3. **export_utils.py** - Unified export functions

## Key Improvements

### 1. Fixed PLS-DA R² and Q² Calculation

**Problem in original code:**
```python
# OLD (INCORRECT)
cvscores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_scaled, y, cv=min(5, len(set(y)))
)
# This uses Random Forest, not PLS-DA for validation!
```

**New (CORRECT):**
```python
# In multivariate_analysis.py
def _calculate_q2(self, X, y):
    """
    Calculate Q² using Leave-One-Out Cross-Validation
    Q² = 1 - PRESS/TSS
    """
    loo = LeaveOneOut()
    y_pred_cv = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pls_cv = PLSRegression(n_components=self.n_components)
        pls_cv.fit(X_train, y_train)
        y_pred_cv[test_idx] = pls_cv.predict(X_test).ravel()
    
    press = np.sum((y - y_pred_cv) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    q2 = 1 - (press / tss)
    
    return q2
```

### 2. Integration Steps

#### Step 1: Add imports to main_gui-2.py

```python
# At the top of main_gui-2.py, add:
from multivariate_analysis import PLSDAAnalyzer, PCAAnalyzer
from mgf_handler import MGFSearchEngine, parse_mgf
from export_utils import ExportManager
```

#### Step 2: Replace run_plsda() method

**Find this method in main_gui-2.py (~line 1500):**

```python
def run_plsda(self):
    """Run PLS-DA analysis with VIP calculation"""
    if getattr(self, 'screened_data', None) is None:
        messagebox.showwarning("Warning", "Please run preprocessing first!")
        return
    
    try:
        self.update_status("Running PLS-DA...")
        n_components = self.plsda_components_var.get()
        exclude_qc = self.plsda_exclude_qc_var.get()
        
        # === REPLACE THIS ENTIRE SECTION ===
        # OLD: plsda_result = self.perform_plsda_local(...)
        
        # NEW: Use PLSDAAnalyzer
        sample_cols = [col for col in self.screened_data.columns if '.mzML' in col]
        
        if exclude_qc:
            filtered_cols = []
            for col in sample_cols:
                basename = re.sub(r'_avg\.mzML.*', '', col)
                groupname = self.group_mapping.get(basename, 'Ungrouped')
                if basename not in self.qc_samples and 'qc' not in groupname.lower():
                    filtered_cols.append(col)
            sample_cols = filtered_cols
        
        if len(sample_cols) < 2:
            raise ValueError("Not enough samples for PLS-DA")
        
        # Prepare data
        X = self.screened_data[sample_cols].T
        X = X.replace(0, np.nan)
        min_val = X.min().min() * 0.2
        if pd.isna(min_val):
            min_val = 0.1
        X = X.fillna(min_val)
        
        # Apply log transform if selected
        if hasattr(self, 'plsda_log_var') and self.plsda_log_var.get():
            X = np.log10(X + 1)
        
        # Get group labels
        y_labels = []
        for col in sample_cols:
            basename = re.sub(r'_avg\.mzML.*', '', col)
            group = self.group_mapping.get(basename, 'Ungrouped')
            y_labels.append(group)
        
        # Initialize analyzer
        analyzer = PLSDAAnalyzer(n_components=n_components)
        
        # Fit and get results
        plsda_result = analyzer.fit_predict(X.values, y_labels)
        
        # Add sample info
        plsda_result['sample_names'] = sample_cols
        plsda_result['groups'] = y_labels
        plsda_result['feature_ids'] = self.screened_data.iloc[:, 0].values
        
        # Store result
        self.plsda_result = plsda_result
        self.plsda_analyzer = analyzer  # Store analyzer for permutation test
        
        # Display results
        self.display_plsda_results(plsda_result)
        self.create_plsda_plot(plsda_result)
        
        self.update_status("PLS-DA complete")
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        messagebox.showerror("Error", f"PLS-DA failed:\n{str(e)}\n{error_detail}")
        self.update_status("Error in PLS-DA")
```

#### Step 3: Update display_plsda_results() method

```python
def display_plsda_results(self, result):
    """Display PLS-DA results with proper metrics"""
    self.plsda_results_text.delete(1.0, tk.END)
    
    text = []
    text.append("=" * 60)
    text.append("PLS-DA RESULTS")
    text.append("=" * 60)
    
    # Display new metrics
    metrics = result['metrics']
    text.append(f"R²X (variance in X): {metrics['r2x']:.4f}")
    text.append(f"R²Y (variance in Y): {metrics['r2y']:.4f}")
    text.append(f"Q² (predictive ability): {metrics['q2']:.4f}")
    text.append(f"CV Accuracy: {metrics['cv_accuracy']:.4f}")
    text.append("")
    text.append("Interpretation:")
    text.append(f"  - R²X > 0.5: {'GOOD' if metrics['r2x'] > 0.5 else 'POOR'} model fit")
    text.append(f"  - Q² > 0.5: {'GOOD' if metrics['q2'] > 0.5 else 'POOR'} predictive power")
    text.append(f"  - Q² > 0: {'Valid' if metrics['q2'] > 0 else 'INVALID'} model")
    text.append("")
    text.append(f"Top 10 VIP Scores (VIP > 1.0 indicates important features)")
    text.append("-" * 60)
    
    # Display top VIP features
    feature_ids = result.get('feature_ids', self.screened_data.iloc[:, 0].values)
    vip_with_ids = list(zip(feature_ids, result['vip_scores']))
    vip_sorted = sorted(vip_with_ids, key=lambda x: x[1], reverse=True)
    
    for i, (feat_id, vip) in enumerate(vip_sorted[:10], 1):
        text.append(f"{i:2d}. {feat_id}: {vip:.3f}")
    
    self.plsda_results_text.insert(1.0, '\n'.join(text))
```

#### Step 4: Update run_plsval() for permutation test

```python
def run_plsval(self):
    """Run Permutation Testing for PLS-DA Validation"""
    if not hasattr(self, 'plsda_analyzer') or self.plsda_analyzer is None:
        messagebox.showwarning("Warning", "Please run PLS-DA first!")
        return
    
    try:
        self.update_status("Running PLS Validation Permutation Test...")
        n_perms = self.n_perms_var.get()
        
        # Get X and y from previous PLS-DA
        sample_cols = self.plsda_result['sample_names']
        X = self.screened_data[sample_cols].T
        X = X.replace(0, np.nan).fillna(X.min().min() * 0.2)
        
        if hasattr(self, 'plsda_log_var') and self.plsda_log_var.get():
            X = np.log10(X + 1)
        
        y_labels = self.plsda_result['groups']
        
        # Run permutation test
        perm_result = self.plsda_analyzer.permutation_test(
            X.values, y_labels, n_permutations=n_perms
        )
        
        # Plot results
        self.plot_permutation_test(perm_result)
        
        self.update_status("PLS Validation complete")
        
        # Show summary
        messagebox.showinfo(
            "Success",
            f"Validation complete!\n\n"
            f"Actual Q²: {perm_result['actual_q2']:.4f}\n"
            f"Q² intercept: {perm_result['q2_intercept']:.4f}\n"
            f"P-value: {perm_result['p_value']:.4f}\n\n"
            f"{'Valid model' if perm_result['p_value'] < 0.05 else 'Model may be overfitted'}"
        )
        
    except Exception as e:
        import traceback
        messagebox.showerror("Error", f"Validation failed:\n{str(e)}\n{traceback.format_exc()}")
```

#### Step 5: Add MGF Spectrum Search Tab

**Add this method after create_heatmap_tab():**

```python
def create_spectrum_tab(self):
    """Tab 9: Spectrum Search"""
    tab = ttk.Frame(self.notebook)
    self.notebook.add(tab, text="Spectrum Search")
    
    # Control frame
    control_frame = ttk.LabelFrame(tab, text="MGF File Management", padding=10)
    control_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Button(
        control_frame,
        text="Load MGF File(s)",
        command=self.load_mgf_files
    ).pack(side=tk.LEFT, padx=5)
    
    self.mgf_status_label = ttk.Label(control_frame, text="No MGF files loaded")
    self.mgf_status_label.pack(side=tk.LEFT, padx=10)
    
    # Search frame
    search_frame = ttk.LabelFrame(tab, text="Search Spectra", padding=10)
    search_frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(search_frame, text="Search by:").grid(row=0, column=0, sticky='w', padx=5)
    self.spectrum_search_type_var = tk.StringVar(value="Feature ID")
    search_type_combo = ttk.Combobox(
        search_frame,
        textvariable=self.spectrum_search_type_var,
        values=["Feature ID", "m/z", "RT range"],
        state="readonly",
        width=15
    )
    search_type_combo.grid(row=0, column=1, sticky='w', padx=5)
    
    ttk.Label(search_frame, text="Value:").grid(row=1, column=0, sticky='w', padx=5)
    self.spectrum_search_value_var = tk.StringVar()
    ttk.Entry(search_frame, textvariable=self.spectrum_search_value_var, width=30).grid(
        row=1, column=1, sticky='ew', padx=5
    )
    
    ttk.Label(search_frame, text="Tolerance:").grid(row=2, column=0, sticky='w', padx=5)
    self.spectrum_tolerance_var = tk.StringVar(value="0.01")
    ttk.Entry(search_frame, textvariable=self.spectrum_tolerance_var, width=10).grid(
        row=2, column=1, sticky='w', padx=5
    )
    
    ttk.Button(search_frame, text="Search", command=self.search_spectra).grid(
        row=3, column=0, columnspan=2, pady=10
    )
    
    ttk.Button(search_frame, text="Annotate Important Features", 
              command=self.annotate_features_with_spectra).grid(
        row=4, column=0, columnspan=2, pady=5
    )
    
    # Results frame
    results_frame = ttk.LabelFrame(tab, text="Search Results", padding=10)
    results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    self.spectrum_results_text = scrolledtext.ScrolledText(
        results_frame, height=20, font=("Courier", 10)
    )
    self.spectrum_results_text.pack(fill=tk.BOTH, expand=True)
    
    # Initialize MGF search engine
    self.mgf_engine = MGFSearchEngine()
```

**Add these methods for MGF functionality:**

```python
def load_mgf_files(self):
    """Load MGF file(s) for spectrum search"""
    filepaths = filedialog.askopenfilenames(
        title="Select MGF file(s)",
        filetypes=[("MGF files", "*.mgf"), ("All files", "*.*")]
    )
    
    if not filepaths:
        return
    
    try:
        num_spectra = self.mgf_engine.load_mgf_files(list(filepaths))
        self.mgf_status_label.config(
            text=f"Loaded {num_spectra} spectra from {len(filepaths)} file(s)"
        )
        messagebox.showinfo("Success", f"Loaded {num_spectra} spectra")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load MGF files:\n{str(e)}")

def search_spectra(self):
    """Search loaded spectra"""
    if not self.mgf_engine.spectra:
        messagebox.showwarning("Warning", "Please load MGF files first!")
        return
    
    search_type = self.spectrum_search_type_var.get()
    search_value = self.spectrum_search_value_var.get().strip()
    
    if not search_value:
        messagebox.showwarning("Warning", "Please enter a search value!")
        return
    
    try:
        self.spectrum_results_text.delete(1.0, tk.END)
        
        if search_type == "Feature ID":
            # Search by one or more Feature IDs (space-separated)
            feature_ids = search_value.split()
            matches = self.mgf_engine.search_by_feature_id(feature_ids)
            
        elif search_type == "m/z":
            target_mz = float(search_value)
            tolerance = float(self.spectrum_tolerance_var.get())
            matches = self.mgf_engine.search_by_mz(target_mz, tolerance)
            
        elif search_type == "RT range":
            # Expected format: "5-10" (minutes)
            rt_min, rt_max = map(float, search_value.split('-'))
            matches = self.mgf_engine.search_by_rt(rt_min, rt_max)
        
        # Display results
        result_text = []
        result_text.append(f"Found {len(matches)} matching spectra:\n")
        result_text.append("=" * 80 + "\n")
        
        for i, spec in enumerate(matches, 1):
            result_text.append(f"\n--- Spectrum {i} ---")
            result_text.append(f"Feature ID: {spec.get('FEATUREID', spec.get('SCANS', 'N/A'))}")
            result_text.append(f"Precursor m/z: {spec.get('PEPMASS', 'N/A')}")
            result_text.append(f"RT (sec): {spec.get('RTINSECONDS', 'N/A')}")
            result_text.append(f"Charge: {spec.get('CHARGE', 'N/A')}")
            result_text.append(f"Num peaks: {len(spec.get('mzs', []))}")
            result_text.append(f"Source file: {spec.get('SOURCEFILE', 'N/A')}")
            
            if 'matched_mzs' in spec:
                result_text.append(f"Matched m/z values: {', '.join(f'{mz:.4f}' for mz in spec['matched_mzs'][:5])}")
            
            result_text.append("-" * 80)
        
        self.spectrum_results_text.insert(1.0, '\n'.join(result_text))
        
    except Exception as e:
        messagebox.showerror("Error", f"Search failed:\n{str(e)}")

def annotate_features_with_spectra(self):
    """Annotate important features (PLS-DA VIP + RF) with spectrum information"""
    if not self.mgf_engine.spectra:
        messagebox.showwarning("Warning", "Please load MGF files first!")
        return
    
    if self.plsda_result is None and self.rf_result is None:
        messagebox.showwarning("Warning", "Please run PLS-DA or Random Forest first!")
        return
    
    try:
        # Get important features
        important_features = set()
        
        if self.plsda_result:
            vips = self.plsda_result['vip_scores']
            feature_ids = self.plsda_result.get('feature_ids', self.screened_data.iloc[:, 0].values)
            top_vip_idx = np.argsort(vips)[-20:]
            important_features.update(feature_ids[top_vip_idx])
        
        if self.rf_result:
            top_rf_features = [f[0] for f in self.rf_result['top_features'][:20]]
            important_features.update(top_rf_features)
        
        # Create DataFrame
        feature_df = pd.DataFrame({'FeatureID': list(important_features)})
        
        # Annotate with spectra
        annotated_df = self.mgf_engine.annotate_features_with_spectra(feature_df)
        
        # Export
        filepath = ExportManager.export_preprocessed_data(
            annotated_df,
            default_filename="important_features_with_spectra.csv"
        )
        
        if filepath:
            # Show summary
            num_with_spectra = annotated_df['HasSpectrum'].sum()
            messagebox.showinfo(
                "Success",
                f"Annotated {len(annotated_df)} important features\n"
                f"{num_with_spectra} have matching spectra ({num_with_spectra/len(annotated_df)*100:.1f}%)"
            )
    
    except Exception as e:
        messagebox.showerror("Error", f"Annotation failed:\n{str(e)}")
```

#### Step 6: Add Export Buttons

**Update the menu bar to add export options:**

```python
def create_menu_bar(self):
    menubar = tk.Menu(self.root)
    self.root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Load CSV Data", command=self.load_csv)
    file_menu.add_separator()
    
    # Export submenu
    export_menu = tk.Menu(file_menu, tearoff=0)
    file_menu.add_cascade(label="Export", menu=export_menu)
    
    export_menu.add_command(
        label="Export Preprocessed Data",
        command=lambda: ExportManager.export_preprocessed_data(self.preprocessed_data)
    )
    export_menu.add_command(
        label="Export PCA Results",
        command=lambda: ExportManager.export_pca_results(self.pca_result)
    )
    export_menu.add_command(
        label="Export Volcano Results",
        command=lambda: ExportManager.export_volcano_results(
            self.volcano_result, self.screened_data,
            self.pvalue_var.get(), self.fc_var.get()
        )
    )
    export_menu.add_command(
        label="Export PLS-DA Results",
        command=lambda: ExportManager.export_plsda_results(
            self.plsda_result, self.screened_data
        )
    )
    export_menu.add_command(
        label="Export RF Results",
        command=lambda: ExportManager.export_rf_results(self.rf_result)
    )
    
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=self.root.quit)
    
    # ... rest of menu bar
```

### 3. Testing Checklist

After integration, test:

- [ ] PLS-DA shows R²X, R²Y, Q² metrics
- [ ] Q² is calculated using proper LOOCV
- [ ] Permutation test shows p-value and intercepts
- [ ] All export functions work (PCA, Volcano, PLS-DA, RF)
- [ ] MGF files load successfully
- [ ] Feature ID search finds correct spectra
- [ ] Annotation exports important features with spectrum info

### 4. Expected Output Format

**PLS-DA Results Display:**
```
============================================================
PLS-DA RESULTS
============================================================
R²X (variance in X): 0.7234
R²Y (variance in Y): 0.8912
Q² (predictive ability): 0.6543
CV Accuracy: 0.9474

Interpretation:
  - R²X > 0.5: GOOD model fit
  - Q² > 0.5: GOOD predictive power
  - Q² > 0: Valid model

Top 10 VIP Scores (VIP > 1.0 indicates important features)
------------------------------------------------------------
 1. 123_456.78_12.34: 2.134
 2. 124_457.22_13.45: 1.987
 ...
```

**Permutation Test Output:**
- Scatter plot: Correlation vs R²/Q²
- Red intercept lines showing R² and Q² at zero correlation
- Actual values marked with stars
- P-value in title
- Interpretation: p < 0.05 = valid model

## Summary

This integration provides:
1. ✅ Corrected PLS-DA R²/Q² calculation using proper LOOCV
2. ✅ Comprehensive export functions for all analyses
3. ✅ MGF spectrum search and annotation
4. ✅ Better model validation with permutation testing
5. ✅ Modular, maintainable code structure

The modular design allows you to update individual components without affecting the entire codebase.
