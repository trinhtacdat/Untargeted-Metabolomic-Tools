# export_utils.py
"""
Unified export functions for all analysis results
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


class ExportManager:
    """Centralized export functionality for all analyses"""
    
    @staticmethod
    def process_feature_ids(df, id_column='FeatureID'):
        """
        Splits composite FeatureID (ID_mz_RT) into separate ID, m/z, RT columns.
        Inserts them immediately after the FeatureID column.
        """
        # Find the column (case-insensitive, ignoring underscore)
        target_col = None
        for col in df.columns:
            if col.lower().replace('_', '') == id_column.lower().replace('_', ''):
                target_col = col
                break
        
        if target_col is None:
            return df
            
        try:
            # Regex to split ID_mz_RT (capturing groups: ID, mz, rt)
            # Matches last two parts as numbers, everything else before is ID
            pattern = r'^(.*)_([\d\.]*)_([\d\.]*)$'
            
            if df[target_col].astype(str).str.match(pattern).any():
                extracted = df[target_col].astype(str).str.extract(pattern)
                extracted.columns = ['ID', 'm/z', 'RT']
                
                # Insert columns after the target column
                loc = df.columns.get_loc(target_col) + 1
                
                df.insert(loc, 'RT', pd.to_numeric(extracted['RT'], errors='ignore'))
                df.insert(loc, 'm/z', pd.to_numeric(extracted['m/z'], errors='ignore'))
                df.insert(loc, 'ID', extracted['ID'])
        except Exception:
            pass # Return original DF if splitting fails
            
        return df

    @staticmethod
    def export_preprocessed_data(preprocessed_data, default_filename="preprocessed_data.csv"):
        """Export preprocessed feature matrix"""
        if preprocessed_data is None:
            messagebox.showwarning("Warning", "No preprocessed data to export!")
            return None
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                df_export = ExportManager.process_feature_ids(preprocessed_data.copy(), 'Feature_ID')
                df_export.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Data exported to:\n{filepath}")
                return filepath
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        return None

    @staticmethod
    def save_plot_high_res(fig, default_filename="plot.png", parent=None):
        """Export plot with custom resolution and size"""
        if fig is None:
            messagebox.showwarning("Warning", "No plot to export!", parent=parent)
            return

        # Create settings dialog
        dialog = tk.Toplevel(parent)
        dialog.title("Export High-Resolution Plot")
        dialog.geometry("350x250")
        dialog.transient(parent)
        dialog.grab_set()
        
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill='both', expand=True)
        
        # Get current size
        curr_w, curr_h = fig.get_size_inches()
        
        # Width
        ttk.Label(frame, text="Width (inches):").grid(row=0, column=0, sticky='w', pady=5)
        w_var = tk.StringVar(value=f"{curr_w:.1f}")
        ttk.Entry(frame, textvariable=w_var, width=10).grid(row=0, column=1, pady=5)
        
        # Height
        ttk.Label(frame, text="Height (inches):").grid(row=1, column=0, sticky='w', pady=5)
        h_var = tk.StringVar(value=f"{curr_h:.1f}")
        ttk.Entry(frame, textvariable=h_var, width=10).grid(row=1, column=1, pady=5)
        
        # DPI
        ttk.Label(frame, text="DPI (resolution):").grid(row=2, column=0, sticky='w', pady=5)
        dpi_var = tk.StringVar(value="300")
        ttk.Entry(frame, textvariable=dpi_var, width=10).grid(row=2, column=1, pady=5)
        
        def do_save():
            try:
                w = float(w_var.get())
                h = float(h_var.get())
                dpi = int(dpi_var.get())
                
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    initialfile=default_filename,
                    filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("PDF Document", "*.pdf"), ("SVG Image", "*.svg"), ("All Files", "*.*")],
                    parent=dialog
                )
                
                if filepath:
                    old_size = fig.get_size_inches()
<<<<<<< HEAD
                    autoscale_text = getattr(fig, '_export_autoscale_text', [])
                    original_font_sizes = [text.get_fontsize() for text in autoscale_text]
                    old_area = max(float(old_size[0] * old_size[1]), 0.01)
                    size_scale = np.sqrt((w * h) / old_area)
                    fig.set_size_inches(w, h) # Apply temp size
                    for text, font_size in zip(autoscale_text, original_font_sizes):
                        text.set_fontsize(max(1.0, font_size * size_scale))
                    try:
                        try:
                            fig.tight_layout()
                        except Exception:
                            pass
=======
                    fig.set_size_inches(w, h) # Apply temp size
                    try:
>>>>>>> 66f77d8c4e0a0004279436d58573ca587e587373
                        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
                        messagebox.showinfo("Success", f"Plot saved to:\n{filepath}", parent=dialog)
                        dialog.destroy()
                    finally:
<<<<<<< HEAD
                        for text, font_size in zip(autoscale_text, original_font_sizes):
                            text.set_fontsize(font_size)
                        fig.set_size_inches(old_size) # Restore size
                        try:
                            fig.tight_layout()
                        except Exception:
                            pass
=======
                        fig.set_size_inches(old_size) # Restore size
>>>>>>> 66f77d8c4e0a0004279436d58573ca587e587373
            except ValueError:
                messagebox.showerror("Input Error", "Width, Height and DPI must be numeric.", parent=dialog)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        ttk.Button(btn_frame, text="Save", command=do_save).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)
    
    @staticmethod
    def export_pca_results(pca_result, default_filename="pca_results.csv"):
        """Export PCA scores with sample info"""
        if pca_result is None:
            messagebox.showwarning("Warning", "No PCA results to export!")
            return None
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Create DataFrame with scores
                df = pd.DataFrame(
                    pca_result['scores'],
                    columns=[f"PC{i+1}" for i in range(pca_result['scores'].shape[1])]
                )
                
                # Add sample names and groups
                df.insert(0, 'Sample', pca_result['sample_names'])
                if 'groups' in pca_result:
                    df.insert(1, 'Group', pca_result['groups'])
                
                # Add variance explained as footer
                variance_info = pd.DataFrame({
                    'Sample': ['Variance Explained (%)'],
                    **{f"PC{i+1}": [f"{v*100:.2f}"] 
                       for i, v in enumerate(pca_result.get('variance_explained', []))}
                })
                
                df = pd.concat([df, variance_info], ignore_index=True)
                df.to_csv(filepath, index=False)
                
                messagebox.showinfo("Success", f"PCA results exported to:\n{filepath}")
                return filepath
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        return None
    
    @staticmethod
    def export_volcano_results(volcano_result, screened_data, 
                               pvalue_threshold=0.05, fc_threshold=2.0,
                               default_filename="volcano_significant_features.csv"):
        """Export significant features from volcano plot"""
        if volcano_result is None:
            messagebox.showwarning("Warning", "No volcano plot results to export!")
            return None
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Use feature IDs from result if available to ensure alignment
                if 'feature_ids' in volcano_result:
                    feature_ids = volcano_result['feature_ids']
                else:
                    feature_ids = screened_data.iloc[:, 0].values
                
                # Handle dictionary key mismatch (main_gui uses 'pvalues' and 'fold_changes')
                pvalues = volcano_result.get('pvalues', volcano_result.get('pvalues_adj'))
                log2fc = volcano_result.get('fold_changes', volcano_result.get('log2_fc'))

                if pvalues is None or log2fc is None:
                    raise KeyError("Missing 'pvalues' or 'fold_changes' in results")
                
                # Identify significant features
                sig_mask = (pvalues < pvalue_threshold) & (np.abs(log2fc) > np.log2(fc_threshold))
                
                df = pd.DataFrame({
                    'FeatureID': feature_ids[sig_mask],
                    'Log2FoldChange': log2fc[sig_mask],
                    'FoldChange': 2 ** np.abs(log2fc[sig_mask]),
                    'AdjustedPValue': pvalues[sig_mask],
                    'MinusLog10P': -np.log10(pvalues[sig_mask]),
                    'Abundance': ['Higher' if fc > 0 else 'Lower' for fc in log2fc[sig_mask]]
                })
                
                # Sort by adjusted p-value
                df = df.sort_values('AdjustedPValue')
                
                df = ExportManager.process_feature_ids(df, 'FeatureID')
                df.to_csv(filepath, index=False)
                
                messagebox.showinfo("Success", 
                    f"Exported {len(df)} significant features to:\n{filepath}")
                return filepath
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        return None
    
    @staticmethod
    def export_volcano_subset(volcano_result, pvalue_threshold=0.05, fc_threshold=2.0, 
                              subset_type="BOTH", default_filename="volcano_subset.csv"):
        """Export subset of significant features (UP, DOWN, or BOTH)"""
        if volcano_result is None:
            messagebox.showwarning("Warning", "No volcano plot results to export!")
            return None
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"volcano_{subset_type.lower()}_features.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                feature_ids = volcano_result['feature_ids']
                pvalues = volcano_result['pvalues'] # corrected_pvalues if available
                log2fc = volcano_result['log2_fc'] if 'log2_fc' in volcano_result else volcano_result['fold_changes']
                
                # Identify features based on subset_type
                if subset_type == "UP":
                    mask = (pvalues < pvalue_threshold) & (log2fc > np.log2(fc_threshold))
                elif subset_type == "DOWN":
                    mask = (pvalues < pvalue_threshold) & (log2fc < -np.log2(fc_threshold))
                else: # BOTH
                    mask = (pvalues < pvalue_threshold) & (np.abs(log2fc) > np.log2(fc_threshold))
                
                df = pd.DataFrame({
                    'FeatureID': feature_ids[mask],
                    'Log2FoldChange': log2fc[mask],
                    'FoldChange': 2 ** np.abs(log2fc[mask]),
                    'PValue': pvalues[mask],
                    'Abundance': ['Higher' if fc > 0 else 'Lower' for fc in log2fc[mask]]
                })
                
                # Sort by p-value
                df = df.sort_values('PValue')
                
                df = ExportManager.process_feature_ids(df, 'FeatureID')
                df.to_csv(filepath, index=False)
                
                messagebox.showinfo("Success", 
                    f"Exported {len(df)} {subset_type} features to:\n{filepath}")
                return filepath
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        return None

    @staticmethod
    def export_plsda_results(plsda_result, screened_data, 
                            default_filename="plsda_results.csv"):
        """Export PLS-DA scores, VIP, and metrics"""
        if plsda_result is None:
            messagebox.showwarning("Warning", "No PLS-DA results to export!")
            return None
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Part 1: Scores
                scores_df = pd.DataFrame(
                    plsda_result['scores'],
                    columns=[f"LV{i+1}" for i in range(plsda_result['scores'].shape[1])]
                )
                scores_df.insert(0, 'Sample', plsda_result['sample_names'])
                scores_df.insert(1, 'Group', plsda_result['groups'])
                
                # Part 2: VIP scores
                feature_ids = screened_data.iloc[:, 0].values
                vip_df = pd.DataFrame({
                    'FeatureID': feature_ids,
                    'VIP': plsda_result['vip_scores']
                })
                vip_df = vip_df.sort_values('VIP', ascending=False)
                
                vip_df = ExportManager.process_feature_ids(vip_df, 'FeatureID')

                # Part 3: Model metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['R²X', 'R²Y', 'Q²', 'CV_Accuracy'],
                    'Value': [
                        f"{plsda_result.get('r2_x', 0):.4f}",
                        f"{plsda_result.get('r2_y', 0):.4f}",
                        f"{plsda_result.get('q2_y', 0):.4f}",
                        f"{plsda_result.get('cv_accuracy', 0):.4f}"
                    ]
                })
                
                # Write to Excel with multiple sheets
                if filepath.endswith('.csv'):
                    # If CSV, create separate files
                    base = filepath.replace('.csv', '')
                    scores_df.to_csv(f"{base}_scores.csv", index=False)
                    vip_df.to_csv(f"{base}_vip.csv", index=False)
                    metrics_df.to_csv(f"{base}_metrics.csv", index=False)
                    msg = f"Exported 3 files:\n{base}_scores.csv\n{base}_vip.csv\n{base}_metrics.csv"
                else:
                    # Excel format
                    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                        scores_df.to_excel(writer, sheet_name='Scores', index=False)
                        vip_df.to_excel(writer, sheet_name='VIP', index=False)
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                    msg = f"Exported to:\n{filepath}"
                
                messagebox.showinfo("Success", msg)
                return filepath
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        return None
    
    @staticmethod
    def export_rf_results(rf_result, default_filename="rf_feature_importance.csv"):
        """Export Random Forest feature importance"""
        if rf_result is None:
            messagebox.showwarning("Warning", "No Random Forest results to export!")
            return None
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Check if full importance data is available to export all features
                if 'importances' in rf_result and 'feature_ids' in rf_result:
                    df = pd.DataFrame({
                        'FeatureID': rf_result['feature_ids'],
                        'Importance': rf_result['importances']
                    })
                    df = df.sort_values('Importance', ascending=False)
                else:
                    df = pd.DataFrame(
                        rf_result['top_features'],
                        columns=['FeatureID', 'Importance']
                    )
                
                df = ExportManager.process_feature_ids(df, 'FeatureID')

                # Add model metrics as header comments
                with open(filepath, 'w', newline='') as f:
                    f.write(f"# Random Forest Classification Results\n")
                    f.write(f"# CV Accuracy: {rf_result['cv_accuracy']:.4f} ± {rf_result['cv_std']:.4f}\n")
                    if not np.isnan(rf_result.get('oob_error', float('nan'))):
                        f.write(f"# OOB Error: {rf_result['oob_error']:.4f}\n")
                    f.write("#\n")
                    
                    # Write DataFrame
                    df.to_csv(f, index=False)
                
                messagebox.showinfo("Success", f"RF results exported to:\n{filepath}")
                return filepath
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        return None
    
    @staticmethod
    def export_heatmap_data(heatmap_data, default_filename="heatmap_data.csv"):
        """Export clustered heatmap data matrix"""
        if heatmap_data is None:
            messagebox.showwarning("Warning", "No heatmap data to export!")
            return None
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # heatmap_data is expected to be a DataFrame
                if isinstance(heatmap_data, pd.DataFrame):
                    df_export = heatmap_data.copy()
                else:
                    df_export = pd.DataFrame(heatmap_data)
                
                df_export = ExportManager.process_feature_ids(df_export, 'Feature_ID')
                df_export.to_csv(filepath, index=False)
                
                messagebox.showinfo("Success", f"Heatmap data exported to:\n{filepath}")
                return filepath
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        return None

    @staticmethod
    def export_pdf_report(generated_plots, plsda_result, rf_result, heatmap_data, screened_data, source_file_info=""):
        """Export all generated plots to a single PDF file."""
        if not generated_plots:
            messagebox.showwarning("Warning", "No plots generated yet to export!")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=f"Metabolomics_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            filetypes=[("PDF files", "*.pdf")]
        )

        if filepath:
            try:
                with PdfPages(filepath) as pdf:
                    # Title Page
                    fig = plt.figure(figsize=(8.5, 11))
                    txt = (f"Herbal Metabolomics Analysis Report\n"
                           f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                           f"Source: {source_file_info}\n")
                    
                    if screened_data is not None:
                        txt += f"\nFeatures (Screened): {len(screened_data)}"
                    
                    if plsda_result:
                        txt += f"\n\nPLS-DA Results:\n"
                        txt += f"R2Y: {plsda_result.get('r2_y', 0):.3f}, Q2: {plsda_result.get('q2_y', 0):.3f}"
                        if 'cv_accuracy' in plsda_result:
                            txt += f"\nCV Accuracy: {plsda_result['cv_accuracy']:.3f}"
                            
                    if rf_result:
                        txt += f"\n\nRandom Forest Results:\n"
                        txt += f"CV Accuracy: {rf_result.get('cv_accuracy', 0):.3f}"
                        if 'oob_error' in rf_result and not np.isnan(rf_result['oob_error']):
                            txt += f"\nOOB Error: {rf_result['oob_error']:.3f}"

                    fig.text(0.1, 0.7, txt, fontsize=12, family='monospace')
                    pdf.savefig(fig)
                    plt.close(fig)

                    # Export Plots in specific order
<<<<<<< HEAD
                    plot_order = ['Preprocessing', 'Global Heatmap', 'PCA', 'Univariate Screening', 'PLS-DA', 'PLS-DA Validation', 'Random Forest', 'Heatmap', 'Venn Diagram']
=======
                    plot_order = ['Preprocessing', 'PCA', 'Univariate Screening', 'PLS-DA', 'PLS-DA Validation', 'Random Forest', 'Heatmap', 'Venn Diagram']
>>>>>>> 66f77d8c4e0a0004279436d58573ca587e587373
                    for name in plot_order:
                        if name in generated_plots:
                            try:
                                pdf.savefig(generated_plots[name])
                            except Exception:
                                pass
                    
                    # Export any remaining plots
                    for name, fig in generated_plots.items():
                        if name not in plot_order:
                            try:
                                pdf.savefig(fig)
                            except Exception:
                                pass

                messagebox.showinfo("Success", f"Report exported to:\n{filepath}")
                return filepath
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        return None
