"""
Herbal Metabolomics Analyzer - Visualization Manager
Handles the creation and embedding of all matplotlib plots for the main GUI.
"""

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse, Circle, Patch
import matplotlib.transforms as transforms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import scipy.cluster.hierarchy as sch
import re
from matplotlib.colors import to_rgb
from collections import defaultdict
import itertools
from scipy import stats
import warnings

try:
    from matplotlib_venn import venn2, venn3
    HAS_VENN = True
except ImportError:
    HAS_VENN = False


class VisualizationManager:
    """Manages creation and display of all plots for the MetabolomicsApp."""

    def __init__(self, app):
        """
        Initialize the VisualizationManager.

        Args:
            app: The main MetabolomicsApp instance.
        """
        self.app = app

    def _get_plot_dims(self):
        """Returns the global plot dimensions and DPI."""
        return (
            self.app.plot_width_var.get(),
            self.app.plot_height_var.get(),
            self.app.plot_dpi_var.get(),
        )

    def _get_font_sizes(self):
        """Calculates font sizes based on plot dimensions."""
        w, h, _ = self._get_plot_dims()
        # Base area based on default 8x6 inches
        scale = np.sqrt((w * h) / 48.0)
        scale = max(0.6, min(scale, 2.5))  # Clamp scaling factor

        return {
            'title': int(14 * scale),
            'label': int(12 * scale),
            'tick': int(10 * scale),
            'legend': int(10 * scale),
            'annot': int(8 * scale),
            'tiny': int(7 * scale)
        }

    def _clear_frame(self, frame):
        """Clears all widgets from a given frame."""
        for widget in frame.winfo_children():
            widget.destroy()

    def _embed_plot(self, fig, frame, hover_handler=None):
        """Embeds a matplotlib figure into a tkinter frame."""
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        if hover_handler:
            canvas.mpl_connect("motion_notify_event", hover_handler)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        return canvas

    def create_distribution_plots(self):
        """Show before/after normalization distribution plots."""
        self._clear_frame(self.app.preprocessing_plot_frame)

        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        fig = Figure(figsize=(w * 1.75, h * 1.33), dpi=dpi)
        self.app.generated_plots['Preprocessing'] = fig

        sample_cols = [col for col in self.app.data_before_norm.columns if '.mzML' in col]

        max_samples = 15
        if len(sample_cols) > max_samples:
            indices = np.linspace(0, len(sample_cols) - 1, max_samples, dtype=int)
            sample_cols_subset = [sample_cols[i] for i in indices]
        else:
            sample_cols_subset = sample_cols

        # Subplot 1: Density plot BEFORE normalization
        ax1 = fig.add_subplot(2, 2, 1)
        for col in sample_cols_subset:
            data_col = self.app.data_before_norm[col].replace(0, np.nan).dropna()
            if len(data_col) > 0:
                ax1.hist(data_col, bins=50, alpha=0.3, density=True)

        ax1.set_xlabel('Peak Intensity', fontsize=fonts['label'])
        ax1.set_ylabel('Density', fontsize=fonts['label'])
        ax1.set_title('Before Normalization - Density Plot', fontsize=fonts['title'], fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Check if normalization method produced log/scaled data or kept intensities
        norm_method = self.app.norm_method_var.get()
        needs_log = norm_method in ["TIC", "Median", "None"]
        xlabel_after = 'log10(Peak Intensity + 1)' if needs_log else 'Value'

        # Subplot 2: Density plot AFTER normalization
        ax2 = fig.add_subplot(2, 2, 2)
        for col in sample_cols_subset:
            data_col = self.app.preprocessed_data[col].replace(0, np.nan).dropna()
            if len(data_col) > 0:
                data_to_plot = np.log10(data_col + 1) if needs_log else data_col
                ax2.hist(data_to_plot, bins=50, alpha=0.3, density=True)

        ax2.set_xlabel(xlabel_after, fontsize=fonts['label'])
        ax2.set_ylabel('Density', fontsize=fonts['label'])
        ax2.set_title('After Normalization - Density Plot', fontsize=fonts['title'], fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Box plot BEFORE normalization
        ax3 = fig.add_subplot(2, 2, 3)
        before_box_data = [self.app.data_before_norm[col].replace(0, np.nan).dropna() for col in sample_cols_subset if len(self.app.data_before_norm[col].replace(0, np.nan).dropna()) > 0]
        bp1 = ax3.boxplot(before_box_data, patch_artist=True, labels=[col.split('.')[0][:10] for col in sample_cols_subset])
        for patch in bp1['boxes']:
            patch.set_facecolor('#FF6B6B')
            patch.set_alpha(0.6)

        ax3.set_xlabel('Samples', fontsize=fonts['label'])
        ax3.set_ylabel('Peak Intensity', fontsize=fonts['label'])
        ax3.set_title('Before Normalization - Box Plot', fontsize=fonts['title'], fontweight='bold')
        ax3.tick_params(axis='x', rotation=45, labelsize=fonts['tiny'])
        ax3.grid(True, alpha=0.3, axis='y')

        # Subplot 4: Box plot AFTER normalization
        ax4 = fig.add_subplot(2, 2, 4)
        after_box_data = []
        for col in sample_cols_subset:
            data_col = self.app.preprocessed_data[col].replace(0, np.nan).dropna()
            if len(data_col) > 0:
                after_box_data.append(np.log10(data_col + 1) if needs_log else data_col)
        
        bp2 = ax4.boxplot(after_box_data, patch_artist=True, labels=[col.split('.')[0][:10] for col in sample_cols_subset])
        for patch in bp2['boxes']:
            patch.set_facecolor('#4ECDC4')
            patch.set_alpha(0.6)

        ax4.set_xlabel('Samples', fontsize=fonts['label'])
        ax4.set_ylabel(xlabel_after, fontsize=fonts['label'])
        ax4.set_title('After Normalization - Box Plot', fontsize=fonts['title'], fontweight='bold')
        ax4.tick_params(axis='x', rotation=45, labelsize=fonts['tiny'])
        ax4.grid(True, alpha=0.3, axis='y')

        fig.tight_layout(pad=2.0)
        self._embed_plot(fig, self.app.preprocessing_plot_frame)

    def create_qc_control_chart(self, feature_id, values, sample_names, stats_dict, target_frame, violations=None):
        """Creates a Shewhart control chart for QC monitoring."""
        self._clear_frame(target_frame)
        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        fig = Figure(figsize=(w, h), dpi=dpi)
        self.app.generated_plots['QC_Control_Chart'] = fig
        ax = fig.add_subplot(111)
        
        mean = stats_dict['mean']
        sd = stats_dict['sd']
        rsd = stats_dict['rsd']
        
        # Plot data points
        x = np.arange(len(values))
        ax.plot(x, values, marker='o', markersize=8, linestyle='-', color='#1f77b4', linewidth=2, label='QC Intensity', zorder=5)
        
        # Highlight violations if any
        if violations:
            viol_idx = list(violations.keys())
            ax.scatter(x[viol_idx], values[viol_idx], marker='X', s=180, color='red', 
                       edgecolor='black', label='Out of Control', zorder=10)
        
        # Overlay limit lines
        ax.axhline(mean, color='green', linestyle='-', linewidth=2, label=f'Mean ({mean:.2e})', zorder=4)
        
        # Limits: ±1SD, ±2SD, ±3SD
        ax.axhline(mean + sd, color='orange', linestyle='--', alpha=0.6, label='±1 SD', zorder=3)
        ax.axhline(mean - sd, color='orange', linestyle='--', alpha=0.6, zorder=3)
        
        ax.axhline(mean + 2*sd, color='red', linestyle='--', alpha=0.6, label='±2 SD (Warning)', zorder=3)
        ax.axhline(mean - 2*sd, color='red', linestyle='--', alpha=0.6, zorder=3)
        
        ax.axhline(mean + 3*sd, color='darkred', linestyle='-', linewidth=1.5, alpha=0.8, label='±3 SD (Control Limit)', zorder=3)
        ax.axhline(mean - 3*sd, color='darkred', linestyle='-', linewidth=1.5, alpha=0.8, zorder=3)
        
        # Shading for better visualization
        ax.fill_between(x, mean - 2*sd, mean + 2*sd, color='yellow', alpha=0.1, zorder=1)
        ax.fill_between(x, mean - 3*sd, mean + 3*sd, color='red', alpha=0.05, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('.mzML', '')[:15] for n in sample_names], rotation=45, ha='right', fontsize=fonts['tiny'])
        ax.set_ylabel('Peak Area / Intensity', fontsize=fonts['label'])
        ax.set_title(f"Shewhart Control Chart: {feature_id}\nQC RSD: {rsd:.2f}%", fontsize=fonts['title'], fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=fonts['legend'])
        ax.grid(True, alpha=0.2, linestyle=':')
        fig.tight_layout()
        self._embed_plot(fig, target_frame)

    def _draw_confidence_ellipse(self, ax, x, y, n_std=2.4477, facecolor='none', **kwargs):
        """Draw confidence ellipse for a group of points."""
        if len(x) < 2:
            return

        cov = np.cov(x, y)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)

        mean_x = np.mean(x)
        mean_y = np.mean(y)

        ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)
        ax.add_patch(ellipse)
        return ellipse

    def create_pca_plot(self, pca_result):
        """Create PCA score plot with group colors, confidence ellipses, and hover tooltips."""
        self._clear_frame(self.app.pca_plot_frame)

        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        fig = Figure(figsize=(w, h), dpi=dpi)
        self.app.generated_plots['PCA'] = fig
        ax = fig.add_subplot(111)

        groups = pca_result.get('groups', None)
        show_ellipses = self.app.show_ellipses_var.get()
        show_labels = self.app.show_labels_var.get()

        scatters = []
        canvas = self._embed_plot(fig, self.app.pca_plot_frame)

        if groups:
            configured_order = pca_result.get('group_order', getattr(self.app, 'pca_group_order', []))
            present_groups = set(groups)
            unique_groups = [group for group in configured_order if group in present_groups]
            unique_groups.extend(sorted(present_groups - set(unique_groups)))
            for group in unique_groups:
                group_indices = [i for i, g in enumerate(groups) if g == group]
                group_scores_pc1 = pca_result['scores'][group_indices, 0]
                group_scores_pc2 = pca_result['scores'][group_indices, 1]
                color = self.app.group_colors.get(group, '#4ECDC4')
                sc = ax.scatter(group_scores_pc1, group_scores_pc2, s=100, alpha=0.7, c=color, edgecolors='black', linewidth=1.5, label=group, zorder=3)
                scatters.append((sc, group_indices))
                if show_ellipses and len(group_indices) >= 3:
                    self._draw_confidence_ellipse(ax, group_scores_pc1, group_scores_pc2, color=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax.legend(loc='best', framealpha=0.9, fontsize=fonts['legend'])
        else:
            sc = ax.scatter(pca_result['scores'][:, 0], pca_result['scores'][:, 1], s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
            scatters.append((sc, list(range(len(pca_result['sample_names'])))))

        if show_labels:
            for i, name in enumerate(pca_result['sample_names']):
                short_name = re.sub(r'\.mzML.*', '', name)
                ax.annotate(short_name, (pca_result['scores'][i, 0], pca_result['scores'][i, 1]), fontsize=fonts['tiny'], alpha=0.6, zorder=2)

        title = "PCA Score Plot" + (" with 95% Confidence Ellipses" if show_ellipses and groups else "")
        if 'permanova_pvalue' in pca_result and pca_result['permanova_pvalue'] is not None:
            title += f"\nPERMANOVA p-value: {pca_result['permanova_pvalue']:.3f}"
        if 'permdisp_pvalue' in pca_result and pca_result['permdisp_pvalue'] is not None:
            if 'permanova_pvalue' in pca_result and pca_result['permanova_pvalue'] is not None:
                title += f" | PERMDISP p-value: {pca_result['permdisp_pvalue']:.3f}"
            else:
                title += f"\nPERMDISP p-value: {pca_result['permdisp_pvalue']:.3f}"
            
        ax.set_xlabel(f"PC1 ({pca_result['variance_explained'][0]:.1f}%)", fontsize=fonts['label'])
        ax.set_ylabel(f"PC2 ({pca_result['variance_explained'][1]:.1f}%)", fontsize=fonts['label'])
        ax.set_title(title, fontsize=fonts['title'], fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
        ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
        ax.axvline(x=0, color='k', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)

        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points", bbox=dict(boxstyle="round", fc="yellow", alpha=0.9), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"), zorder=10)
        annot.set_visible(False)

        def on_hover(event):
            if event.inaxes == ax:
                is_contained = False
                for sc, indices in scatters:
                    cont, ind = sc.contains(event)
                    if cont:
                        idx = indices[ind["ind"][0]]
                        pos = sc.get_offsets()[ind["ind"][0]]
                        annot.xy = pos
                        sample_col = pca_result['sample_names'][idx]
                        clean_name = re.sub(r'\.mzML.*', '', sample_col)
                        group_name = pca_result['groups'][idx] if 'groups' in pca_result else 'Ungrouped'
                        annot.set_text(f"ID: {clean_name}\nGrp: {group_name}")
                        annot.set_visible(True)
                        canvas.draw_idle()
                        is_contained = True
                        break
                if not is_contained and annot.get_visible():
                    annot.set_visible(False)
                    canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_hover)

    def create_volcano_plot(self, result, pval_thresh, fc_thresh):
        """Create interactive volcano plot with hover tooltips."""
        self._clear_frame(self.app.volcano_plot_frame)

        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        fig = Figure(figsize=(w, h), dpi=dpi)
        self.app.generated_plots['Univariate Screening'] = fig
        ax = fig.add_subplot(111)

        fc = result['fold_changes']
        pvals = np.clip(np.array(result['pvalues'], dtype=float), np.finfo(float).tiny, 1.0)
        log_pvals = -np.log10(pvals)

        significant_up = (fc >= np.log2(fc_thresh)) & (pvals <= pval_thresh)
        significant_down = (fc <= -np.log2(fc_thresh)) & (pvals <= pval_thresh)
        not_significant = ~(significant_up | significant_down)

        ax.scatter(fc[not_significant], log_pvals[not_significant], c='gray', s=20, alpha=0.5, label='Not Significant')
        sc_up = ax.scatter(fc[significant_up], log_pvals[significant_up], c='red', s=30, alpha=0.7, label=f'Higher in {result["group2"]}')
        sc_down = ax.scatter(fc[significant_down], log_pvals[significant_down], c='blue', s=30, alpha=0.7, label=f'Lower in {result["group2"]}')

        ax.axhline(-np.log10(pval_thresh), color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(np.log2(fc_thresh), color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(-np.log2(fc_thresh), color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel(f'log2(Fold Change) [{result["group2"]} / {result["group1"]}]', fontsize=fonts['label'])
        ylabel = '-log10(p-value)'
        if result.get('correction_method', 'None') != 'None':
            ylabel = f'-log10({result["correction_method"]} p-value)'
        ax.set_ylabel(ylabel, fontsize=fonts['label'])
        ax.set_title('Volcano Plot', fontsize=fonts['title'], fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points", bbox=dict(boxstyle="round", fc="yellow", alpha=0.8), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        up_idx, down_idx = np.where(significant_up)[0], np.where(significant_down)[0]
        canvas = self._embed_plot(fig, self.app.volcano_plot_frame)

        def on_hover(event):
            if event.inaxes == ax:
                cont, ind = sc_up.contains(event)
                if cont:
                    idx = up_idx[ind["ind"][0]]
                    annot.xy = sc_up.get_offsets()[ind["ind"][0]]
                else:
                    cont, ind = sc_down.contains(event)
                    if cont:
                        idx = down_idx[ind["ind"][0]]
                        annot.xy = sc_down.get_offsets()[ind["ind"][0]]

                if cont:
                    feat_id = result['feature_ids'][idx]
                    raw_p = result.get('raw_pvalues', result['pvalues'])[idx]
                    text = f"{feat_id}\nRaw P: {raw_p:.2e}"
                    if result.get('correction_method', 'None') != 'None':
                        text += f"\nAdj P: {result['pvalues'][idx]:.2e}"
                    annot.set_text(text)
                    annot.set_visible(True)
                    canvas.draw_idle()
                elif annot.get_visible():
                    annot.set_visible(False)
                    canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_hover)

        # --- Context Menu for Exporting Subsets ---
        menu = tk.Menu(self.app, tearoff=0)
        menu.add_command(label="Export Higher Abundance Features (Red)", command=lambda: self.app.export_volcano_subset("UP"))
        menu.add_command(label="Export Lower Abundance Features (Blue)", command=lambda: self.app.export_volcano_subset("DOWN"))
        menu.add_separator()
        menu.add_command(label="Export All Significant Features", command=lambda: self.app.export_volcano_subset("BOTH"))
        
        def show_context_menu(event):
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
                
        canvas.get_tk_widget().bind("<Button-3>", show_context_menu)

    def create_plsda_plot(self, result):
        """Create PLS-DA score plot with 95% Confidence Ellipses AND VIP bar chart."""
        self._clear_frame(self.app.plsda_plot_frame)

        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        fig = Figure(figsize=(w * 2, h), dpi=dpi)
        self.app.generated_plots['PLS-DA'] = fig

        ax1 = fig.add_subplot(121)
        groups = result['groups']
        configured_order = result.get('group_order', getattr(self.app, 'plsda_group_order', []))
        present_groups = set(groups)
        unique_groups = [group for group in configured_order if group in present_groups]
        unique_groups.extend(sorted(present_groups - set(unique_groups)))

        for group in unique_groups:
            indices = [i for i, g in enumerate(groups) if g == group]
            scores_lv1, scores_lv2 = result['scores'][indices, 0], result['scores'][indices, 1]
            color = self.app.group_colors.get(group, '#4ECDC4')
            ax1.scatter(scores_lv1, scores_lv2, s=100, alpha=0.7, c=color, edgecolors='black', linewidth=1.5, label=group)
            self._draw_confidence_ellipse(ax1, scores_lv1, scores_lv2, facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5)

        r2_val, q2_val = result.get('r2_y', 0), result.get('q2_y', 0)
        ax1.set_xlabel('LV1', fontsize=fonts['label'])
        ax1.set_ylabel('LV2', fontsize=fonts['label'])
        ax1.set_title(f"PLS-DA Score Plot (95% Confidence Ellipses)\nR²Y: {r2_val:.3f} | Q²: {q2_val:.3f}", fontsize=fonts['title'], fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(122)
        vips = result['vip_scores']
        feature_ids = self.app.screened_data.iloc[:, 0].values
        top_idx = np.argsort(vips)[-15:]
        ax2.barh(range(15), vips[top_idx], color='purple', alpha=0.7)
        ax2.set_yticks(range(15))
        ax2.set_yticklabels([feature_ids[i] for i in top_idx], fontsize=fonts['annot'])
        ax2.axvline(1.0, color='red', linestyle='--', label='VIP > 1.0 cutoff')
        ax2.set_xlabel('VIP Score', fontsize=fonts['label'])
        ax2.set_title("Top 15 Variable Importance in Projection (VIP)", fontsize=fonts['title'], fontweight='bold')
        ax2.legend(loc='lower right')

        fig.tight_layout()
        self._embed_plot(fig, self.app.plsda_plot_frame)

    def create_pls_val_plot(self, perm_results):
        """Creates the PLS-DA permutation validation plot."""
        self._clear_frame(self.app.pls_val_plot_frame)

        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        fig = Figure(figsize=(w, h), dpi=dpi)
        self.app.generated_plots['PLS-DA Validation'] = fig
        ax = fig.add_subplot(111)

        ax.scatter(perm_results['corrs'], perm_results['r2_perms'], color='#1f77b4', alpha=0.6, label='R² (Permuted)')
        ax.scatter(perm_results['corrs'], perm_results['q2_perms'], color='#2ca02c', alpha=0.6, label='Q² (Permuted)')
        ax.scatter([1.0], [perm_results['r2_actual']], color='#1f77b4', marker='*', s=150, label='R² (Actual)')
        ax.scatter([1.0], [perm_results['q2_actual']], color='#2ca02c', marker='*', s=150, label='Q² (Actual)')

        if len(perm_results['corrs']) > 1:
            # Draw lines connecting the calculated intercepts at correlation 0 directly to the actual points at correlation 1.0
            # This ensures the trend line strictly "connects" to the unpermuted result star.
            ax.plot([0, 1.0], [perm_results['r2_int'], perm_results['r2_actual']], color='#1f77b4', linestyle='--', alpha=0.7)
            ax.plot([0, 1.0], [perm_results['q2_int'], perm_results['q2_actual']], color='#2ca02c', linestyle='--', alpha=0.7)

        title = (f"PLS-DA Permutation Test (n={perm_results['n_perms']})\n"
                 f"R²Y = {perm_results['r2_actual']:.3f}, Q² = {perm_results['q2_actual']:.3f}\n"
                 f"R² int = {perm_results['r2_int']:.3f}, Q² int = {perm_results['q2_int']:.3f} | p-value = {perm_results['p_val']:.3f}")
        ax.set_title(title, fontsize=fonts['title'], fontweight='bold')
        ax.set_xlabel("Correlation between permuted and original Y", fontsize=fonts['label'])
        ax.set_ylabel("Score (R² / Q²)", fontsize=fonts['label'])
        ax.set_xlim(0, 1.05)
        ax.legend(loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()

        self._embed_plot(fig, self.app.pls_val_plot_frame)

    def create_rf_plot(self, result):
        """Create Random Forest feature importance plot."""
        self._clear_frame(self.app.rf_plot_frame)

        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        
        has_curve = result.get('cv_curve') is not None
        has_roc = result.get('roc_data') is not None
        num_subplots = 1 + int(has_curve) + int(has_roc)
        
        fig = Figure(figsize=(w * num_subplots, h), dpi=dpi)
            
        self.app.generated_plots['Random Forest'] = fig
        
        ax = fig.add_subplot(1, num_subplots, 1)

        top_features = result['top_features']
        feature_names, importances = [f[0] for f in top_features], [f[1] for f in top_features]
        y_pos = np.arange(len(feature_names))

        ax.barh(y_pos, importances, align='center', color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=fonts['annot'])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=fonts['label'])
        ax.set_title(f'Random Forest - Top {len(feature_names)} Features', fontsize=fonts['title'], fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plot_idx = 2
        
        if has_curve:
            ax_curve = fig.add_subplot(1, num_subplots, plot_idx)
            curve = result['cv_curve']
            features = curve['features']
            scores = np.array(curve['scores'])
            stds = np.array(curve['stds'])
            
            ax_curve.plot(features, scores, marker='o', linestyle='-', color='darkorange', linewidth=2)
            ax_curve.fill_between(features, scores - stds, scores + stds, color='darkorange', alpha=0.2)
            
            top_n_val = len(feature_names)
            if top_n_val in features:
                idx = features.index(top_n_val)
                ax_curve.scatter([top_n_val], [scores[idx]], color='red', s=100, zorder=5, label=f'Top {top_n_val} Features')
                ax_curve.legend(loc='lower right', fontsize=fonts['legend'])
                
            ax_curve.set_xlabel('Number of Features', fontsize=fonts['label'])
            ax_curve.set_ylabel('Cross-Validation Accuracy', fontsize=fonts['label'])
            ax_curve.set_title('Accuracy vs Number of Features', fontsize=fonts['title'], fontweight='bold')
            ax_curve.grid(True, alpha=0.3)
            plot_idx += 1
            
        if has_roc:
            ax_roc = fig.add_subplot(1, num_subplots, plot_idx)
            roc = result['roc_data']
            ax_roc.plot(roc['fpr'], roc['tpr'], color='darkorange', lw=2, label=f"AUC = {roc['auc']:.3f}")
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate', fontsize=fonts['label'])
            ax_roc.set_ylabel('True Positive Rate', fontsize=fonts['label'])
            ax_roc.set_title(f"ROC Curve ({roc['group2']} vs {roc['group1']})", fontsize=fonts['title'], fontweight='bold')
            ax_roc.legend(loc="lower right", fontsize=fonts['legend'])
            ax_roc.grid(True, alpha=0.3)

        fig.tight_layout()
        self._embed_plot(fig, self.app.rf_plot_frame)

    def create_rf_permutation_plot(self, perm_results):
        """Creates the Random Forest permutation validation plot."""
        self._clear_frame(self.app.rf_plot_frame)
        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        fig = Figure(figsize=(w, h), dpi=dpi)
        self.app.generated_plots['Random Forest Validation'] = fig
        ax = fig.add_subplot(111)

        perm_accs = perm_results['perm_accs']
        actual_acc = perm_results['actual_acc']
        p_val = perm_results['p_val']

        # Plot histogram of null distribution
        ax.hist(perm_accs, bins=15, color='lightgray', edgecolor='white', alpha=0.8, label='Permuted Accuracies')
        
        # Mark actual accuracy
        ax.axvline(actual_acc, color='red', linestyle='--', linewidth=2, label=f'Actual Accuracy ({actual_acc:.3f})')
        
        # Annotate p-value
        title = (f"Random Forest Permutation Test (n={perm_results['n_perms']})\n"
                 f"Empirical p-value: {p_val:.4f}")
        ax.set_title(title, fontsize=fonts['title'], fontweight='bold')
        ax.set_xlabel("Classification Accuracy", fontsize=fonts['label'])
        ax.set_ylabel("Frequency", fontsize=fonts['label'])
        ax.legend(loc='upper left', fontsize=fonts['legend'])
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')
        
        fig.tight_layout()
        self._embed_plot(fig, self.app.rf_plot_frame)

    def create_heatmap_plot(self, ordered_data, feature_labels, final_sample_names, col_linkage, row_linkage, n_bio, n_qc, cmap='coolwarm', target_frame=None, plot_title=None, plot_key='Heatmap', column_colors=None, column_groups=None, show_group_legend=False):
        """Creates the heatmap plot."""
        if target_frame is None:
            target_frame = self.app.heatmap_plot_frame
            
        self._clear_frame(target_frame)

        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        
        # Accommodate extra width if plotting an extra dendrogram
        fig_width = w * 1.6 if row_linkage is not None else w * 1.5
        fig = Figure(figsize=(fig_width, h * 1.1), dpi=dpi)
        self.app.generated_plots[plot_key] = fig
        
        # Determine if we should show the column dendrogram
        show_col_dendro = col_linkage is not None and n_bio > 1
        h_ratios = [1, 0.2, 5] if show_col_dendro else [0.01, 0.2, 5]

        if row_linkage is not None:
            gs = GridSpec(3, 2, width_ratios=[1, 7], height_ratios=h_ratios, hspace=0.02, wspace=0.01)
            gs.update(left=0.05, right=0.82, top=0.9)
            ax_dendro_col = fig.add_subplot(gs[0, 1])
            ax_group_colors = fig.add_subplot(gs[1, 1])
            ax_dendro_row = fig.add_subplot(gs[2, 0])
            ax_heatmap = fig.add_subplot(gs[2, 1])
            
            sch.dendrogram(row_linkage, orientation='left', ax=ax_dendro_row, no_labels=True, color_threshold=0, above_threshold_color='black')
            ax_dendro_row.set_ylim(0, 10 * len(feature_labels))
            ax_dendro_row.invert_yaxis()  # Invert to align with imshow
            ax_dendro_row.axis('off')
        else:
            gs = GridSpec(3, 1, height_ratios=h_ratios, hspace=0.02)
            gs.update(left=0.1, right=0.82, top=0.9)
            ax_dendro_col = fig.add_subplot(gs[0, 0])
            ax_group_colors = fig.add_subplot(gs[1, 0])
            ax_heatmap = fig.add_subplot(gs[2, 0])

        if show_col_dendro:
            sch.dendrogram(col_linkage, orientation='top', ax=ax_dendro_col, no_labels=True, color_threshold=0, above_threshold_color='black')
            ax_dendro_col.set_xlim(0, 10 * (n_bio + n_qc))

        ax_dendro_col.axis('off')
        
        if column_colors:
            rgb_array = np.array([[to_rgb(c) for c in column_colors]])
            ax_group_colors.imshow(rgb_array, aspect='auto', interpolation='nearest')
            ax_group_colors.set_xticks([])
            ax_group_colors.set_yticks([])
            for spine in ax_group_colors.spines.values():
                spine.set_visible(False)
        else:
            ax_group_colors.axis('off')
        
        # Clamp the colormap limits to [-3, 3] so outliers don't wash out the Z-score variance
        cax = ax_heatmap.imshow(ordered_data, aspect='auto', cmap=cmap, interpolation='nearest', vmin=-3, vmax=3)

        if n_bio > 0 and n_qc > 0:
            ax_heatmap.axvline(x=n_bio - 0.5, color='black', linewidth=1, linestyle='--', alpha=0.9)

        ax_heatmap.set_xticks(range(len(final_sample_names)))
        ax_heatmap.set_xticklabels(final_sample_names, rotation=45, ha='right', fontsize=fonts['tiny'])
        export_autoscale_text = list(ax_heatmap.get_xticklabels())

        for tick_label in ax_heatmap.get_xticklabels():
            group_text = tick_label.get_text().split('[')[-1].replace(']', '')
            if group_text in self.app.group_colors:
                tick_label.set_color(self.app.group_colors[group_text])
                tick_label.set_fontweight('bold')

        if 0 < len(feature_labels) <= 100:
            ax_heatmap.set_yticks(range(len(feature_labels)))
            ax_heatmap.set_yticklabels(feature_labels, fontsize=fonts['annot'])
        else:
            ax_heatmap.set_yticks([])
            ax_heatmap.set_yticklabels([])

        # Move y-axis labels to the right side to prevent overlap with the dendrogram
        ax_heatmap.yaxis.tick_right()
        ax_heatmap.yaxis.set_label_position("right")

        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.6])
        fig.colorbar(cax, cax=cbar_ax, label="Z-score")

        if show_group_legend and column_groups:
            visible_groups = list(dict.fromkeys(column_groups))
            handles = [Patch(facecolor=self.app.group_colors.get(group, '#CCCCCC'),
                             edgecolor='none', label=group) for group in visible_groups]
            group_legend = fig.legend(handles=handles, title="Groups", loc='upper right',
                                      bbox_to_anchor=(0.92, 0.9), fontsize=fonts['tiny'],
                                      title_fontsize=fonts['annot'], frameon=True)
            export_autoscale_text.extend(group_legend.get_texts())
            export_autoscale_text.append(group_legend.get_title())
        
        if plot_title is None:
            plot_title = "Heatmap of Top Selected Features" + (" (Clustered)" if row_linkage is not None else " (Sorted by RT)")
        fig.suptitle(plot_title, fontsize=fonts['title'], fontweight='bold', y=0.95)

        # Let high-resolution export scale sample labels and their group legend.
        fig._export_autoscale_text = export_autoscale_text

        # --- Interactive Tooltip Logic ---
        annot = ax_heatmap.annotate("", xy=(0,0), xytext=(20, 20), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"), zorder=10)
        annot.set_visible(False)

        def on_hover(event):
            if event.inaxes == ax_heatmap:
                # Determine fractional index of the cell being hovered
                x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                if 0 <= x < len(final_sample_names) and 0 <= y < len(feature_labels):
                    val = ordered_data[y, x]
                    sample_name = final_sample_names[x].replace('\n', ' ')
                    feat_name = feature_labels[y]
                    annot.xy = (event.xdata, event.ydata)
                    annot.set_text(f"Sample: {sample_name}\nFeature: {feat_name}\nZ-score: {val:.2f}")
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
            elif annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        self._embed_plot(fig, target_frame, hover_handler=on_hover)

    def create_hca_only_plot(self, linkage_matrix, labels, title="Sample HCA Dendrogram",
                             target_frame=None, label_colors=None, visible_groups=None,
                             plot_key='HCA Only'):
        """Create a sample-only HCA dendrogram without a heatmap."""
        if target_frame is None:
            target_frame = self.app.hca_plot_frame

        self._clear_frame(target_frame)
        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()

        fig = Figure(figsize=(w * 1.35, h), dpi=dpi)
        self.app.generated_plots[plot_key] = fig
        ax = fig.add_subplot(111)

        sch.dendrogram(
            linkage_matrix,
            labels=labels,
            orientation='top',
            ax=ax,
            leaf_rotation=45,
            leaf_font_size=fonts['tick'],
            color_threshold=0,
            above_threshold_color='black'
        )

        ax.set_title(title, fontsize=fonts['title'], fontweight='bold', pad=18)
        ax.set_ylabel("Distance", fontsize=fonts['label'])
        ax.grid(True, axis='y', alpha=0.25, linestyle=':')

        export_autoscale_text = list(ax.get_xticklabels())
        if label_colors:
            for tick_label in ax.get_xticklabels():
                label = tick_label.get_text()
                if label in label_colors:
                    tick_label.set_color(label_colors[label])
                    tick_label.set_fontweight('bold')

        if visible_groups:
            handles = [
                Patch(facecolor=self.app.group_colors.get(group, '#CCCCCC'),
                      edgecolor='none', label=group)
                for group in visible_groups
            ]
            legend = ax.legend(handles=handles, title="Groups", loc='upper right',
                               fontsize=fonts['legend'], title_fontsize=fonts['legend'],
                               frameon=True)
            export_autoscale_text.extend(legend.get_texts())
            export_autoscale_text.append(legend.get_title())

        fig.tight_layout()
        fig._export_autoscale_text = export_autoscale_text
        self._embed_plot(fig, target_frame)

    def create_venn_plot(self, sets, group_names, target_frame=None, plot_key='Venn Diagram', plot_title=None):
        """
        Create a schematic Venn diagram (2 or 3 sets) using matplotlib patches.
        Does not require matplotlib-venn library.
        """
        if target_frame is None:
            target_frame = self.app.venn_plot_frame
            
        self._clear_frame(target_frame)
        fonts = self._get_font_sizes()
        
        w, h, dpi = self._get_plot_dims()
        # Increase width to accommodate side-by-side plots
        fig = Figure(figsize=(w * 1.8, h), dpi=dpi)
        self.app.generated_plots[plot_key] = fig
        
        # --- Left: Venn Diagram ---
        ax = fig.add_subplot(121)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Use application group colors if available, otherwise fallback to defaults
        default_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#E5CCFF', '#99CCFF']
        colors = [self.app.group_colors.get(g, default_colors[i % len(default_colors)]) for i, g in enumerate(group_names)]
        
        venn_drawn = False
        # === Logic for >3 Groups (UpSet-style Bar Plot) ===
        if len(group_names) > 3:
            # Calculate intersection counts dynamically
            feature_map = defaultdict(list)
            for g_name, f_set in sets.items():
                for f in f_set:
                    feature_map[f].append(g_name)
            
            # Count signatures
            sig_counts = defaultdict(int)
            for f, g_list in feature_map.items():
                g_list.sort()
                sig = "\n+\n".join(g_list) if len(g_list) < 4 else " & ".join(g_list)
                sig_counts[sig] += 1
            
            # Sort by count (descending) and take top 15
            sorted_sigs = sorted(sig_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            labels = [x[0] for x in sorted_sigs]
            values = [x[1] for x in sorted_sigs]
            
            # Plot Bar Chart on the Left Axis
            bars = ax.bar(range(len(values)), values, color='teal', alpha=0.6)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fonts['tiny'])
            ax.set_ylabel("Intersection Size")
            ax.set_title(f"Top Intersections ({len(group_names)} Groups)", fontsize=fonts['label'], fontweight='bold')
            
            # Annotate bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom', fontsize=fonts['tiny'])
            
            # Mark as drawn so we skip the circle logic
            venn_drawn = True

        if not venn_drawn and HAS_VENN:
            try:
                total_count = len(set.union(*[sets[g] for g in group_names]))
                if len(group_names) == 2:
                    v = venn2([sets[g] for g in group_names], set_labels=None, set_colors=colors[:2], ax=ax, alpha=0.5)
                    if v:
                        for s_id in ['10', '01', '11']:
                            lbl = v.get_label_by_id(s_id)
                            if lbl and lbl.get_text():
                                val = int(lbl.get_text())
                                pct = (val / total_count) * 100 if total_count > 0 else 0
                                lbl.set_text(f"{val}\n({pct:.1f}%)")
                                lbl.set_fontsize(fonts['tick'])
                                lbl.set_ha('center')
                                # Move unique features outward
                                if s_id in ['10', '01']:
                                    x, y = lbl.get_position()
                                    lbl.set_position((x * 1.4, y))
                    ax.set_xlim(-1.1, 1.1)
                    ax.set_ylim(-1.0, 1.0)
                    venn_drawn = True
                elif len(group_names) == 3:
                    v = venn3([sets[g] for g in group_names], set_labels=None, set_colors=colors[:3], ax=ax, alpha=0.5)
                    if v:
                        for s_id in ['100', '010', '001', '110', '101', '011', '111']:
                            lbl = v.get_label_by_id(s_id)
                            if lbl and lbl.get_text():
                                val = int(lbl.get_text())
                                pct = (val / total_count) * 100 if total_count > 0 else 0
                                lbl.set_text(f"{val}\n({pct:.1f}%)")
                                lbl.set_fontsize(fonts['tiny'] if s_id not in ['100', '010', '001'] else fonts['tick'])
                                lbl.set_ha('center')
                                # Move unique features outward
                                if s_id in ['100', '010', '001']:
                                    x, y = lbl.get_position()
                                    new_x, new_y = x * 1.6, y * 1.6
                                    lbl.set_position((new_x, new_y))
                    ax.set_xlim(-1.1, 1.1)
                    ax.set_ylim(-1.1, 1.1)
                    venn_drawn = True
                if venn_drawn:
                    title = plot_title if plot_title else "Feature Overlap (Proportional)"
                    ax.set_title(title, fontsize=fonts['title'], fontweight='bold', y=0.90)
            except Exception:
                venn_drawn = False

        if not venn_drawn and len(group_names) == 2:
            if not HAS_VENN:
                ax.text(0.5, 0.0, "(Install 'matplotlib-venn' for proportional plots)", ha='center', fontsize=8, color='gray', transform=ax.transAxes)
            # 2-Set Venn
            A = sets[group_names[0]]
            B = sets[group_names[1]]
            
            only_A = len(A - B)
            only_B = len(B - A)
            AB = len(A & B)

            total_count = len(A | B)
            def fmt(val):
                pct = (val / total_count) * 100 if total_count > 0 else 0
                return f"{val}\n({pct:.1f}%)"
            
            # Draw circles
            c1 = Circle((0.35, 0.5), 0.25, alpha=0.5, color=colors[0], label=group_names[0])
            c2 = Circle((0.65, 0.5), 0.25, alpha=0.5, color=colors[1], label=group_names[1])
            
            ax.add_patch(c1)
            ax.add_patch(c2)
            
            # Add labels
            ax.text(0.05, 0.5, fmt(only_A), ha='right', va='center', fontsize=fonts['label'], fontweight='bold')
            ax.text(0.95, 0.5, fmt(only_B), ha='left', va='center', fontsize=fonts['label'], fontweight='bold')
            ax.text(0.5, 0.5, fmt(AB), ha='center', va='center', fontsize=fonts['label'], fontweight='bold')
            
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(0, 1.0)
            
        elif not venn_drawn and len(group_names) == 3:
            # 3-Set Venn
            A = sets[group_names[0]]
            B = sets[group_names[1]]
            C = sets[group_names[2]]
            
            # Calculate intersections
            ABC = len(A & B & C)
            AB_noC = len((A & B) - C)
            AC_noB = len((A & C) - B)
            BC_noA = len((B & C) - A)
            A_only = len(A - B - C)
            B_only = len(B - A - C)
            C_only = len(C - A - B)

            total_count = len(A | B | C)
            def fmt(val):
                pct = (val / total_count) * 100 if total_count > 0 else 0
                return f"{val}\n({pct:.1f}%)"
            
            # Draw circles (Schematic positions)
            r = 0.25
            # Top, Left, Right
            pos = [(0.5, 0.65), (0.35, 0.4), (0.65, 0.4)]
            
            for i, p in enumerate(pos):
                circle = Circle(p, r, alpha=0.4, color=colors[i])
                ax.add_patch(circle)

            # Add numbers (approximate centroids for standard venn)
            # Top (A only)
            ax.text(0.5, 0.95, fmt(A_only), ha='center', va='center', fontsize=fonts['tick'])
            # Left (B only)
            ax.text(0.1, 0.2, fmt(B_only), ha='right', va='center', fontsize=fonts['tick'])
            # Right (C only)
            ax.text(0.9, 0.2, fmt(C_only), ha='left', va='center', fontsize=fonts['tick'])
            
            # Intersections
            ax.text(0.5, 0.5, fmt(ABC), ha='center', va='center', fontsize=fonts['label'], fontweight='bold') # Center
            ax.text(0.40, 0.55, fmt(AB_noC), ha='center', va='center', fontsize=fonts['tiny']) # A & B
            ax.text(0.60, 0.55, fmt(AC_noB), ha='center', va='center', fontsize=fonts['tiny']) # A & C
            ax.text(0.5, 0.30, fmt(BC_noA), ha='center', va='center', fontsize=fonts['tiny']) # B & C
            
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)

        if len(group_names) <= 3:
            legend_handles = [Patch(color=colors[i], label=group_names[i]) for i in range(len(group_names))]
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), frameon=False, ncol=len(group_names), fontsize=fonts['legend'])

        if not venn_drawn:
            title = plot_title if plot_title else "Feature Overlap Venn Diagram"
            ax.set_title(title, fontsize=fonts['title'], fontweight='bold', y=0.90)
        
        # --- Right: Bar Chart ---
        ax2 = fig.add_subplot(122)
        counts = [len(sets[g]) for g in group_names]
        
        bars = ax2.bar(group_names, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom', fontsize=fonts['tick'], fontweight='bold')
        
        ax2.set_title("Total Features per Group", fontsize=fonts['title'], fontweight='bold')
        ax2.set_ylabel("Number of Detected Features")
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=15)
        
        fig.tight_layout(pad=3.0)
        
        canvas = self._embed_plot(fig, target_frame)

        # Context Menu for Exporting Subsets (Only for Group Comparison Venn)
        if target_frame == self.app.venn_plot_frame:
            menu = tk.Menu(self.app, tearoff=0)
            menu.add_command(label="Export Common Features (Intersection)", command=lambda: self.app.export_venn_data_subset('COMMON'))
            menu.add_separator()
            for g in group_names:
                menu.add_command(label=f"Export Unique to {g}", command=lambda g=g: self.app.export_venn_data_subset(f'UNIQUE_{g}'))
            
            menu.add_separator()
            menu.add_command(label="Export All Unique Features (Diff)", command=lambda: self.app.export_venn_data_subset('UNIQUE_COMBINED'))

            def show_context_menu(event):
                try:
                    menu.tk_popup(event.x_root, event.y_root)
                finally:
                    menu.grab_release()
                    
            canvas.get_tk_widget().bind("<Button-3>", show_context_menu)

    def create_upset_plot(self, sets, group_names, target_frame, plot_key='UpSet Plot'):
        """Creates an UpSet-style plot for comparing feature intersections across >=2 groups."""
        self._clear_frame(target_frame)
        fonts = self._get_font_sizes()
        
        w, h, dpi = self._get_plot_dims()
        fig = Figure(figsize=(w * 1.8, h), dpi=dpi)
        self.app.generated_plots[plot_key] = fig
        
        default_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#E5CCFF', '#99CCFF']
        colors = [self.app.group_colors.get(g, default_colors[i % len(default_colors)]) for i, g in enumerate(group_names)]
        
        gs = GridSpec(2, 1, figure=fig, height_ratios=[2.5, 1], hspace=0.05)
        ax_bar = fig.add_subplot(gs[0, 0])
        ax_matrix = fig.add_subplot(gs[1, 0], sharex=ax_bar)
        
        feature_map = defaultdict(list)
        for g_name, f_set in sets.items():
            for f in f_set:
                feature_map[f].append(g_name)
        
        sig_counts = defaultdict(int)
        for f, g_list in feature_map.items():
            sig = tuple(sorted(g_list))
            sig_counts[sig] += 1
        
        sorted_sigs = sorted(sig_counts.items(), key=lambda x: x[1], reverse=True)
        signatures = [x[0] for x in sorted_sigs]
        values = [x[1] for x in sorted_sigs]
        
        if not values:
            ax_bar.text(0.5, 0.5, "No intersections found", ha='center', va='center')
            self._embed_plot(fig, target_frame)
            return

        bars = ax_bar.bar(range(len(values)), values, color='teal', alpha=0.6)
        ax_bar.set_ylabel("Intersection Size", fontsize=fonts['label'])
        ax_bar.set_title(f"Top {len(values)} Intersections ({len(group_names)} Groups)", fontsize=fonts['title'], fontweight='bold')
        ax_bar.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_bar.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom', fontsize=fonts['tiny'])
            
        ax_matrix.invert_yaxis()
        ax_matrix.set_yticks(range(len(group_names)))
        ax_matrix.set_yticklabels(group_names, fontsize=fonts['tick'])
        ax_matrix.set_xticks(range(len(values)))
        ax_matrix.tick_params(axis='x', bottom=False, labelbottom=False)
        
        for spine in ax_matrix.spines.values():
            spine.set_visible(False)
        ax_matrix.tick_params(axis='y', left=False)
        
        for i in range(len(group_names)):
            ax_matrix.axhline(i, color='gray', alpha=0.2, zorder=1)
            
        for col_idx, sig in enumerate(signatures):
            row_indices = [group_names.index(g) for g in sig]
            ax_matrix.scatter([col_idx]*len(group_names), range(len(group_names)), color='lightgray', s=50, zorder=2)
            ax_matrix.scatter([col_idx]*len(row_indices), row_indices, color='black', s=80, zorder=3)
            if len(row_indices) > 1:
                ax_matrix.plot([col_idx, col_idx], [min(row_indices), max(row_indices)], color='black', lw=2, zorder=2)
                
        fig.tight_layout(pad=3.0)
        canvas = self._embed_plot(fig, target_frame)

        # Context Menu for Exporting Subsets
        if target_frame == self.app.venn_plot_frame or target_frame == self.app.model_venn_plot_frame:
            menu = tk.Menu(self.app, tearoff=0)
            menu.add_command(label="Export Common Features (Intersection)", command=lambda: self.app.export_venn_data_subset('COMMON'))
            menu.add_separator()
            for g in group_names:
                menu.add_command(label=f"Export Unique to {g}", command=lambda g=g: self.app.export_venn_data_subset(f'UNIQUE_{g}'))
            menu.add_separator()
            menu.add_command(label="Export All Unique Features (Diff)", command=lambda: self.app.export_venn_data_subset('UNIQUE_COMBINED'))
        elif hasattr(self.app, 'upset_plot_frame') and target_frame == self.app.upset_plot_frame:
            menu = tk.Menu(self.app, tearoff=0)
            menu.add_command(label="Export Common Features (Intersection)", command=lambda: self.app.export_upset_data_subset('COMMON'))
            menu.add_separator()
            for g in group_names:
                menu.add_command(label=f"Export Unique to {g}", command=lambda g=g: self.app.export_upset_data_subset(f'UNIQUE_{g}'))
            menu.add_separator()
            menu.add_command(label="Export All Unique Features (Diff)", command=lambda: self.app.export_upset_data_subset('UNIQUE_COMBINED'))
        else:
            menu = None

        if menu:
            def show_context_menu(event):
                menu.tk_popup(event.x_root, event.y_root)
            canvas.get_tk_widget().bind("<Button-3>", show_context_menu)

    def create_feature_distribution_plot(self, feature_id, plot_data, group_labels, sample_names=None, ylabel="Peak Intensity", show_points=True, show_violin=True, show_boxplot=True, font_scale=1.0):
        """Creates a distribution plot (violin/box/swarm) for a single feature's intensity across groups."""
        self._clear_frame(self.app.feature_viewer_plot_frame)

        w, h, dpi = self._get_plot_dims()
        fonts = self._get_font_sizes()
        font_scale = max(0.6, min(float(font_scale), 3.0))
        fonts = {key: max(1, int(value * font_scale)) for key, value in fonts.items()}
        fig = Figure(figsize=(w, h), dpi=dpi)
        self.app.generated_plots['Feature_Viewer'] = fig
        ax = fig.add_subplot(111)

        positions = np.arange(1, len(plot_data) + 1)
        
        if show_violin:
            valid_vp_data = []
            valid_vp_pos = []
            for i, d in enumerate(plot_data):
                # Violin plots need variance to calculate KDE
                if len(d) > 1 and np.var(d) > 0:
                    valid_vp_data.append(d)
                    valid_vp_pos.append(positions[i])
                elif len(d) > 1:
                    # Very small jitter if variance is 0 to avoid KDE crash
                    valid_vp_data.append(d + np.random.normal(0, 1e-8, len(d)))
                    valid_vp_pos.append(positions[i])
                    
            if len(valid_vp_data) > 0:
                # Lower bw_method makes the KDE fit tighter and the ends pointier
                vp = ax.violinplot(valid_vp_data, positions=valid_vp_pos, showmeans=False, showmedians=False, showextrema=False, widths=0.7, bw_method=0.4)
                for i, pc in enumerate(vp['bodies']):
                    group_name = group_labels[valid_vp_pos[i] - 1]
                    color = self.app.group_colors.get(group_name, '#CCCCCC')
                    pc.set_facecolor(color)
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.5)

        if show_boxplot:
            # Adjust box width dynamically depending on whether violin plot is active
            widths = 0.15 if show_violin else 0.5
            median_color = 'black' if show_violin else 'red'
            
            bp = ax.boxplot(plot_data, positions=positions, widths=widths, patch_artist=True, labels=group_labels,
                            medianprops=dict(color=median_color, linewidth=2),
                            boxprops=dict(linewidth=1.2, color='black'),
                            whiskerprops=dict(linewidth=1.2, color='black'),
                            capprops=dict(linewidth=1.2, color='black'),
                            showfliers=not show_points)

            # Color the boxes based on group colors
            for i, patch in enumerate(bp['boxes']):
                if show_violin:
                    patch.set_facecolor('white')
                    patch.set_alpha(0.9)
                else:
                    group_name = group_labels[i]
                    color = self.app.group_colors.get(group_name, '#CCCCCC')
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(group_labels)
            
        scatters = []
        # Add individual data points as a swarm plot
        if show_points:
            for i, data_points in enumerate(plot_data):
                if len(data_points) > 0:
                    # Add bounded uniform jitter to the x-axis for a more even, bee-swarm-like spread
                    x_jitter = np.random.uniform(positions[i] - 0.12, positions[i] + 0.12, size=len(data_points))
                    sc = ax.scatter(x_jitter, data_points, alpha=0.7, color='#333333', edgecolors='white', linewidths=0.6, s=30, zorder=3)
                    
                    if sample_names and i < len(sample_names):
                        scatters.append((sc, sample_names[i], group_labels[i]))

        ax.set_xlim(0.5, len(plot_data) + 0.5)
        ax.set_ylabel(ylabel, fontsize=fonts['label'])
        ax.set_xlabel("Sample Groups", fontsize=fonts['label'])
        
        title_text = f"Peak Intensity Distribution for Feature:\n{feature_id}"
        
        # --- Statistical Annotations ---
        if len(plot_data) >= 2:
            y_max = max([np.max(d) if len(d) > 0 else 0 for d in plot_data])
            y_min = min([np.min(d) if len(d) > 0 else 0 for d in plot_data])
            y_range = y_max - y_min if y_max > y_min else 1.0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if len(plot_data) == 2:
                    # Welch's T-test for exactly 2 groups
                    try:
                        stat_val, pval = stats.ttest_ind(plot_data[0], plot_data[1], equal_var=False)
                        if not np.isnan(pval):
                            y_bracket = y_max + 0.05 * y_range
                            h_bracket = 0.02 * y_range
                            ax.plot([1, 1, 2, 2], [y_bracket, y_bracket+h_bracket, y_bracket+h_bracket, y_bracket], lw=1.2, c='k')
                            pval_text = f"p = {pval:.3f}" if pval > 0.001 else f"p = {pval:.2e}"
                            if pval < 0.001:
                                pval_text += "***"
                            elif pval < 0.01:
                                pval_text += "**"
                            elif pval < 0.05:
                                pval_text += "*"
                            ax.text(1.5, y_bracket + h_bracket * 1.5, pval_text, ha='center', va='bottom', fontsize=fonts['tick'], fontweight='bold')
                            ax.set_ylim(top=y_bracket + h_bracket * 5)
                            title_text += f"\nWelch's t-test {pval_text}"
                    except Exception:
                        pass
                else:
                    # ANOVA for > 2 groups
                    try:
                        valid_data = [d for d in plot_data if len(d) > 1]
                        if len(valid_data) >= 2:
                            f_stat, pval = stats.f_oneway(*valid_data)
                            if not np.isnan(pval):
                                pval_text = f"p = {pval:.3f}" if pval > 0.001 else f"p = {pval:.2e}"
                                title_text += f"\nOne-way ANOVA {pval_text}"
                                
                            # Pairwise significance brackets if <= 5 groups (avoiding clutter)
                            if 2 < len(plot_data) <= 5:
                                combos = list(itertools.combinations(range(len(plot_data)), 2))
                                y_bracket = y_max + 0.05 * y_range
                                h_bracket = 0.02 * y_range
                                
                                bracket_drawn = False
                                for (i, j) in combos:
                                    if len(plot_data[i]) < 2 or len(plot_data[j]) < 2: continue
                                    st, pv = stats.ttest_ind(plot_data[i], plot_data[j], equal_var=False)
                                    if not np.isnan(pv) and pv < 0.05:
                                        ax.plot([i+1, i+1, j+1, j+1], [y_bracket, y_bracket+h_bracket, y_bracket+h_bracket, y_bracket], lw=1.0, c='k', alpha=0.7)
                                        sig_text = "***" if pv < 0.001 else ("**" if pv < 0.01 else "*")
                                        ax.text((i+1 + j+1)/2.0, y_bracket + h_bracket * 1.5, sig_text, ha='center', va='bottom', fontsize=fonts['annot'], fontweight='bold', color='k')
                                        y_bracket += h_bracket * 6.0
                                        bracket_drawn = True
                                if bracket_drawn:
                                    ax.set_ylim(top=y_bracket + h_bracket * 6)
                    except Exception:
                        pass

        ax.set_title(title_text, fontsize=fonts['title'], fontweight='bold')
        
        ax.tick_params(axis='x', rotation=30, labelsize=fonts['tick'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        fig._export_autoscale_text = list(ax.get_xticklabels())

        # --- Interactive Tooltip Logic ---
        annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                            arrowprops=dict(arrowstyle="->"), zorder=10)
        annot.set_visible(False)

        def on_hover(event):
            if event.inaxes == ax:
                is_contained = False
                for sc, names, grp in scatters:
                    cont, ind = sc.contains(event)
                    if cont:
                        idx = ind["ind"][0]
                        pos = sc.get_offsets()[idx]
                        annot.xy = pos
                        s_name = names[idx]
                        val = pos[1]
                        
                        # Format scientifically if large, normally if log/small
                        val_str = f"{val:.2e}" if (val > 10000 and "Log10" not in ylabel) else f"{val:.4f}"
                        annot.set_text(f"Sample: {s_name}\nGroup: {grp}\nInt: {val_str}")
                        annot.set_visible(True)
                        event.canvas.draw_idle()
                        is_contained = True
                        break
                if not is_contained and annot.get_visible():
                    annot.set_visible(False)
                    event.canvas.draw_idle()

        fig.tight_layout()
        self._embed_plot(fig, self.app.feature_viewer_plot_frame, hover_handler=on_hover)
