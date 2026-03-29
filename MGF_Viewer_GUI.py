import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
from typing import Any, Dict, Generator, List
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def parse_mgf(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Parses an MGF file and yields one spectrum at a time.
    Handles standard MGF format.
    """
    try:
        with open(file_path, 'r') as f:
            spectrum = {}
            in_spectrum = False
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == 'BEGIN IONS':
                    spectrum = {'mzs': [], 'intensities': []}
                    in_spectrum = True
                    continue
                
                if line == 'END IONS':
                    if in_spectrum:
                        yield spectrum
                    in_spectrum = False
                    continue

                if in_spectrum:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        spectrum[key] = value
                    else:
                        try:
                            parts = line.split()
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            spectrum['mzs'].append(mz)
                            spectrum['intensities'].append(intensity)
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse peak line: {line}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

class MGFViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MGF Spectrum Viewer")
        self.root.geometry("1000x800")

        self.spectra: List[Dict[str, Any]] = []
        self.loaded_files = set()
        self.hover_annotation = None
        self.plotted_mzs = np.array([])
        self.plotted_intensities = np.array([])
        self.draggable_labels = []
        self.dragged_artist = None
        self.drag_press_info = None

        # --- Main Layout ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Frame for file selection ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        self.file_label = ttk.Label(top_frame, text="No file selected.")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        browse_button = ttk.Button(top_frame, text="Browse...", command=self.load_file)
        browse_button.pack(side=tk.RIGHT)

        # --- Content Frame for list and plot ---
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        content_frame.grid_columnconfigure(1, weight=3) # Give more weight to plot column
        content_frame.grid_rowconfigure(0, weight=1)

        # --- Left panel for spectra list and info ---
        # Create a container for the canvas and scrollbar
        left_container = ttk.Frame(content_frame)
        left_container.grid(row=0, column=0, sticky="ns")

        # Create a canvas and a scrollbar
        canvas = tk.Canvas(left_container, borderwidth=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        
        # This frame will contain all the widgets and be placed inside the canvas
        self.scrollable_frame = ttk.Frame(canvas, padding=5)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar into the container
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Place all left-side widgets into the scrollable_frame ---
        ttk.Label(self.scrollable_frame, text="Spectra:", font="-weight bold").pack(anchor="w")
        
        # --- Search Panel ---
        search_panel = ttk.LabelFrame(self.scrollable_frame, text="Search Spectra", padding=5)
        search_panel.pack(fill=tk.X, pady=(2, 5))
        
        ttk.Label(search_panel, text="Search by:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.search_type_var = tk.StringVar(value="Feature ID / Scan")
        search_type_combo = ttk.Combobox(search_panel, textvariable=self.search_type_var,
                                         values=["Feature ID / Scan", "m/z", "PEPMASS", "RT (min)"], width=15, state='readonly')
        search_type_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=2, pady=2)

        ttk.Label(search_panel, text="Value(s):").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_panel, textvariable=self.search_var)
        search_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=2, pady=2)

        ttk.Label(search_panel, text="Tolerance:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        self.mz_tolerance_var = tk.StringVar(value="0.01")
        mz_tolerance_entry = ttk.Entry(search_panel, textvariable=self.mz_tolerance_var, width=10)
        mz_tolerance_entry.grid(row=2, column=1, sticky="w", padx=2, pady=2)

        search_button = ttk.Button(search_panel, text="Find", command=self.perform_search)
        search_button.grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        reset_button = ttk.Button(search_panel, text="Reset", command=self.reset_search)
        reset_button.grid(row=3, column=2, sticky="ew", padx=2, pady=2)

        search_panel.grid_columnconfigure(1, weight=1)

        self.spectra_listbox = tk.Listbox(self.scrollable_frame, height=10, selectmode=tk.EXTENDED) # Reduced height, allow multi-select
        self.spectra_listbox.pack(fill=tk.X, expand=False)
        self.spectra_listbox.bind('<<ListboxSelect>>', self.on_spectrum_select)

        # --- Plotting Buttons ---
        plot_button_frame = ttk.Frame(self.scrollable_frame)
        plot_button_frame.pack(fill=tk.X, pady=(10,5))
        self.plot_single_button = ttk.Button(plot_button_frame, text="Plot Selected Spectrum", command=self.plot_selected_spectrum, state=tk.DISABLED)
        self.plot_single_button.pack(fill=tk.X, pady=2)
        self.plot_mirror_button = ttk.Button(plot_button_frame, text="Plot Mirror Spectrum (2 selected)", command=self.plot_mirror_spectrum, state=tk.DISABLED)
        self.plot_mirror_button.pack(fill=tk.X, pady=2)

        # --- Plotting options ---
        options_frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Options", padding=5)
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Intensity Threshold:").grid(row=0, column=0, sticky="w", pady=2)
        self.intensity_threshold_var = tk.StringVar(value="0")
        threshold_entry = ttk.Entry(options_frame, textvariable=self.intensity_threshold_var, width=15)
        threshold_entry.grid(row=0, column=1, sticky="w", pady=2)

        self.show_labels_var = tk.BooleanVar(value=False)
        show_labels_check = ttk.Checkbutton(options_frame, text="Show m/z Labels", variable=self.show_labels_var)
        show_labels_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(options_frame, text="Label Font Size:").grid(row=2, column=0, sticky="w", pady=2)
        self.font_size_var = tk.StringVar(value="8")
        font_size_spinbox = ttk.Spinbox(options_frame, from_=6, to=16, textvariable=self.font_size_var, width=13)
        font_size_spinbox.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(options_frame, text="Peak Color:").grid(row=3, column=0, sticky="w", pady=2)
        self.peak_color_var = tk.StringVar(value="grey")
        peak_color_combo = ttk.Combobox(options_frame, textvariable=self.peak_color_var, 
                                        values=['grey', 'black', 'blue', 'red', 'green'], width=12, state='readonly')
        peak_color_combo.grid(row=3, column=1, sticky="w", pady=2)
        
        ttk.Label(options_frame, text="Mirror Peak Color:").grid(row=4, column=0, sticky="w", pady=2)
        self.mirror_peak_color_var = tk.StringVar(value="red")
        mirror_peak_color_combo = ttk.Combobox(options_frame, textvariable=self.mirror_peak_color_var,
                                               values=['red', 'blue', 'green', 'black', 'orange', 'purple'], width=12, state='readonly')
        mirror_peak_color_combo.grid(row=4, column=1, sticky="w", pady=2)

        self.normalize_var = tk.BooleanVar(value=False)
        normalize_check = ttk.Checkbutton(options_frame, text="Normalize Intensity", variable=self.normalize_var)
        normalize_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(options_frame, text="Min m/z:").grid(row=6, column=0, sticky="w", pady=2)
        self.min_mz_var = tk.StringVar()
        min_mz_entry = ttk.Entry(options_frame, textvariable=self.min_mz_var, width=15)
        min_mz_entry.grid(row=6, column=1, sticky="w", pady=2)

        ttk.Label(options_frame, text="Max m/z:").grid(row=7, column=0, sticky="w", pady=2)
        self.max_mz_var = tk.StringVar()
        max_mz_entry = ttk.Entry(options_frame, textvariable=self.max_mz_var, width=15)
        max_mz_entry.grid(row=7, column=1, sticky="w", pady=2)

        ttk.Label(options_frame, text="Min Intensity:").grid(row=8, column=0, sticky="w", pady=2)
        self.min_intensity_var = tk.StringVar(value="0")
        min_intensity_entry = ttk.Entry(options_frame, textvariable=self.min_intensity_var, width=15)
        min_intensity_entry.grid(row=8, column=1, sticky="w", pady=2)

        ttk.Label(options_frame, text="Max Intensity:").grid(row=9, column=0, sticky="w", pady=2)
        self.max_intensity_var = tk.StringVar()
        max_intensity_entry = ttk.Entry(options_frame, textvariable=self.max_intensity_var, width=15)
        max_intensity_entry.grid(row=9, column=1, sticky="w", pady=2)

        # --- Title options ---
        title_frame = ttk.LabelFrame(self.scrollable_frame, text="Title Options", padding=5)
        title_frame.pack(fill=tk.X, pady=5)

        ttk.Label(title_frame, text="Custom Title:").grid(row=0, column=0, columnspan=2, sticky="w")
        self.title_text = tk.Text(title_frame, height=3, width=30)
        self.title_text.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(title_frame, text="Font Size:").grid(row=2, column=0, sticky="w", pady=2)
        self.title_size_var = tk.StringVar(value="12")
        title_size_spinbox = ttk.Spinbox(title_frame, from_=8, to=24, textvariable=self.title_size_var, width=10)
        title_size_spinbox.grid(row=2, column=1, sticky="w", pady=2)

        self.title_bold_var = tk.BooleanVar(value=False)
        title_bold_check = ttk.Checkbutton(title_frame, text="Bold", variable=self.title_bold_var)
        title_bold_check.grid(row=3, column=0, sticky="w")

        self.title_italic_var = tk.BooleanVar(value=False)
        title_italic_check = ttk.Checkbutton(title_frame, text="Italic", variable=self.title_italic_var)
        title_italic_check.grid(row=3, column=1, sticky="w")

        # --- Axis options ---
        axis_frame = ttk.LabelFrame(self.scrollable_frame, text="Axis Options", padding=5)
        axis_frame.pack(fill=tk.X, pady=5)

        ttk.Label(axis_frame, text="X-Label:").grid(row=0, column=0, sticky="w", pady=2)
        self.xlabel_var = tk.StringVar()
        xlabel_entry = ttk.Entry(axis_frame, textvariable=self.xlabel_var, width=15)
        xlabel_entry.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(axis_frame, text="Y-Label:").grid(row=1, column=0, sticky="w", pady=2)
        self.ylabel_var = tk.StringVar()
        ylabel_entry = ttk.Entry(axis_frame, textvariable=self.ylabel_var, width=15)
        ylabel_entry.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(axis_frame, text="Label Size:").grid(row=2, column=0, sticky="w", pady=2)
        self.axis_label_size_var = tk.StringVar(value="10")
        axis_label_size_spinbox = ttk.Spinbox(axis_frame, from_=8, to=20, textvariable=self.axis_label_size_var, width=13)
        axis_label_size_spinbox.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(axis_frame, text="Tick Size:").grid(row=3, column=0, sticky="w", pady=2)
        self.tick_label_size_var = tk.StringVar(value="8")
        tick_label_size_spinbox = ttk.Spinbox(axis_frame, from_=6, to=16, textvariable=self.tick_label_size_var, width=13)
        tick_label_size_spinbox.grid(row=3, column=1, sticky="w", pady=2)

        # --- Legend options ---
        legend_frame = ttk.LabelFrame(self.scrollable_frame, text="Legend Options", padding=5)
        legend_frame.pack(fill=tk.X, pady=5)

        self.show_legend_var = tk.BooleanVar(value=True)
        show_legend_check = ttk.Checkbutton(legend_frame, text="Show Legend", variable=self.show_legend_var)
        show_legend_check.grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(legend_frame, text="Legend 1 Text:").grid(row=1, column=0, sticky="w", pady=2)
        self.legend1_text_var = tk.StringVar()
        legend1_entry = ttk.Entry(legend_frame, textvariable=self.legend1_text_var, width=15)
        legend1_entry.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(legend_frame, text="Legend 2 Text:").grid(row=2, column=0, sticky="w", pady=2)
        self.legend2_text_var = tk.StringVar()
        legend2_entry = ttk.Entry(legend_frame, textvariable=self.legend2_text_var, width=15)
        legend2_entry.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(legend_frame, text="Font Size:").grid(row=3, column=0, sticky="w", pady=2)
        self.legend_size_var = tk.StringVar(value="10")
        legend_size_spinbox = ttk.Spinbox(legend_frame, from_=6, to=20, textvariable=self.legend_size_var, width=13)
        legend_size_spinbox.grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(legend_frame, text="Location:").grid(row=4, column=0, sticky="w", pady=2)
        self.legend_loc_var = tk.StringVar(value="best")
        legend_loc_combo = ttk.Combobox(legend_frame, textvariable=self.legend_loc_var,
                                        values=['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'],
                                        width=12, state='readonly')
        legend_loc_combo.grid(row=4, column=1, sticky="w", pady=2)

        self.info_label = ttk.Label(self.scrollable_frame, text="Spectrum Info:", font="-weight bold")
        self.info_label.pack(anchor="w", pady=(10, 0))
        self.info_text = tk.Text(self.scrollable_frame, height=10, width=35, state=tk.DISABLED, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, expand=False)

        # --- Right panel for plot ---
        plot_frame = ttk.Frame(content_frame, padding=5)
        plot_frame.grid(row=0, column=1, sticky="nsew")

        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        
        # Create and pack the Matplotlib navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Connect all event handlers
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)

        # --- Status Bar ---
        self.status_bar = ttk.Label(main_frame, text="Ready. Select a file to begin.", anchor=tk.W, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)


    def load_file(self):
        file_paths = filedialog.askopenfilenames(
            title="Select an MGF file",
            filetypes=(("MGF files", "*.mgf"), ("All files", "*.*"))
        )
        if not file_paths:
            return

        # Ask user whether to append or clear existing list
        if self.spectra and messagebox.askquestion("Load Mode", "Add files to the current list? (No will clear the list)") == 'no':
            self.spectra.clear()
            self.spectra_listbox.delete(0, tk.END)
            self.loaded_files.clear()

        new_spectra_count = 0
        for file_path in file_paths:
            if file_path in self.loaded_files:
                continue # Skip already loaded files

            self.loaded_files.add(file_path)
            file_name = os.path.basename(file_path)

            for spec in parse_mgf(file_path):
                # Add source file information to each spectrum
                spec['SOURCE_FILE'] = file_name
                self.spectra.append(spec)
                new_spectra_count += 1

        # Update the listbox with all spectra
        self.spectra_listbox.delete(0, tk.END)
        for spec in self.spectra:
            # Use SCANS or TITLE as identifier, fallback to index
            title = spec.get('SCANS', spec.get('TITLE', 'N/A'))
            file_name = spec.get('SOURCE_FILE', 'unknown')
            self.spectra_listbox.insert(tk.END, f"{file_name} - Scan: {title}")

        self.file_label.config(text=f"{len(self.loaded_files)} file(s) loaded.")
        self.status_bar.config(text=f"Added {new_spectra_count} new spectra. Total: {len(self.spectra)}. Select a spectrum to plot.")

    def on_spectrum_select(self, event=None):
        selection_count = len(self.spectra_listbox.curselection())
        
        # Update button states based on selection
        if selection_count == 1:
            self.plot_single_button.config(state=tk.NORMAL)
            self.plot_mirror_button.config(state=tk.DISABLED)
        elif selection_count == 2:
            self.plot_single_button.config(state=tk.DISABLED)
            self.plot_mirror_button.config(state=tk.NORMAL)
        else:
            self.plot_single_button.config(state=tk.DISABLED)
            self.plot_mirror_button.config(state=tk.DISABLED)
            
        self._update_spectrum_info()

    def perform_search(self):
        """Filters the spectra list based on the selected search criteria."""
        search_type = self.search_type_var.get()
        search_value = self.search_var.get().strip()

        if not search_value:
            self.status_bar.config(text="Please enter a search value.")
            return

        found_indices = []
        if search_type == "Feature ID / Scan":
            search_ids = set(search_value.split())
            for i, spec in enumerate(self.spectra):
                # Check FEATURE_ID, SCANS, and TITLE for a match
                spec_id = str(spec.get('FEATURE_ID', spec.get('SCANS', spec.get('TITLE', '')))).strip()
                if spec_id in search_ids:
                    found_indices.append(i)
        
        elif search_type == "m/z":
            search_terms = search_value.split()
            if not search_terms:
                self.status_bar.config(text="Please enter m/z value(s) or range(s).")
                return

            for i, spec in enumerate(self.spectra):
                mzs = np.array(spec.get('mzs', []))
                if mzs.size == 0:
                    continue

                # Check if the spectrum contains a peak matching any of the search terms
                for term in search_terms:
                    try:
                        if '-' in term: # Range search, e.g., "100-200"
                            min_mz, max_mz = map(float, term.split('-'))
                            if np.any((mzs >= min_mz) & (mzs <= max_mz)):
                                found_indices.append(i)
                                break # Move to the next spectrum
                        else: # Single m/z value with tolerance
                            target_mz = float(term)
                            tolerance = float(self.mz_tolerance_var.get())
                            if np.any(np.abs(mzs - target_mz) <= tolerance):
                                found_indices.append(i)
                                break # Move to the next spectrum
                    except (ValueError, IndexError):
                        messagebox.showerror("Invalid Input", f"Could not parse m/z value or range: '{term}'.\nUse numbers (e.g., 150.5) or ranges (e.g., 100-200).")
                        return
        
        elif search_type == "PEPMASS":
            try:
                target_pepmass = float(search_value)
                tolerance = float(self.mz_tolerance_var.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a single numeric value for PEPMASS.")
                return

            for i, spec in enumerate(self.spectra):
                pepmass_str = spec.get('PEPMASS', '').split()[0] # Handle cases like "PEPMASS=500.2 1000"
                if pepmass_str:
                    try:
                        if abs(float(pepmass_str) - target_pepmass) <= tolerance:
                            found_indices.append(i)
                    except ValueError:
                        continue # Ignore if PEPMASS is not a valid float

        elif search_type == "RT (min)":
            try:
                if '-' in search_value: # Range search, e.g., "30-60"
                    min_rt_min, max_rt_min = map(float, search_value.split('-'))
                else: # Single value with tolerance
                    target_rt_min = float(search_value)
                    # Using the tolerance value in minutes
                    tolerance_min = float(self.mz_tolerance_var.get())
                    min_rt_min, max_rt_min = target_rt_min - tolerance_min, target_rt_min + tolerance_min
                
                # Convert user input from minutes to seconds for comparison
                min_rt_sec, max_rt_sec = min_rt_min * 60, max_rt_min * 60
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a numeric value or range (e.g., 1.5-2.5) for RT (min).")
                return

            for i, spec in enumerate(self.spectra):
                rt_str = spec.get('RTINSECONDS')
                if rt_str:
                    try:
                        rt_val_sec = float(rt_str)
                        if min_rt_sec <= rt_val_sec <= max_rt_sec:
                            found_indices.append(i)
                    except ValueError:
                        continue # Ignore if RTINSECONDS is not a valid float

        self._update_listbox(found_indices)

    def reset_search(self):
        """Resets the search and shows all spectra."""
        self.search_var.set("")
        all_indices = list(range(len(self.spectra)))
        self._update_listbox(all_indices)
        self.status_bar.config(text=f"Search reset. Displaying {len(self.spectra)} spectra.")

    def _update_listbox(self, indices_to_show: List[int]):
        """
        Updates the spectra listbox to show only the spectra at the given indices.
        """
        # Store a map from the listbox index to the original spectrum index
        self.listbox_map = {new_idx: old_idx for new_idx, old_idx in enumerate(indices_to_show)}

        self.spectra_listbox.delete(0, tk.END)
        if not indices_to_show:
            self.status_bar.config(text="No spectra found matching the criteria.")
            return

        for index in indices_to_show:
            spec = self.spectra[index]
            title = spec.get('SCANS', spec.get('FEATURE_ID', spec.get('TITLE', 'N/A')))
            file_name = spec.get('SOURCE_FILE', 'unknown')
            self.spectra_listbox.insert(tk.END, f"{file_name} - Scan: {title}")

        # Select the first item in the filtered list
        self.spectra_listbox.selection_set(0)
        self.on_spectrum_select()

        self.status_bar.config(text=f"Found {len(indices_to_show)} matching spectra.")
        
    def _update_spectrum_info(self):
        selection_indices = self.spectra_listbox.curselection() # This is a tuple of indices
        if not selection_indices:
            # Clear info text if nothing is selected
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.config(state=tk.DISABLED)
            return

        info_str = ""
        # Display info for all selected spectra
        for i, list_idx in enumerate(selection_indices):
            try:
                original_index = self.listbox_map[list_idx]
            except (AttributeError, KeyError): # Fallback for when list is not filtered
                original_index = list_idx
        
            spectrum = self.spectra[original_index]
            info_str += f"--- Spectrum {i+1} ---\n"
            for key in ['SOURCE_FILE', 'PEPMASS', 'CHARGE', 'RTINSECONDS', 'SCANS', 'TITLE']:
                if key in spectrum:
                    info_str += f"{key}: {spectrum[key]}\n"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_str)
        self.info_text.config(state=tk.DISABLED)

    def on_press(self, event):
        """Handle mouse button press for dragging labels."""
        # Check if the click is on any of the draggable labels
        for label in self.draggable_labels:
            contains, _ = label.contains(event)
            if contains:
                # Left-click to start dragging
                if event.button == 1:
                    self.dragged_artist = label
                    # Store initial position and mouse coordinates
                    x0, y0 = label.get_position()
                    self.drag_press_info = (x0, y0, event.xdata, event.ydata)
                    return # Stop after finding the first label
                # Right-click to delete the label
                elif event.button == 3:
                    label.remove()
                    self.draggable_labels.remove(label)
                    self.canvas.draw_idle()
                    self.status_bar.config(text="Label deleted.")
                    return # Stop after finding and deleting the label

    def on_release(self, event):
        """Handle mouse button release to stop dragging."""
        # Reset dragging state
        self.dragged_artist = None
        self.drag_press_info = None

    def on_drag(self, event):
        """Handle mouse motion for dragging."""
        # If we are not dragging an artist, do nothing
        if self.dragged_artist is None or event.inaxes is None:
            return

        # Unpack stored press information
        x0, y0, xpress, ypress = self.drag_press_info

        # Calculate the change in mouse position
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        # Update the artist's position
        self.dragged_artist.set_position((x0 + dx, y0 + dy))

        # Redraw the canvas to show the new position
        self.canvas.draw_idle()

    def on_hover(self, event):
        """Handle mouse hover events on the plot canvas."""
        # Ensure the annotation exists and the event is within the plot area
        if not self.hover_annotation or event.inaxes != self.hover_annotation.axes:
            return

        # Check if there are any peaks to hover over
        if self.dragged_artist: # Don't show hover info while dragging a label
            return

        # Check if there are any peaks to hover over
        if len(self.plotted_mzs) == 0:
            return

        # Find the index of the peak closest to the mouse's x-coordinate
        dx = np.abs(self.plotted_mzs - event.xdata)
        idx = np.argmin(dx)
        
        # Define a tolerance for how close the mouse must be to a peak
        # Here, tolerance is 0.5% of the total m/z range
        mz_range = self.plotted_mzs.max() - self.plotted_mzs.min()
        x_tolerance = mz_range * 0.005 if mz_range > 0 else 0.1

        # Use the plotted intensities for the check
        intensities = self.plotted_intensities
        
        # Check if the closest peak is within the tolerance and mouse is below the peak's intensity
        if dx[idx] < x_tolerance and event.ydata > 0 and event.ydata < intensities[idx]:
            mz, intensity = self.plotted_mzs[idx], intensities[idx]
            
            # Update annotation text and position
            self.hover_annotation.set_text(f"m/z: {mz:.4f}")
            self.hover_annotation.set_position((mz, intensity))
            self.hover_annotation.set_visible(True)
            
            # Update status bar
            self.status_bar.config(text=f"Hovering over: m/z={mz:.4f}, Intensity={intensity:.2f}")
        else:
            # If not close to any peak, hide the annotation and clear status bar
            if self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.status_bar.config(text="Plot updated. Hover over a peak to see details.")
        
        # Redraw the canvas to show/hide the annotation
        self.canvas.draw_idle()

    def plot_selected_spectrum(self):
        selection_indices = self.spectra_listbox.curselection()
        if not selection_indices:
            messagebox.showwarning("Warning", "Please select a spectrum to plot.")
            return

        selected_index = selection_indices[0]
        
        # Use the map to get the correct index from the original self.spectra list
        try:
            original_index = self.listbox_map[selected_index]
            spectrum = self.spectra[original_index]
        except (AttributeError, KeyError): # Fallback for when list is not filtered
            spectrum = self.spectra[selected_index]


        # Get threshold from GUI, with error handling
        try:
            threshold = float(self.intensity_threshold_var.get())
        except (ValueError, TypeError):
            threshold = 0.0
            self.intensity_threshold_var.set("0.0")

        # Get other plot options
        show_labels = self.show_labels_var.get()
        try:
            font_size = int(self.font_size_var.get())
        except (ValueError, TypeError):
            font_size = 8
        peak_color = self.peak_color_var.get()
        normalize = self.normalize_var.get()

        # Get title options
        custom_title = self.title_text.get("1.0", tk.END).strip()
        try:
            title_size = int(self.title_size_var.get())
        except (ValueError, TypeError):
            title_size = 12
        title_weight = 'bold' if self.title_bold_var.get() else 'normal'
        title_style = 'italic' if self.title_italic_var.get() else 'normal'

        # Get axis options
        custom_xlabel = self.xlabel_var.get()
        custom_ylabel = self.ylabel_var.get()
        try:
            axis_label_size = int(self.axis_label_size_var.get())
        except (ValueError, TypeError):
            axis_label_size = 10
        try:
            tick_label_size = int(self.tick_label_size_var.get())
        except (ValueError, TypeError):
            tick_label_size = 8

        # Get mass range from GUI, with error handling
        try:
            min_mz = float(self.min_mz_var.get()) if self.min_mz_var.get() else None
        except ValueError:
            min_mz = None
            self.min_mz_var.set("") # Clear invalid entry
        try:
            max_mz = float(self.max_mz_var.get()) if self.max_mz_var.get() else None
        except ValueError:
            max_mz = None
            self.max_mz_var.set("") # Clear invalid entry
        
        # Get Y-axis range from GUI
        try:
            min_intensity = float(self.min_intensity_var.get()) if self.min_intensity_var.get() else 0.0
        except ValueError:
            min_intensity = 0.0
            self.min_intensity_var.set("0.0")
        try:
            max_intensity_user = float(self.max_intensity_var.get()) if self.max_intensity_var.get() else None
        except ValueError:
            max_intensity_user = None
            self.max_intensity_var.set("")

        self.fig.clear()
        # Clear the list of draggable labels for the new plot
        self.draggable_labels.clear()

        ax = self.fig.add_subplot(111)

        # Create a persistent but initially invisible annotation
        self.hover_annotation = ax.annotate("", xy=(0,0), xytext=(0,5),
                                            textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                                            visible=False)
        
        sns.set_theme(style="ticks")

        # Filter peaks based on the intensity threshold
        mzs = np.array(spectrum.get('mzs', []))
        intensities = np.array(spectrum.get('intensities', []))

        # Apply threshold filter on original intensities
        mask = np.ones_like(mzs, dtype=bool) # Start with all peaks included
        if threshold > 0:
            mask &= (intensities >= threshold)
        
        # Apply mass range filter
        if min_mz is not None:
            mask &= (mzs >= min_mz)
        if max_mz is not None:
            mask &= (mzs <= max_mz)

        filtered_mzs = mzs[mask]
        filtered_intensities = intensities[mask]

        # Handle normalization
        y_label = "Intensity"
        if normalize and filtered_intensities.size > 0:
            max_intensity = filtered_intensities.max()
            self.plotted_intensities = (filtered_intensities / max_intensity) * 100 if max_intensity > 0 else filtered_intensities
            y_label = "Relative Intensity (%)"
        else:
            self.plotted_intensities = filtered_intensities
        
        self.plotted_mzs = filtered_mzs

        if self.plotted_mzs.size > 0:
            ax.vlines(self.plotted_mzs, [0], self.plotted_intensities, color=peak_color, linewidth=1)
            # Add m/z labels if the checkbox is ticked
            if show_labels:
                for mz, intensity in zip(self.plotted_mzs, self.plotted_intensities):
                    label = ax.text(mz, intensity, f"{mz:.4f}", ha='center', va='bottom', fontsize=font_size, rotation=45)
                    self.draggable_labels.append(label)

        else:
            ax.text(0.5, 0.5, 'No peak data in this spectrum', ha='center', va='center', transform=ax.transAxes)
        
        pepmass = spectrum.get('PEPMASS', 'N/A')
        charge = spectrum.get('CHARGE', 'N/A')
        scans = spectrum.get('SCANS', 'N/A')
        
        if custom_title:
            fontdict = {
                'fontsize': title_size,
                'fontweight': title_weight,
                'fontstyle': title_style
            }
            ax.set_title(custom_title, fontdict=fontdict)
        else:
            ax.set_title(f"Mass Spectrum (Scan: {scans})\nPEPMASS: {pepmass}, CHARGE: {charge}")

        # Set axis labels and font sizes
        ax.set_xlabel(custom_xlabel or "m/z", fontsize=axis_label_size)
        ax.set_ylabel(custom_ylabel or y_label, fontsize=axis_label_size)

        # Set tick label sizes
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)


        # Set axis limits to remove gaps
        final_min_mz = min_mz if min_mz is not None else (self.plotted_mzs.min() if self.plotted_mzs.size > 0 else 0)
        final_max_mz = max_mz if max_mz is not None else (self.plotted_mzs.max() if self.plotted_mzs.size > 0 else 1)
        ax.set_xlim(final_min_mz, final_max_mz)

        # Set Y-axis limits. If user provides a max, use it. Otherwise, add 5% padding to the top.
        final_max_intensity = max_intensity_user
        if final_max_intensity is None and self.plotted_intensities.size > 0:
            final_max_intensity = self.plotted_intensities.max() * 1.05
        ax.set_ylim(bottom=min_intensity, top=final_max_intensity)
        
        sns.despine(ax=ax)
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_bar.config(text="Plot updated. Hover over a peak to see details.")

    def plot_mirror_spectrum(self):
        selection_indices = self.spectra_listbox.curselection()
        if len(selection_indices) != 2:
            messagebox.showwarning("Warning", "Please select exactly two spectra for a mirror plot.")
            return

        # --- Get Plotting Options (same as single plot) ---
        try:
            threshold = float(self.intensity_threshold_var.get())
        except (ValueError, TypeError):
            threshold = 0.0
        normalize = self.normalize_var.get()
        peak_color = self.peak_color_var.get()
        mirror_peak_color = self.mirror_peak_color_var.get()
        try:
            min_mz = float(self.min_mz_var.get()) if self.min_mz_var.get() else None
        except ValueError:
            min_mz = None
        try:
            max_mz = float(self.max_mz_var.get()) if self.max_mz_var.get() else None
        except ValueError:
            max_mz = None

        # Get label options
        show_labels = self.show_labels_var.get()
        try:
            font_size = int(self.font_size_var.get())
        except (ValueError, TypeError):
            font_size = 8


        # Get title options
        custom_title = self.title_text.get("1.0", tk.END).strip()
        try:
            title_size = int(self.title_size_var.get())
        except (ValueError, TypeError):
            title_size = 12
        title_weight = 'bold' if self.title_bold_var.get() else 'normal'
        title_style = 'italic' if self.title_italic_var.get() else 'normal'

        # Get axis options
        custom_xlabel = self.xlabel_var.get()
        custom_ylabel = self.ylabel_var.get()
        try:
            axis_label_size = int(self.axis_label_size_var.get())
        except (ValueError, TypeError):
            axis_label_size = 10
        try:
            tick_label_size = int(self.tick_label_size_var.get())
        except (ValueError, TypeError):
            tick_label_size = 8

        # Get Y-axis range from GUI
        try:
            min_intensity = float(self.min_intensity_var.get()) if self.min_intensity_var.get() else 0.0
        except ValueError:
            min_intensity = 0.0
        try:
            max_intensity_user = float(self.max_intensity_var.get()) if self.max_intensity_var.get() else None
        except ValueError:
            max_intensity_user = None

        # Get legend options
        show_legend = self.show_legend_var.get()
        legend1_text = self.legend1_text_var.get().strip()
        legend2_text = self.legend2_text_var.get().strip()
        try:
            legend_size = int(self.legend_size_var.get())
        except (ValueError, TypeError):
            legend_size = 10
        legend_loc = self.legend_loc_var.get()



        # --- Get Spectra ---
        spectra_to_plot = []
        for list_idx in selection_indices:
            try:
                original_index = self.listbox_map[list_idx]
                spectra_to_plot.append(self.spectra[original_index])
            except (AttributeError, KeyError):
                spectra_to_plot.append(self.spectra[list_idx])

        spec1, spec2 = spectra_to_plot

        # --- Clear and Setup Plot ---
        self.fig.clear()
        self.draggable_labels.clear()
        ax = self.fig.add_subplot(111)
        sns.set_theme(style="ticks")

        # --- Process and Plot Spectrum 1 (Top) ---
        mzs1 = np.array(spec1.get('mzs', []))
        intensities1 = np.array(spec1.get('intensities', []))

        mask1 = np.ones_like(mzs1, dtype=bool)
        if threshold > 0: mask1 &= (intensities1 >= threshold)
        if min_mz is not None: mask1 &= (mzs1 >= min_mz)
        if max_mz is not None: mask1 &= (mzs1 <= max_mz)

        plot_mzs1 = mzs1[mask1]
        plot_intensities1 = intensities1[mask1]

        if normalize and plot_intensities1.size > 0:
            max_int1 = plot_intensities1.max()
            plot_intensities1 = (plot_intensities1 / max_int1) * 100 if max_int1 > 0 else plot_intensities1

        label1 = legend1_text or f"Scan: {spec1.get('SCANS', 'N/A')}"

        if plot_mzs1.size > 0:
            ax.vlines(plot_mzs1, [0], plot_intensities1, color=peak_color, linewidth=1, label=label1)
            # Add labels for spectrum 1
            if show_labels:
                for mz, intensity in zip(plot_mzs1, plot_intensities1):
                    label = ax.text(mz, intensity, f"{mz:.4f}", ha='center', va='bottom', fontsize=font_size, rotation=45)
                    self.draggable_labels.append(label)

        # --- Process and Plot Spectrum 2 (Bottom) ---
        mzs2 = np.array(spec2.get('mzs', []))
        intensities2 = np.array(spec2.get('intensities', []))

        mask2 = np.ones_like(mzs2, dtype=bool)
        if threshold > 0: mask2 &= (intensities2 >= threshold)
        if min_mz is not None: mask2 &= (mzs2 >= min_mz)
        if max_mz is not None: mask2 &= (mzs2 <= max_mz)

        plot_mzs2 = mzs2[mask2]
        plot_intensities2 = intensities2[mask2]

        if normalize and plot_intensities2.size > 0:
            max_int2 = plot_intensities2.max()
            plot_intensities2 = (plot_intensities2 / max_int2) * 100 if max_int2 > 0 else plot_intensities2

        label2 = legend2_text or f"Scan: {spec2.get('SCANS', 'N/A')}"

        if plot_mzs2.size > 0:
            # Plot with negative intensities
            ax.vlines(plot_mzs2, [0], -plot_intensities2, color=mirror_peak_color, linewidth=1, label=label2)
            # Add labels for spectrum 2
            if show_labels:
                for mz, intensity in zip(plot_mzs2, plot_intensities2):
                    label = ax.text(mz, -intensity, f"{mz:.4f}", ha='center', va='top', fontsize=font_size, rotation=45)
                    self.draggable_labels.append(label)

        # --- Finalize Plot Appearance ---
        ax.axhline(0, color='black', linewidth=0.5) # Add a zero line

        # Set Title
        if custom_title:
            fontdict = {
                'fontsize': title_size,
                'fontweight': title_weight,
                'fontstyle': title_style
            }
            ax.set_title(custom_title, fontdict=fontdict)
        else:
            scans1 = spec1.get('SCANS', 'N/A')
            scans2 = spec2.get('SCANS', 'N/A')
            ax.set_title(f"Mirror Plot: Scan {scans1} vs Scan {scans2}")

        # Set Labels
        y_label = "Relative Intensity (%)" if normalize else "Intensity"
        ax.set_xlabel(custom_xlabel or "m/z", fontsize=axis_label_size)
        ax.set_ylabel(custom_ylabel or y_label, fontsize=axis_label_size)

        # Set tick label sizes
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)


        # Set Axis Limits
        all_mzs = np.concatenate((plot_mzs1, plot_mzs2))
        if all_mzs.size > 0:
            final_min_mz = min_mz if min_mz is not None else all_mzs.min()
            final_max_mz = max_mz if max_mz is not None else all_mzs.max()
            ax.set_xlim(final_min_mz, final_max_mz)
        else:
            ax.set_xlim(0, 1) # Default range if no data

        # Set Y-axis limits
        final_max_intensity = max_intensity_user
        if final_max_intensity is None:
            all_intensities = np.concatenate((plot_intensities1, plot_intensities2))
            if all_intensities.size > 0:
                final_max_intensity = all_intensities.max() * 1.05
            else:
                final_max_intensity = 100 if normalize else 1.0

        # For mirror plot, min_intensity is applied symmetrically
        # We use the absolute value of min_intensity from the UI, but it's usually 0.
        # The bottom limit will be the negative of the top limit.
        ax.set_ylim(-final_max_intensity, final_max_intensity)
        
        # Format Y-axis to show absolute values
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{abs(x):.0f}"))

        if show_legend:
            ax.legend(loc=legend_loc, fontsize=legend_size)
        sns.despine(ax=ax)
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_bar.config(text="Mirror plot updated.")


if __name__ == '__main__':
    root = tk.Tk()
    app = MGFViewerApp(root)
    root.mainloop()
