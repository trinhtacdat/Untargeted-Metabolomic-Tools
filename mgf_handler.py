# mgf_handler.py
"""
MGF file parsing and spectrum search functionality
Adapted from MGF_Viewer_GUI.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Generator

def parse_mgf(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """
    Parse an MGF file and yield one spectrum at a time.
    Handles standard MGF format from MZmine.
    
    Yields:
        dict: Spectrum dictionary with keys:
            - 'FEATUREID': Feature ID from TITLE
            - 'SCANS': Scan number
            - 'PEPMASS': Precursor m/z
            - 'RTINSECONDS': Retention time in seconds
            - 'CHARGE': Charge state
            - 'mzs': list of m/z values
            - 'intensities': list of intensity values
    """
    try:
        with open(filepath, 'r') as f:
            spectrum = {}
            in_spectrum = False
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line == "BEGIN IONS":
                    spectrum = {'mzs': [], 'intensities': []}
                    in_spectrum = True
                    continue
                
                if line == "END IONS":
                    if in_spectrum:
                        yield spectrum
                    in_spectrum = False
                    continue
                
                if in_spectrum:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        spectrum[key] = value
                    else:
                        # Peak line: m/z intensity
                        try:
                            parts = line.split()
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            spectrum['mzs'].append(mz)
                            spectrum['intensities'].append(intensity)
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse peak line: {line}")
    
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return


class MGFSearchEngine:
    """Search MGF spectra by Feature ID, m/z, or RT"""
    
    def __init__(self):
        self.spectra = []
        self.loaded_files = []
        
    def load_mgf_files(self, filepaths: List[str]):
        """Load one or more MGF files"""
        for filepath in filepaths:
            filename = filepath.split('/')[-1]
            for spec in parse_mgf(filepath):
                spec['SOURCEFILE'] = filename
                self.spectra.append(spec)
        
        self.loaded_files.extend(filepaths)
        return len(self.spectra)
    
    def search_by_feature_id(self, feature_ids: List[str]) -> List[Dict]:
        """
        Search spectra by Feature ID
        
        Args:
            feature_ids: List of feature IDs to search (e.g., ["123_456.78_12.34"])
        
        Returns:
            List of matching spectra
        """
        feature_set = set(str(fid).strip() for fid in feature_ids)
        matches = []
        
        for spec in self.spectra:
            # Try multiple ID fields
            spec_id = str(spec.get('FEATUREID', spec.get('SCANS', spec.get('TITLE', '')))).strip()
            
            # Also check if TITLE contains the feature ID
            title = spec.get('TITLE', '')
            if spec_id in feature_set or any(fid in title for fid in feature_set):
                matches.append(spec)
        
        return matches
    
    def search_by_mz(self, target_mz: float, tolerance: float = 0.01) -> List[Dict]:
        """
        Search spectra containing peaks at target m/z ± tolerance
        
        Args:
            target_mz: Target m/z value
            tolerance: m/z tolerance (default 0.01 Da)
        
        Returns:
            List of matching spectra with matched peaks
        """
        matches = []
        
        for spec in self.spectra:
            mzs = np.array(spec.get('mzs', []))
            if mzs.size == 0:
                continue
            
            # Check if any peak is within tolerance
            if np.any(np.abs(mzs - target_mz) <= tolerance):
                matched_peaks = mzs[np.abs(mzs - target_mz) <= tolerance]
                spec_copy = spec.copy()
                spec_copy['matched_mzs'] = matched_peaks.tolist()
                matches.append(spec_copy)
        
        return matches
    
    def search_by_rt(self, rt_min: float, rt_max: float) -> List[Dict]:
        """
        Search spectra by retention time range (in minutes)
        
        Args:
            rt_min: Minimum retention time (minutes)
            rt_max: Maximum retention time (minutes)
        
        Returns:
            List of matching spectra
        """
        matches = []
        rt_min_sec = rt_min * 60
        rt_max_sec = rt_max * 60
        
        for spec in self.spectra:
            rt_str = spec.get('RTINSECONDS', '')
            if rt_str:
                try:
                    rt_sec = float(rt_str)
                    if rt_min_sec <= rt_sec <= rt_max_sec:
                        matches.append(spec)
                except ValueError:
                    continue
        
        return matches
    
    def annotate_features_with_spectra(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate a feature table with spectrum information
        
        Args:
            feature_df: DataFrame with 'FeatureID' column
        
        Returns:
            DataFrame with added columns: 'HasSpectrum', 'NumPeaks', 'PrecursorMZ', 'RT_minutes'
        """
        result_df = feature_df.copy()
        result_df['HasSpectrum'] = False
        result_df['NumPeaks'] = 0
        result_df['PrecursorMZ'] = ''
        result_df['RT_minutes'] = ''
        
        # Create lookup dictionary
        spectrum_dict = {}
        for spec in self.spectra:
            spec_id = str(spec.get('FEATUREID', spec.get('SCANS', spec.get('TITLE', '')))).strip()
            spectrum_dict[spec_id] = spec
        
        # Annotate each feature
        for idx, row in result_df.iterrows():
            feature_id = str(row['FeatureID']).strip()
            
            if feature_id in spectrum_dict:
                spec = spectrum_dict[feature_id]
                result_df.at[idx, 'HasSpectrum'] = True
                result_df.at[idx, 'NumPeaks'] = len(spec.get('mzs', []))
                result_df.at[idx, 'PrecursorMZ'] = spec.get('PEPMASS', '')
                
                rt_sec = spec.get('RTINSECONDS', '')
                if rt_sec:
                    try:
                        result_df.at[idx, 'RT_minutes'] = f"{float(rt_sec)/60:.2f}"
                    except ValueError:
                        pass
        
        return result_df
    
    def export_annotated_features(self, feature_df: pd.DataFrame, output_path: str):
        """Export annotated feature table to CSV"""
        annotated_df = self.annotate_features_with_spectra(feature_df)
        annotated_df.to_csv(output_path, index=False)
        return output_path
