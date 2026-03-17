import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from src.data.loader import *
from src.visualisation.view_heatmap import show_hsi_viewer
from src.visualisation.view_unmixing import show_unmixing_viewer
from src.visualisation.view_predict import show_prediction_map

HELP_TEXT = {
    "args": """
  PATH: Absolute or relative path to your data directory.
        Accepts ~ for home directory (e.g. ~/data/SB008).
        The directory should contain .txt spectral data files.
        Example: ~/Data/SB008  or  /home/user/project/data/SB008

  ROWS (x): Number of rows in the grid (integer).
  COLS (y): Number of columns in the grid (integer).

  Example input: ~/Data/SB008 10 13
""",
    "mode": """
  Modes:
    1 - Heatmap:   Generate a spatial heatmap from spectral data.
    2 - Unmixing:  Decompose spectra into end-member components.
    3 - Predict:   Run prediction pipeline (incomplete).
""",
    "heatmap": """
  PIPELINE: Pre-processing pipeline to apply before plotting.

    0 - None (raw data)

    1 - P1: First preprocessing protocol from Georgiev et al. (2023) [1].
        Steps:
          - Cosmic ray removal (Whitaker-Hayes algorithm)
          - Denoising (Gaussian filter)
          - Baseline correction (Asymmetric Least Squares)
          - Normalisation (Area under the curve)

    2 - P2: Second preprocessing protocol from Georgiev et al. (2023) [1].
        Steps:
          - Cosmic ray removal (Whitaker-Hayes algorithm)
          - Denoising (Savitzky-Golay filter, window=9, poly order=3)
          - Baseline correction (Adaptive Smoothness Penalized Least Squares)
          - Normalisation (MinMax)

    3 - P3: Third preprocessing protocol from Georgiev et al. (2023) [1].
        Steps:
          - Cosmic ray removal (Whitaker-Hayes algorithm)
          - Baseline correction (polynomial fitting, order=3)
          - Normalisation (Vector)

  References:
    [1] Georgiev et al. (2023). RamanSPy: An open-source Python package for integrative
        Raman spectroscopy data analysis. arXiv:2307.13650.

  ROLLING WINDOW WIDTH: Viewing window in cm⁻¹ (e.g. 1.0).
  START / END: Wavenumber range to plot in cm⁻¹. Press Enter to use full range.
""",
    "unmixing": """
  END MEMBERS: Number of spectral components to unmix into.
               Enter -1 to estimate automatically via SVD.
  START / END: Wavenumber range to analyse in cm⁻¹. Press Enter to use full range.
""",
    "predict": """
  CONFIDENCE THRESHOLD: Minimum confidence (0-1) for a prediction to be displayed.
                        Pixels below this threshold are shown as 'Unknown'.
                        Example: 0.80 means only predictions ≥80% confidence are shown.
  SAVE PATH: File path to save the prediction map image.
             Press Enter to use the default (outputs/prediction_map.png).
""",
}

#TODO: not just via SVD

def _input(prompt_str=""):
    """Thin wrapper so callers can check for '?' and show help externally."""
    return input(prompt_str).strip()

def prompt_args():
    print("\nEnter path, grid rows and grid columns (or '?' for help):")
    print("e.g. ~/Code/Data_SH/SB008 10 13")
    while True:
        raw = _input("> ")
        if raw == "?":
            print(HELP_TEXT["args"])
            continue
        parts = raw.split()
        if len(parts) != 3:
            print("Please enter a path and two integers (rows, cols), or '?' for help.")
            continue
        path, x, y = parts
        try:
            x, y = int(x), int(y)
        except ValueError:
            print("x and y must be integers, please try again.")
            continue

        path = os.path.expanduser(path)

        if not os.path.isdir(path):
            print(f"Directory not found: {path}")
            print("Please check the path and try again.")
            continue

        txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]
        if not txt_files:
            print(f"No .txt files found in: {path}")
            print("Please check the path and try again.")
            continue

        n_files = len(txt_files)
        if x * y != n_files:
            print(f"Grid dimensions {x}x{y}={x*y} do not match the {n_files} .txt files found in: {path}")
            print("Please check the dimensions and try again.")
            continue

        print(f"Found {n_files} .txt file(s) in {path}.")
        break

    mode = prompt_mode()
    if mode == "heatmap":
        kwargs = prompt_heatmap_args()
    elif mode == "unmixing":
        kwargs = prompt_unmixing_args()
    elif mode == "predict":
        kwargs = prompt_predict_args()

    return path, x, y, mode, kwargs

def prompt_mode():
    print("\nSelect mode (or '?' for help):")
    print("  [1] Heatmap")
    print("  [2] Unmixing")
    print("  [3] Predict (incomplete)")
    choices = {"1": "heatmap", "2": "unmixing", "3": "predict"}
    while True:
        val = _input("> ")
        if val == "?":
            print(HELP_TEXT["mode"])
            continue
        if val in choices:
            return choices[val]
        print("Invalid choice, please enter 1, 2, or 3.")

def prompt_heatmap_args():
    print("\nPipeline? [0] None  [1] P1  [2] P2  [3] P3 (default: 0, or '?' for help)")
    while True:
        pipeline_id = _input("> ") or "0"
        if pipeline_id == "?":
            print(HELP_TEXT["heatmap"])
            continue
        if pipeline_id in ("0", "1", "2", "3"):
            break
        print("Please enter 0, 1, 2, or 3.")

    print("\nRolling window width in cm⁻¹ (default: 2, or '?' for help)")
    while True:
        rolling_window_width = _input("> ") or "2"
        if rolling_window_width == "?":
            print(HELP_TEXT["heatmap"])
            continue
        try:
            rolling_window_width = float(rolling_window_width)
            break
        except ValueError:
            print("Please enter a number (e.g. 1 or 25.5).")

    print("\nStart of spectra in cm⁻¹ (press Enter to skip, or '?' for help)")
    while True:
        start = _input("> ") or None
        if start == "?":
            print(HELP_TEXT["heatmap"])
            continue
        try:
            start = float(start) if start else None
            break
        except ValueError:
            print("Please enter a number or press Enter to skip.")

    print("\nEnd of spectra in cm⁻¹ (press Enter to skip, or '?' for help)")
    while True:
        end = _input("> ") or None
        if end == "?":
            print(HELP_TEXT["heatmap"])
            continue
        try:
            end = float(end) if end else None
            break
        except ValueError:
            print("Please enter a number or press Enter to skip.")

    return {
        "pipeline_id": int(pipeline_id),
        "rolling_window_width": rolling_window_width,
        "start": start,
        "end": end,
    }

def prompt_unmixing_args():
    print("\nNumber of end members (-1 to estimate automatically, or '?' for help)")
    while True:
        end_members = _input("> ") or "-1"
        if end_members == "?":
            print(HELP_TEXT["unmixing"])
            continue
        try:
            end_members = int(end_members)
            break
        except ValueError:
            print("Please enter an integer, or -1 to estimate automatically.")

    print("\nPipeline? [0] None  [1] P1  [2] P2  [3] P3 (default: 0, or '?' for help)")
    while True:
        pipeline_id = _input("> ") or "0"
        if pipeline_id == "?":
            print(HELP_TEXT["heatmap"])
            continue
        if pipeline_id in ("0", "1", "2", "3"):
            break
        print("Please enter 0, 1, 2, or 3.")

    print("\nStart of spectra in cm⁻¹ (press Enter to skip, or '?' for help)")
    while True:
        start = _input("> ") or None
        if start == "?":
            print(HELP_TEXT["unmixing"])
            continue
        try:
            start = float(start) if start else None
            break
        except ValueError:
            print("Please enter a number or press Enter to skip.")

    print("\nEnd of spectra in cm⁻¹ (press Enter to skip, or '?' for help)")
    while True:
        end = _input("> ") or None
        if end == "?":
            print(HELP_TEXT["unmixing"])
            continue
        try:
            end = float(end) if end else None
            break
        except ValueError:
            print("Please enter a number or press Enter to skip.")

    return {
        "end_members": end_members,
        "pipeline_id" : int(pipeline_id),
        "start": start,
        "end": end,
    }

def prompt_predict_args():
    print("\nConfidence threshold (default: 0.80, or '?' for help)")
    while True:
        confidence_threshold = _input("> ") or "0.80"
        if confidence_threshold == "?":
            print(HELP_TEXT["predict"])
            continue
        try:
            confidence_threshold = float(confidence_threshold)
            if 0 < confidence_threshold <= 1:
                break
            print("Please enter a value between 0 and 1 (e.g. 0.80).")
        except ValueError:
            print("Please enter a number between 0 and 1 (e.g. 0.80).")

    print("\nSave path for prediction map (default: outputs/prediction_map.png, or '?' for help)")
    while True:
        save_path = _input("> ") or "outputs/prediction_map.png"
        if save_path == "?":
            print(HELP_TEXT["predict"])
            continue
        break

    return {
        "confidence_threshold": confidence_threshold,
        "save_path": save_path,
    }

def main():
    path, x, y, mode, kwargs = prompt_args()

    if mode == "heatmap":
        cumulative_cube, spectra_of_pixel, raman_shift, pixel_map = get_cumulative_hsi_cube(
            path, x, y, kwargs["pipeline_id"],
            kwargs["start"], kwargs["end"]
        )
        show_hsi_viewer(cumulative_cube, spectra_of_pixel, raman_shift, pixel_map, x, y)

    elif mode == "unmixing":
        hsi_cube = get_raw_hsi_cube(path, x, y)
        pipeline = build_pipeline(kwargs["pipeline_id"])
        if pipeline is not None:
            hsi_cube = pipeline.apply(hsi_cube)
        show_unmixing_viewer(hsi_cube, kwargs["end_members"], kwargs["start"], kwargs["end"])

    elif mode == "predict":
        show_prediction_map(path, x, y, confidence_threshold=kwargs["confidence_threshold"], save_path=kwargs["save_path"])
        
if __name__ == "__main__":
    main()