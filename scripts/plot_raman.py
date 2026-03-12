from src.data.loader import *
from src.visualisation.spatial import *
from src.cnn.predict import predict
from src.visualisation.spatial import show_prediction_map

#TODO: since the trained model does so well on preprocessed data, pass in preprocessed data
def prompt_args():
    print("\nEnter path, grid rows and grid columns:")
    print("e.g. ~/Code/Data_SH/SB008 10 13")
    while True:
        parts = input("> ").strip().split()
        if len(parts) == 3:
            path, x, y = parts[0], parts[1], parts[2]
            try:
                x, y = int(x), int(y)
                break
            except ValueError:
                print("x and y must be integers, please try again")
        else:
            path, x, y = "~/Code/Data_SH/SB008", "10", "13"
            try:
                x, y = int(x), int(y)
                break
            except ValueError:
                print("x and y must be integers, please try again")

    mode = prompt_mode()

    if mode == "heatmap":
        kwargs = prompt_heatmap_args()
    elif mode == "unmixing":
        kwargs = prompt_unmixing_args()
    elif mode == "predict":
        kwargs = {}

    return path, x, y, mode, kwargs

def prompt_mode():
    print("\n Select mode:")
    print("  [1] Heatmap")
    print("  [2] Unmixing")
    print("  [3] Predict")

    choices = {"1": "heatmap", "2": "unmixing", "3": "predict"}

    while True:
        val = input("> ").strip()
        if val in choices:
            return choices[val]
        print(f"Invalid choice, please enter 1, 2, or 3.")

def prompt_heatmap_args():
    print("\nPipeline? [0] None  [1] P1  [2] P2  [3] P3  (default: 0)")
    pipeline_id = input("> ").strip() or "0"
    while pipeline_id not in ("0", "1", "2", "3"):
        print("Please enter 0, 1, 2 or 3")
        pipeline_id = input("> ").strip() or "0"

    print("\nRolling window width in cm⁻¹ (default: 1)")
    rolling_window_width = input("> ").strip() or "1"

    print("\nSpectra start in cm⁻¹ (press Enter to skip)")
    start = input("> ").strip() or None

    print("\nSpectra end in cm⁻¹ (press Enter to skip)")
    end = input("> ").strip() or None

    return {
        "pipeline_id": int(pipeline_id),
        "rolling_window_width": float(rolling_window_width),
        "start": float(start) if start else None,
        "end": float(end) if end else None,
    }

def prompt_unmixing_args():
    print("\nNumber of end_members (press Enter to estimate automatically)")
    end_members = input("> ").strip() or "-1"

    print("\nStart of spectra region in cm⁻¹ (press Enter to skip)")
    start = input("> ").strip() or None

    print("\nEnd of spectra region in cm⁻¹ (press Enter to skip)")
    end = input("> ").strip() or None

    return {
        "end_members": int(end_members),
        "start": float(start) if start else None,
        "end": float(end) if end else None,
    }

def main():
    
    path, x, y, mode, kwargs = prompt_args()

    if mode == "heatmap":
        area_cube, spectra_of_pixel, raman_shift, idx_step, pixel_map = get_area_under_hsi_cube(
            path, x, y, kwargs["pipeline_id"], kwargs["rolling_window_width"],
            kwargs["start"], kwargs["end"]
        )
        show_hsi_viewer(area_cube, spectra_of_pixel, raman_shift, idx_step, pixel_map, x, y)

    elif mode == "unmixing":
        hsi_cube = get_raw_hsi_cube(path, x, y)
        show_unmixing_viewer(hsi_cube, kwargs["end_members"], kwargs["start"], kwargs["end"])

    elif mode == "predict":
        predicted_labels_map, predicted_top5_map = predict(path, x, y)
        show_prediction_map(
            predicted_labels_map, predicted_top5_map,
            save_path="outputs/mineral_map.png"
        )

if __name__ == "__main__":
    main()