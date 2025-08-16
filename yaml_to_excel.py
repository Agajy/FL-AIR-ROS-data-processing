import os
import json
import pandas as pd

def flatten_pose_data(data: dict, prefix: str = ""):
    flat = {}
    flat["prefix"] = prefix
    # Conversion explicite du timestamp en float si possible
    ts = data.get("timestamp")
    try:
        flat["timestamp"] = float(ts) if ts is not None else None
    except (ValueError, TypeError):
        flat["timestamp"] = ts
    flat["frame_id"] = data.get("frame_id")
    position = data.get("position", {})
    orientation = data.get("orientation", {})
    for k, v in position.items():
        flat[f"pos_{k}"] = v
    for k, v in orientation.items():
        flat[f"ori_{k}"] = v
    return flat

def process_complex_json(data_list):
    rows = []
    for entry in data_list:
        row = []
        for key, value in entry.items():
            if value is None or not isinstance(value, dict):
                continue
            flat = flatten_pose_data(value, prefix=key)
            row.append(flat)
        rows.extend(row)
    return pd.DataFrame(rows)

def process_simple_json(data_list):
    rows = [flatten_pose_data(entry, prefix=entry.get("frame_id", "unknown")) for entry in data_list]
    return pd.DataFrame(rows)

def process_json_file(filepath):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Erreur de lecture JSON dans {filepath}: {e}")
            return None

    if isinstance(data, list):
        if isinstance(data[0], dict) and all("position" in d for d in data):
            return process_simple_json(data)
        else:
            return process_complex_json(data)
    else:
        print(f"Format inconnu dans {filepath}")
        return None

def process_all_folders(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"Traitement du dossier : {folder_name}")
        excel_path = os.path.join(folder_path, f"{folder_name}.xlsx")
        sheet_data = {}

        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".json") or file_name == "metadata.json":
                continue

            json_path = os.path.join(folder_path, file_name)
            print(f"Traitement du fichier : {file_name}")

            df = process_json_file(json_path)
            if df is not None and not df.empty:
                sheet_name = os.path.splitext(file_name)[0][:31]  # nom max 31 caractères
                sheet_data[sheet_name] = df

        if sheet_data:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for sheet_name, df in sheet_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Fichier Excel généré : {excel_path}\n")
        else:
            print(f"Aucun fichier JSON exploitable dans '{folder_name}', Excel non généré.\n")


if __name__ == "__main__":
    dossier_racine = r"C:\Users\aurel\Bureau\A_volière mesure"  
    process_all_folders(dossier_racine)
