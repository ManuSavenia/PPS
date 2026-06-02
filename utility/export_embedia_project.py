from __future__ import annotations

import argparse
import re as stdlib_re
import shutil
import types
import sys
from pathlib import Path

try:
    import joblib
except Exception:
    import pickle as joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")
    return df


def select_balanced_examples(df: pd.DataFrame, per_class: int, split_name: str) -> pd.DataFrame:
    label_column = df.columns[-1]
    selected_parts = []

    for label in sorted(df[label_column].dropna().unique()):
        class_rows = df[df[label_column] == label].head(per_class).copy()
        if class_rows.empty:
            continue
        class_rows.insert(0, "source_split", split_name)
        class_rows.insert(1, "source_label", label)
        class_rows.insert(2, "source_row", class_rows.index.astype(int))
        selected_parts.append(class_rows)

    if not selected_parts:
        raise ValueError(f"No samples selected from {split_name}")

    return pd.concat(selected_parts, ignore_index=True)


def build_example_bundle(train_df: pd.DataFrame, test_df: pd.DataFrame, train_per_class: int, test_per_class: int) -> pd.DataFrame:
    parts = []

    if train_per_class > 0:
        parts.append(select_balanced_examples(train_df, train_per_class, "train"))

    if test_per_class > 0:
        parts.append(select_balanced_examples(test_df, test_per_class, "test"))

    if not parts:
        raise ValueError("At least one of train_per_class or test_per_class must be greater than zero")

    bundle = pd.concat(parts, ignore_index=True)
    return bundle


def export_raw_bundle(bundle: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.to_csv(output_path, index=False)


def copy_full_quant8_runtime(embedia_root: Path, project_embedia_dir: Path) -> None:
    runtime_root = embedia_root / "embedia" / "libraries" / "mcu" / "generic" / "full_quant8"
    files_to_copy = [
        "common.c",
        "common.h",
        "neural_net.c",
        "neural_net.h",
        "quant8.c",
        "quant8.h",
        "realtype.h",
    ]

    project_embedia_dir.mkdir(parents=True, exist_ok=True)

    for filename in files_to_copy:
        shutil.copy2(runtime_root / filename, project_embedia_dir / filename)

    runtime_types_dir = runtime_root / "neural_net"
    project_types_dir = project_embedia_dir / "neural_net"
    project_types_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime_types_dir / "_types.h", project_types_dir / "_types.h")


def prepare_example_arrays(bundle: pd.DataFrame):
    feature_columns = bundle.columns[3:-1]
    label_column = bundle.columns[-1]

    features = bundle.loc[:, feature_columns].to_numpy(dtype=np.float32)
    labels = bundle.loc[:, label_column].to_numpy(dtype=np.int32).reshape(-1, 1)
    return features, labels


def load_feature_normalizer(repo_root: Path):
    normalizer_path = repo_root / "Cuantization_Test" / "Data_Sets" / "metadata" / "normalizer.pkl"
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Normalizer not found: {normalizer_path}")
    return joblib.load(normalizer_path)


def configure_embedia_project(embedia_root: Path):
    if not embedia_root.exists():
        raise FileNotFoundError(f"EmbedIA root not found: {embedia_root}")

    if "regex" not in sys.modules:
        sys.modules["regex"] = stdlib_re

    if "tensorboard" not in sys.modules:
        tensorboard_module = types.ModuleType("tensorboard")
        tensorboard_manager_module = types.ModuleType("tensorboard.manager")

        def data_source_from_info(*args, **kwargs):
            return None

        tensorboard_manager_module.data_source_from_info = data_source_from_info
        tensorboard_module.manager = tensorboard_manager_module

        sys.modules["tensorboard"] = tensorboard_module
        sys.modules["tensorboard.manager"] = tensorboard_manager_module

    if "prettytable" not in sys.modules:
        prettytable_module = types.ModuleType("prettytable")

        class PrettyTable:
            def __init__(self, *args, **kwargs):
                self.field_names = []
                self.align = {}
                self.rows = []

            def add_row(self, row):
                self.rows.append(list(row))

            def __str__(self):
                header = " | ".join(map(str, self.field_names)) if self.field_names else ""
                body = "\n".join(" | ".join(map(str, row)) for row in self.rows)
                return "\n".join(part for part in [header, body] if part)

        prettytable_module.PrettyTable = PrettyTable
        sys.modules["prettytable"] = prettytable_module

    try:
        import IPython  # noqa: F401
    except ModuleNotFoundError:
        ipython_module = types.ModuleType("IPython")
        ipython_utils_module = types.ModuleType("IPython.utils")
        ipython_terminal_module = types.ModuleType("IPython.utils.terminal")

        def set_term_title(*args, **kwargs):
            return None

        ipython_terminal_module.set_term_title = set_term_title
        ipython_utils_module.terminal = ipython_terminal_module
        ipython_module.utils = ipython_utils_module

        sys.modules["IPython"] = ipython_module
        sys.modules["IPython.utils"] = ipython_utils_module
        sys.modules["IPython.utils.terminal"] = ipython_terminal_module

    sys.path.insert(0, str(embedia_root.parent))

    from embedia.model_generator.project_options import (  # noqa: WPS433
        DebugMode,
        ModelDataType,
        ModelMicro,
        ProjectFiles,
        ProjectOptions,
        ProjectType,
    )
    import embedia.model_generator.generate_files as generate_files  # noqa: WPS433
    from embedia.project_generator import ProjectGenerator  # noqa: WPS433

    original_find_source_file = generate_files.find_source_file

    def recursive_find_source_file(filename, search_paths):
        found = original_find_source_file(filename, search_paths)
        if found is not None:
            return found

        for folder in search_paths:
            folder_path = Path(folder)
            if not folder_path.exists():
                continue
            for candidate in folder_path.rglob(filename):
                return str(candidate)

        for candidate in embedia_root.rglob(filename):
            return str(candidate)

        return None

    generate_files.find_source_file = recursive_find_source_file

    return (
        DebugMode,
        ModelDataType,
        ModelMicro,
        ProjectFiles,
        ProjectOptions,
        ProjectType,
        ProjectGenerator,
    )


def export_embedia_project(
    repo_root: Path,
    embedia_root: Path,
    output_folder: Path,
    project_name: str,
    train_per_class: int,
    test_per_class: int,
):
    embedia_package_root = embedia_root / "embedia"

    (
        DebugMode,
        ModelDataType,
        ModelMicro,
        ProjectFiles,
        ProjectOptions,
        ProjectType,
        ProjectGenerator,
    ) = configure_embedia_project(embedia_package_root)

    model_path = repo_root / "Cuantization_Test" / "Models" / "base" / "fingers_model_no_quantization.h5"
    train_csv = repo_root / "Cuantization_Test" / "Data_Sets" / "raw" / "fingers_train.csv"
    test_csv = repo_root / "Cuantization_Test" / "Data_Sets" / "raw" / "fingers_test.csv"

    model = load_model(model_path)
    train_df = load_raw_dataset(train_csv)
    test_df = load_raw_dataset(test_csv)
    bundle = build_example_bundle(train_df, test_df, train_per_class, test_per_class)

    output_folder.mkdir(parents=True, exist_ok=True)
    raw_bundle_path = output_folder / f"{project_name}_raw_examples.csv"
    export_raw_bundle(bundle, raw_bundle_path)

    features, labels = prepare_example_arrays(bundle)
    normalizer = load_feature_normalizer(repo_root)
    features = normalizer.transform(features).astype(np.float32)

    options = ProjectOptions()
    options.embedia_folder = str(embedia_package_root)
    options.project_type = ProjectType.C
    options.micro = ModelMicro.GENERIC
    options.data_type = ModelDataType.FULL_QUANT8
    options.debug_mode = DebugMode.DISCARD
    options.files = {ProjectFiles.LIBRARY, ProjectFiles.MAIN, ProjectFiles.MODEL}
    options.example_data = features
    options.example_ids = labels
    options.example_labels = labels
    options.clean_output = True
    options.verbose = True
    options.output_subfolder = "embedia"

    generator = ProjectGenerator(options)
    generator.create_project(str(output_folder), project_name, model, options)

    copy_full_quant8_runtime(embedia_root, output_folder / project_name / "embedia")

    return {
        "model_path": model_path,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "raw_examples": raw_bundle_path,
        "project_folder": output_folder / project_name,
        "project_name": project_name,
    }


def parse_args():
    repo_root = get_repo_root()
    default_output = repo_root / "EmbedIA_exports"

    parser = argparse.ArgumentParser(
        description="Export the PPS finger-classification model to an EmbedIA FULL_QUANT8 project."
    )
    parser.add_argument(
        "--embedia-root",
        type=Path,
        default=Path("/home/manuel/Documents/Facultad/EmbedIA"),
        help="Path to the EmbedIA framework repository.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=default_output,
        help="Folder where the generated EmbedIA project will be written.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="fingers_full_quant8",
        help="Name of the generated EmbedIA project.",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=1,
        help="How many normalized train samples to include per class.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=1,
        help="How many normalized test samples to include per class.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = get_repo_root()

    print("PPS -> EmbedIA export")
    print(f"  Repo root:     {repo_root}")
    print(f"  EmbedIA root:  {args.embedia_root}")
    print(f"  Package root:  {args.embedia_root / 'embedia'}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  Project name:  {args.project_name}")
    print(f"  Samples:       train/class={args.train_per_class}, test/class={args.test_per_class}")
    print("  Decisions: use the float Keras model from PPS, keep the raw normalized samples as a CSV snapshot, and let EmbedIA generate the FULL_QUANT8 project without modifying the framework.")

    result = export_embedia_project(
        repo_root=repo_root,
        embedia_root=args.embedia_root,
        output_folder=args.output_folder,
        project_name=args.project_name,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )

    print("Export complete")
    print(f"  Project folder: {result['project_folder']}")
    print(f"  Raw examples:   {result['raw_examples']}")
    print(f"  Model source:   {result['model_path']}")
    print(f"  Train CSV:      {result['train_csv']}")
    print(f"  Test CSV:       {result['test_csv']}")


if __name__ == "__main__":
    main()