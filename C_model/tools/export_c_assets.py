import os
import sys


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from utility.export_c_assets import export_c_assets

    out_dir = export_c_assets(project_root=root_dir)
    print("Export complete:", out_dir)


if __name__ == "__main__":
    main()
