import sys
import argparse
from pathlib import Path

# スクリプト配置ディレクトリとリポジトリルートをパスに追加
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for p in (str(APP_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from io_utils import (
    resolve_project_dir,
    resolve_project_inputs,
    load_glb,
    load_ply_and_colmap,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, required=True, help="projects/ 配下のプロジェクト名またはパス")
    args = parser.parse_args()

    # 1. 入力パスの解決
    project_dir = resolve_project_dir(args.project)
    glb_path, ply_path, colmap_dir = resolve_project_inputs(project_dir)
    print(f"Project: {project_dir}")
    print(f"GLB    : {glb_path}")
    print(f"PLY    : {ply_path}")
    print(f"COLMAP : {colmap_dir}")

    # 2. データの読み込み
    meshes = load_glb(glb_path)
    gaussian, scene, train_cams = load_ply_and_colmap(
        ply_path=ply_path,
        colmap_dir=colmap_dir,
        work_dir=str(project_dir / "_cache"),
    )
    print(f"読み込み完了: メッシュ数={len(meshes)}, ガウシアン点数={gaussian.get_xyz.shape[0]:,}, カメラ数={len(train_cams)}")


if __name__ == "__main__":
    main()
