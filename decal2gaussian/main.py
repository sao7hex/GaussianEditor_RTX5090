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
from mesh_renderer import render_glb_from_cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, required=True, help="projects/ 配下のプロジェクト名またはパス")
    parser.add_argument("--cam_id", type=int, default=None, help="特定のカメラIDのみレンダリングする場合に指定 (未指定時は全カメラ)")
    parser.add_argument("--shaded", action="store_true", help="ライティングあり (非フラット) でレンダリングする")
    parser.add_argument("--alpha_mode", type=str, default="MASK", choices=["MASK", "BLEND", "OPAQUE"], help="透明領域の描画モード (デフォルト: MASK)")
    parser.add_argument("--alpha_cutoff", type=float, default=0.05, help="MASKモード時の透明判定しきい値 (デフォルト: 0.05)")
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

    # 3. COLMAP カメラ視点から GLB メッシュをレンダリング
    render_dir = project_dir / "output" / "renders"
    cam_indices = [args.cam_id] if args.cam_id is not None else None
    saved_paths = render_glb_from_cameras(
        meshes=meshes,
        cameras=train_cams,
        output_dir=str(render_dir),
        cam_indices=cam_indices,
        flat=not args.shaded,
        alpha_mode=args.alpha_mode,
        alpha_cutoff=args.alpha_cutoff,
    )
    print(f"レンダリング完了: {len(saved_paths)} 枚を保存しました -> {render_dir}")


if __name__ == "__main__":
    main()
