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
    parser.add_argument("--depth_tolerance", type=float, default=0.0005, help="同一平面判定の深度許容誤差 (デフォルト: 0.0005)")
    parser.add_argument("--double_sided", action="store_true", help="背面カリングを無効化し、法線が裏向きのポリゴンも両面描画する")
    parser.add_argument("--render_last_mesh", action="store_true", help="最後のメッシュ (地面等) を透過せずそのまま描画する")
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

    # 3. COLMAP カメラ視点から GLB メッシュをレンダリング (オクルーダー版 & 非オクルーダー版の両方を出力)
    occluded_dir = project_dir / "output" / "renders_occluded"
    full_dir = project_dir / "output" / "renders_full"
    cam_indices = [args.cam_id] if args.cam_id is not None else None
    saved_paths = render_glb_from_cameras(
        meshes=meshes,
        cameras=train_cams,
        output_dir=str(occluded_dir),
        full_output_dir=str(full_dir),
        cam_indices=cam_indices,
        flat=not args.shaded,
        alpha_mode=args.alpha_mode,
        alpha_cutoff=args.alpha_cutoff,
        depth_tolerance=args.depth_tolerance,
        double_sided=args.double_sided,
        occlude_last_mesh=not args.render_last_mesh,
    )
    print(f"レンダリング完了: {len(saved_paths)} 枚ずつ保存しました")
    print(f"  - オクルーダー版 (最後のメッシュ透過): {occluded_dir}")
    print(f"  - 通常版 (全メッシュ描画)          : {full_dir}")


if __name__ == "__main__":
    main()
