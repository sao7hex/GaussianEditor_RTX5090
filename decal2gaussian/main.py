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
from merge_renderer import merge_occluded_with_original_images
from gs_trainer import train_3dgs_from_merged_images
from train_runner import run_gaussiansplatting_train_py


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
    parser.add_argument("--gs_iterations", type=int, default=3000, help="3DGS 学習ステップ数 (デフォルト: 3000)")
    parser.add_argument("--no_refine", action="store_true", help="画像を結合するときに、refineせずに結合する")
    parser.add_argument("--from_scratch", action="store_true", help="既存PLYを使わず、COLMAP初期点群からゼロから3DGSを生成する")
    parser.add_argument("--use_train_py", action="store_true", help="gaussiansplatting/train.py を呼び出して3DGSを生成する")
    parser.add_argument("--skip_gs", action="store_true", help="3DGS学習をスキップし、レンダリングと画像合成のみ行う")
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
        load_ply=not args.from_scratch,
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

    # 4. 元画像 (背面) とオクルーダー版レンダリング画像 (前面) を2Dレイヤー合成
    if args.no_refine:
        merged_dir = project_dir / "output" / "renders_not-refine_merged"
    else:
        merged_dir = project_dir / "output" / "renders_refine_merged"
    merged_paths = merge_occluded_with_original_images(
        cameras=train_cams,
        colmap_dir=colmap_dir,
        occluded_dir=str(occluded_dir),
        output_dir=str(merged_dir),
        cam_indices=cam_indices,
        no_refine_Flag=args.no_refine
    )
    print(f"レイヤー合成完了: {len(merged_paths)} 枚を保存しました -> {merged_dir}")

    # 5. renders_merged/ の画像で 3DGS シーンを生成・保存
    if not args.skip_gs:
        # --use_train_pyをオンにした時，train_runner.pyが動く
        if args.use_train_py:
            saved_ply = run_gaussiansplatting_train_py(
                root_dir=ROOT_DIR,
                colmap_dir=colmap_dir,
                merged_dir=str(merged_dir),
                cameras=train_cams,
                work_dataset_dir=str(project_dir / "_cache" / "train_dataset"),
                model_out_dir=str(project_dir / "output" / "train_py_model"),
                iterations=args.gs_iterations,
            )
            print(f"train.py による 3DGS 生成完了: {saved_ply}")

        # 何も指定していない・--from_scratchをオンにした時，gs_trainer.pyが動く
        else:
            output_ply = project_dir / "output" / "finetuned_model" / "point_cloud.ply"
            saved_ply = train_3dgs_from_merged_images(
                gaussian=gaussian,
                scene=scene,
                cameras=train_cams,
                merged_dir=str(merged_dir),
                output_ply_path=str(output_ply),
                iterations=args.gs_iterations,
                from_scratch=args.from_scratch,
            )
            print(f"3DGS 生成完了: {saved_ply}")


if __name__ == "__main__":
    main()
