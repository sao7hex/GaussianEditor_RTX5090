import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import trimesh

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaussiansplatting.scene import Scene
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.arguments import ModelParams


def resolve_project_dir(project_arg: str) -> Path:
    """--project で指定された名前またはパスからプロジェクトルートを解決する"""
    candidate = Path(project_arg)
    if candidate.exists():
        return candidate.resolve()
    return (APP_DIR / "projects" / project_arg).resolve()


def resolve_project_inputs(project_dir: Path) -> Tuple[str, str, str]:
    """projects/<プロジェクト名>/input/ 配下から .glb, .ply, COLMAPディレクトリを探索する"""
    input_dir = project_dir / "input"
    if not input_dir.exists():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {input_dir}")

    # 1. .glb ファイルを検索
    glb_files = sorted(input_dir.rglob("*.glb"))
    if not glb_files:
        raise FileNotFoundError(f".glb が見つかりません: {input_dir}")
    glb_path = str(glb_files[0])

    # 2. .ply ファイルを検索 (COLMAPの sparse/ 配下にある points3D.ply 等は除外)
    ply_files = sorted(
        p for p in input_dir.rglob("*.ply")
        if "sparse" not in p.relative_to(input_dir).parts
    )
    if not ply_files:
        raise FileNotFoundError(f"3DGS用の .ply が見つかりません: {input_dir}")
    preferred = [p for p in ply_files if p.name.lower() == "point_cloud.ply"]
    ply_path = str(preferred[0] if preferred else ply_files[0])

    # 3. COLMAPディレクトリ (直下に sparse/ があるフォルダ) を検索
    colmap_dir = None
    if (input_dir / "sparse").is_dir():
        colmap_dir = str(input_dir)
    else:
        for sparse_dir in sorted(input_dir.rglob("sparse")):
            if sparse_dir.is_dir():
                colmap_dir = str(sparse_dir.parent)
                break
    if colmap_dir is None:
        raise FileNotFoundError(f"sparse/ を含むCOLMAPデータが見つかりません: {input_dir}")

    return glb_path, ply_path, colmap_dir


def load_glb(glb_path: str) -> List[Tuple[str, trimesh.Trimesh]]:
    """GLBを読み込み、(ノード名, 変換済みTrimesh) のリストを返す"""
    loaded = trimesh.load(glb_path, process=False)
    meshes: List[Tuple[str, trimesh.Trimesh]] = []
    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name].copy()
            geom.apply_transform(transform)
            meshes.append((node_name, geom))
    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append(("mesh_0", loaded))
    return meshes


def load_ply_and_colmap(ply_path: str, colmap_dir: str, work_dir: str, sh_degree: int = 3):
    """COLMAPカメラと指定PLYをロードして (gaussian, scene, cameras) を返す"""
    dummy_parser = argparse.ArgumentParser()
    lp = ModelParams(dummy_parser)
    model_params = lp.extract(dummy_parser.parse_args([]))
    model_params.source_path = os.path.abspath(colmap_dir)
    model_params.model_path = os.path.abspath(work_dir)  # input.ply / cameras.json 出力先
    model_params.sh_degree = sh_degree
    model_params.images = "images"
    model_params.resolution = -1
    model_params.data_device = "cuda"

    os.makedirs(model_params.model_path, exist_ok=True)

    gaussian = GaussianModel(
        sh_degree=sh_degree,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2.0,
    )

    # load_iteration=None にして自動PLY探索をスキップし、指定PLYを直接ロード
    scene = Scene(args=model_params, gaussians=gaussian, load_iteration=None, shuffle=False)
    gaussian.load_ply(ply_path)

    n_pts = gaussian.get_xyz.shape[0]
    gaussian.max_radii2D = torch.zeros(n_pts, device="cuda")

    train_cams = scene.getTrainCameras()
    return gaussian, scene, train_cams
