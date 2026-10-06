### WebUI 説明書 (日本語版)

#### 1. データセットと事前学習済みモデルの準備
`download.sh` スクリプトを使用して MipNeRF-360 データセットをダウンロードします：
```bash
sh download.sh
```
または、[3DGS の手順](https://github.com/graphdeco-inria/gaussian-splatting#processing-your-own-scenes) に従って [InstructNerf2Nerf データセット](https://drive.google.com/drive/folders/1v4MLNoSwxvSlWb26xvjxeoHpgjhi_s-s) など他のデータセットを準備します。

[3DGS](https://github.com/graphdeco-inria/gaussian-splatting#evaluation) から事前学習済みの `.ply` モデルをダウンロードします（`download.sh` を使用した場合は既に完了しています）：
```bash
mkdir dataset
cd dataset
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
unzip models.zip
```
または、[3DGS の手順](https://github.com/graphdeco-inria/gaussian-splatting#running) に従ってゼロから学習させます。

プロジェクトページで紹介されている `.splat` ファイルは [こちら](https://huggingface.co/datasets/Yiwen-ntu/GaussianEditor_Result/tree/main) からダウンロードできます。

デモで使用した InstructNeRF2NeRF データセット（face, bear）の `.ply` ファイルおよび COLMAP 結果は [こちら](https://huggingface.co/datasets/Yiwen-ntu/GaussianEditor_Result/tree/main/InstructNeRF2NeRF_PLY_Files) にあります。

#### 2. WebUI の起動
データセットの準備ができたら、次のコマンドで WebUI を起動します：
```bash
python webui.py --gs_source <plyファイルのパス> --colmap_dir <データセットディレクトリ>
```
ここで `--gs_source` は事前学習済みの `.ply` ファイル（例: `../../point_cloud.ply` など）を指し、`--colmap_dir` は COLMAP の出力先ディレクトリ（COLMAP 出力の `sparse` フォルダが `--colmap_dir` のサブフォルダである必要があります）を指します。

InstructNeRF2NeRF データセットの face シーンを例にすると、[ここ](https://huggingface.co/datasets/Yiwen-ntu/GaussianEditor_Result/tree/main/InstructNeRF2NeRF_PLY_Files) からダウンロードした後、`--gs_source` は `"../face/face.ply"`、`--colmap_dir` は `"../face"` となります。

`download.sh` を使用した場合（3DGS の事前学習済み GS を使用し、対応する `.ply` ファイルを `./dataset/<scene-name>` にダウンロードした場合）、以下のように起動できます：
```bash
python webui.py \
    --colmap_dir ./dataset/<scene-name> \
    --gs_source ./dataset/<scene-name>/point_cloud/iteration_7000/point_cloud.ply
```

リモートサーバーを使用している場合は、以下のコマンドを実行してウェブサイトをローカルにポートフォワーディングしてください：
```bash
ssh -L 8084:127.0.0.1:8084 ユーザー名@リモートサーバーのアドレス
```
その後、ブラウザで `127.0.0.1:8084` を開いて編集を開始します。

#### 3. WebUI ステップバイステップガイド
まず論文と [デモ動画](https://www.youtube.com/watch?v=TdZIICSFqsU&ab_channel=YiwenChen) をご覧になることをお勧めします。
GaussianEditor の WebUI は現在、次の5つの機能を備えています：**テキストによるセマンティック追跡（3Dセグメンテーション）、クリックによるセマンティック追跡、編集（Edit）、削除（Delete）、追加（Add）**。

WebUI は **セグメンテーション、編集、削除** の学習視点として COLMAP から出力されたカメラ情報を必要とします。現在、受け入れ可能なのは `PINHOLE` カメラのみです。学習には COLMAP カメラがロードされるため、WebUI 上の現在の視点は学習プロセスに影響しないことに注意してください。まず WebUI の基本操作を説明し、その後上記5つの機能を順番に解説します。

##### (1) 基本操作（Basic Usage）
<img width="235" alt="1701045938591" src="https://github.com/buaacyw/GaussianEditor/assets/52091468/bcb8ef14-651b-47d8-b816-064ed72cab8c">

- `Resolution`: 画面に送信されるレンダリング解像度。これを変更しても **編集** や **削除** の学習解像度（512固定）には影響しません。ただし、2Dマスクをユーザーが指定する **追加（Add）** で使用される2Dインペインティングには影響します。WebUI の動作が非常に重い場合は、`Resolution` を下げてください。
- `FoV Scaler`: 画面に送信されるレンダリングの視野角（FoV）倍率。`Resolution` と同様に、**追加（Add）** にのみ影響します。通常、これを変更する必要はありません。
- `Renderer Output`: 画面に送信されるレンダラーの出力タイプ。深度（Depth）は将来的にアップデート予定です。
- `Save Gaussian`: 編集完了後、クリックするとガウシアンが `ui_result` に保存されます。保存後、WebUI を再起動して保存したガウシアンをロードすることで、2回目の編集を適用できます。WebUI を1回起動した状態で直接複数回編集を行うことにはまだバグが存在します。
- `Show Frame`: [Viser](https://github.com/nerfstudio-project/viser/tree/main/examples) の視点ナビゲーション設定により、ガウシアンシーンのフレームはデフォルトで標準化されていないことが多いため、カメラ操作がやや難しく感じられる場合があります。その場合は `Show Frame` をクリックし、COLMAP からプリロードされたカメラ視点のいずれかにジャンプしてください。

##### (2) テキストによるセマンティック追跡（Semantic Tracing by Text）

![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/1e66ce57-aa79-4144-9b4c-9918712ce0fb)

手順:
1. `Text Seg Prompt` に目的の対象を指定します。
2. `Tracing Begin!` をクリックします。
3. 30〜60秒後、`Seg Threshold`（セグメンテーションの信頼度スコアを制御するスライダー）の調整が可能になります。
   
![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/eac3b13f-46bd-4b87-bbed-3c820fbd016a)

4. `End Seg Scale!` をクリックして最終結果を確定します。
5. `Semantic Group` に `Text Seg Prompt` で入力したテキスト名で新しいグループが追加されます。グループを切り替えて異なるマスクを確認できます。

- `Text Seg Prompt`: SAM を実行するためのプロンプト。
- `Semantic Group`: すべてのセグメンテーション結果。これらを切り替えて使用します。
- `Seg Camera Nums`: SAM に使用するカメラ視点の数。視点数が少ないほど高速にセグメンテーションできます。通常は12視点で十分な結果が得られます。
- `Show Semantic Mask`: マスクを表示します。これは学習には影響しません。

##### (3) クリックによるセマンティック追跡（Tracing by Click）

手順:
1. `Enable SAM` と `Add SAM Points` を開きます（有効化）。
2. セグメンテーションしたい部分をクリックします。単一の視点からポイントを追加するだけでも良好な結果が得られるため、通常は視点を移動する必要はありません。視点を動かしたい場合は、まず `Add SAM Points` を閉じ、視点を移動してから再度開いてください。
3. ポイントを追加した後、`Add SAM Points` を閉じ、`SAM Group Name` に目的のパーツ名を指定します。`SAM Group Name` は `Semantic Group` 内の名前としてのみ使用され、セグメンテーションのテキストプロンプトとしては使用されません。
4. `Show Semantic Mask` をクリックしてセグメンテーション結果を確認します。

##### (4) 編集（Edit）
![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/7b0a13b6-dec3-4135-b892-3bf5e4a7315d)

プロンプトを入力して編集を開始するだけです。`Semantic Group` を切り替えることで、変更したいパーツを指定できます。デフォルトの `ALL` はガウシアン全体が更新されることを意味し、この場合は [ハイパーパラメータ調整ガイド](https://github.com/buaacyw/GaussianEditor/blob/95a0bbfb0e88c84a963ab3b67eed416b4af0fc60/docs/hyperparameter.md?plain=1#L22) で述べられているように高密度化（densification）を減少させる必要があります。学習開始後、`Show Edit Frame` を開くことで現在の編集2Dフレームを確認できます。

##### (5) 削除（Delete）

![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/09c9da14-7ac0-4903-9688-39f095428a39)

**編集（Edit）** と同じ手順ですが、`Semantic Group` でマスクを **必ず指定** する必要があります。マスクされた部分が削除され、この処理によって生じたアーティファクトを修復するために2Dインペインティング手法が使用されます。`Text` は背景のアーティファクトを修正するために使用されることに注意してください。したがって、`Text` にオブジェクトカテゴリを入力するのではなく、背景の説明を入力する必要があります。

##### (6) 追加（Add）

**追加（Add）** の使用はやや複雑なため、まず公式動画をご覧になることを強くお勧めします。

![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/de41f8a5-7b61-4501-84fd-16f4444fc02a)

手順:
1. インペイントマスクを指定します。まず `Draw Bounding Box` をクリックし、画面上をクリックします。最初に左上、次に右下です。`Left Up` と `Right Down` に幅と高さの順序でピクセル座標が表示されます。`Left Up` と `Right Down` に直接数値を入力することも可能です。
2. `Text` にインペイントしたい内容のテキストを指定します。
3. `Edit Begin!` をクリックします。クリック後、2Dインペインティングが開始され、インペインティングに使用されるカメラ視点と2Dマスクが固定されます。これで自由にカメラ視点を動かしたり画面上をクリックしたりできます。

![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/5988aba9-f2cb-497f-b3f7-d1340ba3ae2b)

4. 数秒後に2Dインペインティングの結果が表示されます。結果に満足できない場合は、プロンプトとシード値を変更してより良い結果を得るようにしてください。その後、`End 2D Inpainting!` をクリックします。
5. `Refine text` を入力します。これは [Wonder3D](https://github.com/xxlong0/Wonder3D) によって生成された粗いメッシュを [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) で微調整するために使用されます。粗いメッシュをリファインするために "make it a ...." のようなプロンプトを使用する必要があります。

![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/297f79ab-3f78-4cfc-89ec-bd10357f16c9)

6. 約7〜8分後、画面に結果が `Depth Scale` と共に表示されます。DPT を使用して深度を予測し、生成されたガウシアンをガウシアンシーンと位置合わせします。残念ながら DPT は十分に良好な深度マップを提供できない場合があるため、その場合は DPT が予測した深度を手動でスケーリング（調整）します。
7. 最後に `End Depth Scale` をクリックして最終結果を取得します。
