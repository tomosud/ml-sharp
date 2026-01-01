# Depth生成処理の調査報告

## 概要

本ドキュメントは、SHARP (Single-image Human-centric Scene Gaussian Splatting) プロジェクトにおけるdepth生成処理の実装を調査し、事前準備したRGB画像とdepth画像を使用して後段の処理を行う際の実装方針をまとめたものです。

## 現在の処理フロー

### 1. エントリポイント

**ファイル**: [run.bat](run.bat:8)

```batch
sharp predict -i input\images -o output\gaussians -c models\sharp_2572gikvuh.pt
```

- `sharp predict`コマンドを実行
- 入力: RGB画像ディレクトリ
- 出力: 3D Gaussians (.ply形式)
- モデル: 学習済みチェックポイント

### 2. Predict処理

**ファイル**: [src/sharp/cli/predict.py](src/sharp/cli/predict.py:159-206)

**主要な処理フロー**:

```python
def predict_image(predictor, image, f_px, device):
    # 1. 画像を1536x1536にリサイズ
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(1536, 1536),
        mode="bilinear",
        align_corners=True,
    )

    # 2. NDC空間でGaussiansを予測
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    # 3. メートル空間に変換
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device),
        intrinsics_resized, internal_shape
    )

    return gaussians
```

**処理の詳細**:
- 入力画像を内部解像度(1536x1536)にリサイズ
- `disparity_factor = f_px / width` を計算
- PredictorでNDC空間のGaussiansを生成
- Unprojectionでメートル空間に変換

### 3. Depth生成の核心部分

**ファイル**: [src/sharp/models/predictor.py](src/sharp/models/predictor.py:103-192)

**クラス**: `RGBGaussianPredictor`

**forward処理**:

```python
def forward(self, image, disparity_factor, depth=None):
    # ① Monodepthモデルでdisparityを推定
    monodepth_output = self.monodepth_model(image)  # Line 127
    monodepth_disparity = monodepth_output.disparity

    # ② Disparityからdepthに変換
    disparity_factor = disparity_factor[:, None, None, None]
    monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)  # Line 131

    # ③ Depth alignmentを適用 (オプション)
    monodepth, _ = self.depth_alignment(
        monodepth, depth, monodepth_output.decoder_features
    )  # Line 176-180

    # ④ Initializerで基本的なGaussianパラメータを生成
    init_output = self.init_model(image, monodepth)  # Line 182

    # ⑤ Feature modelでGaussian特徴量を計算
    image_features = self.feature_model(
        init_output.feature_input,
        encodings=monodepth_output.output_features
    )  # Line 183-185

    # ⑥ Prediction headでdelta値を予測
    delta_values = self.prediction_head(image_features)  # Line 186

    # ⑦ Composerで最終的なGaussiansを合成
    gaussians = self.gaussian_composer(
        delta=delta_values,
        base_values=init_output.gaussian_base_values,
        global_scale=init_output.global_scale,
    )  # Line 187-191

    return gaussians
```

**処理グラフ**:

```
monodepth        depth (optional)
    |              |
    +------+-------+
           |
   +-------+--------+
   |depth_alignment |     # Ground truthへのアライメント (オプション)
   +-------+--------+
           |
           v
  monodepth (aligned)
           |
     +-----+----+
     |init_model|           # Base Gaussiansの初期化
     +-----+----+
           |
           v
    init_output
      |       |
      |  +----+-----+
      |  |  feature |       # Delta値の予測
      |  |   model  |
      |  +----+-----+
      |       |
      |       v
      |  delta_values
      |       |
      |  +----+----------+
      +->|gaussian      |  # 最終的なGaussiansの合成
         |composer      |
         +----+----------+
              |
              v
          gaussians
```

### 4. Monodepthモデル

**ファイル**: [src/sharp/models/monodepth.py](src/sharp/models/monodepth.py)

**クラス**: `MonodepthDensePredictionTransformer` (30-103行)

**アーキテクチャ**: Vision Transformer (DPT) ベース

**構成**:
- **Encoder**: `SlidingPyramidNetwork` - マルチスケール特徴抽出
- **Decoder**: `MultiresConvDecoder` - 特徴の融合と高解像度化
- **Head**: Disparity予測のための畳み込み層

**forward処理** (92-98行):

```python
def forward(self, image):
    encodings = self.encoder(self.normalizer(image))
    num_encoder_features = len(self.encoder.dims_encoder)
    features = self.decoder(encodings[:num_encoder_features])
    disparity = self.head(features)
    return disparity
```

**出力**:
- `disparity`: 逆深度マップ (1チャンネルまたは2チャンネル)

### 5. Initializer

**ファイル**: [src/sharp/models/initializer.py](src/sharp/models/initializer.py:127-253)

**クラス**: `MultiLayerInitializer`

**役割**: Depthから基本的なGaussianパラメータを初期化

**forward処理**:

```python
def forward(self, image, depth):
    # Depthの正規化
    if self.normalize_depth:
        depth, depth_factor = _rescale_depth(depth)
        global_scale = 1.0 / depth_factor

    # Disparityレイヤーの作成
    disparity = create_disparity_layers(depth)  # Line 152-206

    # Base値の準備
    base_x_ndc, base_y_ndc = _create_base_xy(...)  # NDC座標
    base_scales = _create_base_scale(disparity, ...)
    base_quaternions = [1, 0, 0, 0]  # 回転なし
    base_opacities = min(1.0 / num_layers, 0.5)
    base_colors = 0.5 (または画像から)

    # Feature inputの準備
    features_in = prepare_feature_input(image, depth)

    return InitializerOutput(
        gaussian_base_values=GaussianBaseValues(...),
        feature_input=features_in,
        global_scale=global_scale,
    )
```

**出力**:
- `mean_x_ndc`, `mean_y_ndc`: NDC空間のXY座標
- `mean_inverse_z_ndc`: 逆深度 (disparity)
- `scales`: ガウシアンのスケール
- `quaternions`: 回転 (初期値は単位四元数)
- `colors`: 色 (初期値は0.5またはRGB画像から)
- `opacities`: 不透明度 (初期値は1/num_layers)

### 6. Gaussian Composer

**ファイル**: [src/sharp/models/composer.py](src/sharp/models/composer.py:92-155)

**クラス**: `GaussianComposer`

**役割**: Base値とDelta値を組み合わせて最終的なGaussiansを生成

**forward処理**:

```python
def forward(self, delta, base_values, global_scale, flatten_output=True):
    # Delta値のアップサンプリング (必要に応じて)
    if scale_factor != 1:
        delta = self.upsample_delta_value(delta, scale_factor)

    # Mean vectorsの計算
    mean_vectors = self._forward_mean(base_values, delta)

    # 各パラメータの活性化関数適用
    singular_values = self._scale_activation(base_scales, delta[:, 3:6], ...)
    quaternions = self._quaternion_activation(base_values.quaternions, delta[:, 6:10])
    colors = self._color_activation(base_values.colors, delta[:, 10:13])
    opacities = self._opacity_activation(base_values.opacities, delta[:, 13])

    # Flatten処理 (オプション)
    if flatten_output:
        # [B, C, N, H, W] -> [B, N*H*W, C]
        ...

    # Global scalingでメートル空間に変換
    if global_scale is not None:
        mean_vectors = global_scale * mean_vectors
        singular_values = global_scale * singular_values

    return Gaussians3D(
        mean_vectors, singular_values, quaternions, colors, opacities
    )
```

**Mean activation** (186-209行):

```python
def _mean_activation(self, base, learned_delta):
    # XY座標: 単純な加算
    xx = base[:, 0:1] + learned_delta[:, 0:1]
    yy = base[:, 1:2] + learned_delta[:, 1:2]

    # Z座標: Softplus activation
    a = base[:, 2:3]  # Base inverse depth
    b = learned_delta[:, 2:3]
    inverse_zz = F.softplus(inverse_softplus(a) + b)
    zz = 1.0 / (inverse_zz + 1e-3)

    # NDC座標からメートル座標に変換
    mean_vectors = torch.cat([zz * xx, zz * yy, zz], dim=1)
    return mean_vectors
```

### 7. Unprojection

**ファイル**: [src/sharp/utils/gaussians.py](src/sharp/utils/gaussians.py:89-98)

**関数**: `unproject_gaussians`

**役割**: NDC空間のGaussiansをメートル空間に変換

```python
def unproject_gaussians(gaussians_ndc, extrinsics, intrinsics, image_shape):
    unprojection_matrix = get_unprojection_matrix(
        extrinsics, intrinsics, image_shape
    )
    gaussians = apply_transform(gaussians_ndc, unprojection_matrix[:3])
    return gaussians
```

## Depth層数の仕様

### デフォルト設定

**1層のdepthで動作可能** (推奨)

[src/sharp/models/initializer.py](src/sharp/models/initializer.py:188-206):

```python
if self.num_layers == 1:
    disparity = first_disparity
else:  # Fill in the rest layers.
    following_depth = depth if depth.shape[1] == 1 else depth[:, 1:]
    if self.rest_layer_depth_option == "surface_min":
        following_disparity = _create_surface_layer(following_depth, "min")
    # ...
```

**動作モード**:

| `num_layers` | Depth入力形状 | 動作 |
|-------------|--------------|------|
| 1 | `(B, 1, H, W)` | 1層のGaussiansを生成 |
| 2 | `(B, 1, H, W)` | 1層目のdepthから2層のGaussiansを生成 |
| 2 | `(B, 2, H, W)` | 各層に対応するdepthから2層のGaussiansを生成 |

### Monodepth出力

[src/sharp/models/monodepth.py](src/sharp/models/monodepth.py:209-212):

```python
if self.num_monodepth_layers == 2 and self.sorting_monodepth:
    first_layer_disparity = disparity.max(dim=1, keepdims=True).values
    second_layer_disparity = disparity.min(dim=1, keepdims=True).values
    disparity = torch.cat([first_layer_disparity, second_layer_disparity], dim=1)
```

**結論**: 単一のdepth画像 `(1, 1, H, W)` を準備すれば十分

## カメラパラメータの推定

### 現在の実装

**ファイル**: [src/sharp/utils/io.py](src/sharp/utils/io.py:29-81)

**関数**: `load_rgb`

**焦点距離の取得フロー**:

```python
def load_rgb(path):
    # 画像とEXIF情報を読み込み
    img_pil = Image.open(path)
    img_exif = extract_exif(img_pil)

    # EXIFから焦点距離を取得
    f_35mm = img_exif.get("FocalLengthIn35mmFilm", ...)

    # 見つからない場合はデフォルト値
    if f_35mm is None or f_35mm < 1:
        LOGGER.warn("Did not find focallength in exif data - Setting to 30mm.")
        f_35mm = 30.0

    # ピクセル単位に変換
    f_px = convert_focallength(img.shape[1], img.shape[0], f_35mm)

    return img, icc_profile, f_px
```

### 焦点距離の変換

**ファイル**: [src/sharp/utils/io.py](src/sharp/utils/io.py:97-99)

```python
def convert_focallength(width, height, f_mm=30):
    """35mmフィルム換算の焦点距離をピクセル単位に変換"""
    return f_mm * np.sqrt(width**2 + height**2) / np.sqrt(36**2 + 24**2)
```

**計算式**:
```
f_px = f_35mm * diagonal_pixels / diagonal_35mm
     = f_35mm * sqrt(W² + H²) / sqrt(36² + 24²)
     ≈ f_35mm * sqrt(W² + H²) / 43.27
```

### Disparity Factorの計算

**ファイル**: [src/sharp/cli/predict.py](src/sharp/cli/predict.py:171)

```python
disparity_factor = torch.tensor([f_px / width]).float().to(device)
```

**使用箇所**: [src/sharp/models/predictor.py](src/sharp/models/predictor.py:131)

```python
monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)
```

### カメラ内部パラメータ

**ファイル**: [src/sharp/cli/predict.py](src/sharp/cli/predict.py:135-144)

```python
intrinsics = torch.tensor([
    [f_px, 0, (width - 1) / 2.0, 0],
    [0, f_px, (height - 1) / 2.0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], device=device, dtype=torch.float32)
```

**パラメータ**:
- `fx = fy = f_px`: 焦点距離 (ピクセル単位)
- `cx = (width - 1) / 2.0`: 主点のX座標
- `cy = (height - 1) / 2.0`: 主点のY座標

## 事前準備したデータで置き換える場合の実装方針

### オプション1: Monodepth出力を置き換え (推奨)

**注入ポイント**: [src/sharp/models/predictor.py](src/sharp/models/predictor.py:127-131)

**実装方法**:

```python
def forward(self, image, disparity_factor, depth=None, external_depth=None):
    if external_depth is not None:
        # 事前準備したdepthを使用
        monodepth = external_depth
        # Dummy output for compatibility
        monodepth_output = MonodepthOutput(
            disparity=disparity_factor / monodepth.clamp(min=1e-4, max=1e4),
            encoder_features=[],
            decoder_features=None,
            output_features=[],
        )
    else:
        # 既存のMonodepthモデルを使用
        monodepth_output = self.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity
        disparity_factor = disparity_factor[:, None, None, None]
        monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)

    # 以下は既存の処理
    ...
```

**利点**:
- Monodepthモデルの推論をスキップできる
- 後段の処理(initializer, composer)はそのまま利用可能
- 最小限の変更で実装可能

**課題**:
- `monodepth_output.output_features`が空になるため、feature modelでエラーが発生する可能性
- 対策: Dummy featuresを生成、またはfeature modelをスキップするモードを追加

### オプション2: predict_image関数でdepthを直接渡す

**注入ポイント**: [src/sharp/cli/predict.py](src/sharp/cli/predict.py:159-164)

**実装方法**:

```python
def predict_image(predictor, image, f_px, device, external_depth=None):
    # ...

    if external_depth is not None:
        # 事前準備したdepthをリサイズ
        external_depth_resized = F.interpolate(
            external_depth[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )
        # Ground truthとして渡す
        gaussians_ndc = predictor(
            image_resized_pt, disparity_factor, depth=external_depth_resized
        )
    else:
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    # ...
```

**利点**:
- より上位レベルでの制御
- Depth alignmentの仕組みを活用できる可能性

**課題**:
- 現在の実装では`depth`パラメータはtraining時のground truthとして使用
- Alignmentがある場合、予期しない変換が適用される可能性

### オプション3: CLIレベルで外部depth入力をサポート (最もクリーン)

**注入ポイント**: [src/sharp/cli/predict.py](src/sharp/cli/predict.py:76-83)

**実装方法**:

```python
@click.option(
    "--depth-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to pre-computed depth maps (optional).",
    required=False,
)
@click.option(
    "--focal-length",
    type=float,
    default=None,
    help="Focal length in pixels (optional, overrides EXIF).",
    required=False,
)
def predict_cli(
    input_path, output_path, checkpoint_path,
    with_rendering, device, verbose,
    depth_path=None, focal_length=None
):
    # ...

    for image_path in image_paths:
        image, _, f_px = io.load_rgb(image_path)

        # 焦点距離の上書き
        if focal_length is not None:
            f_px = focal_length

        # 事前準備したdepthの読み込み
        external_depth = None
        if depth_path is not None:
            depth_file = depth_path / f"{image_path.stem}.npy"  # or .png, .exr
            if depth_file.exists():
                external_depth = load_depth(depth_file)

        gaussians = predict_image(
            gaussian_predictor, image, f_px,
            torch.device(device), external_depth
        )
        # ...
```

**利点**:
- コードの変更が最小限
- 既存の機能との互換性を保ちやすい
- 実験しやすい

**課題**:
- Depth画像のフォーマット定義が必要 (.npy, .png, .exr など)
- ファイル命名規則の決定が必要

### オプション4: 専用のpredictorクラスを作成 (最も柔軟)

**実装方法**:

```python
class RGBDGaussianPredictor(nn.Module):
    """RGB画像と事前計算されたDepthからGaussiansを予測"""

    def __init__(self, init_model, feature_model, prediction_head,
                 gaussian_composer):
        super().__init__()
        self.init_model = init_model
        self.feature_model = feature_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer

    def forward(self, image, depth, disparity_factor=None):
        # Monodepthモデルをスキップ

        # Initializerで基本的なGaussianパラメータを生成
        init_output = self.init_model(image, depth)

        # Feature modelでGaussian特徴量を計算
        # (monodepth featuresなしのモード)
        image_features = self.feature_model(
            init_output.feature_input,
            encodings=None  # または画像から直接特徴抽出
        )

        # Delta値を予測
        delta_values = self.prediction_head(image_features)

        # 最終的なGaussiansを合成
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )

        return gaussians
```

**利点**:
- 最も柔軟な実装
- Monodepthモデルの依存を完全に排除
- 新しいアーキテクチャの実験が容易

**課題**:
- 実装コストが高い
- Feature modelがmonodepth encodingsに依存している場合、再設計が必要

## 必要な入力フォーマット

### RGB画像

**既存のinput/imagesと同じフォーマット**:
- 対応拡張子: `.jpg`, `.png`, `.heic`, `.tiff` など
- EXIF情報があれば焦点距離を自動取得

### Depth画像 (メートル単位)

**推奨フォーマット**:

#### NumPy形式 (.npy)
```python
depth = np.load("depth.npy")  # Shape: (H, W) or (1, H, W)
```

**特徴**:
- 浮動小数点精度を完全に保持
- 最も簡単に扱える

#### PNG形式 (16bit)
```python
depth_png = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH)
depth = depth_png.astype(np.float32) * depth_scale
```

**特徴**:
- 可視化しやすい
- 16bitで十分な精度

#### OpenEXR形式 (.exr)
```python
import OpenEXR
# ...
```

**特徴**:
- HDR対応
- 広範囲のdepth値を扱える

### Depth画像の仕様

**必須要件**:
- **形状**: `(H, W)` または `(1, H, W)` (バッチ次元なし)
- **値の範囲**: メートル単位の実depth値
  - 例: 0.5m ~ 100m
- **データ型**: `float32` または `float64`

**オプション (2層の場合)**:
- **形状**: `(2, H, W)`
- **値**: 1層目 = 前景depth、2層目 = 背景depth

### カメラパラメータ

**オプション1: EXIF情報を利用** (推奨)
- RGB画像にEXIF情報があれば自動的に焦点距離が取得される
- 何も変更不要

**オプション2: 明示的に指定**

コマンドライン引数:
```bash
sharp predict -i input/images -o output/gaussians \
  --depth-path input/depths \
  --focal-length 512.0
```

または、JSONファイル:
```json
{
  "image_001.jpg": {
    "focal_length_px": 512.0,
    "principal_point": [319.5, 239.5]
  }
}
```

## 推奨実装アプローチ

### フェーズ1: CLIオプションの追加 (最優先)

1. `--depth-path`オプションを追加
2. `--focal-length`オプションを追加
3. Depth画像の読み込み処理を実装
4. `predict_image`関数を拡張してexternal depthを受け取る

**実装見積もり**: 1-2日

### フェーズ2: Predictorの拡張

1. `RGBGaussianPredictor`に`external_depth`モードを追加
2. Monodepth encodingsなしでfeature modelが動作するように対応
   - オプションA: Dummy featuresを生成
   - オプションB: 画像から直接特徴を抽出する別経路を追加

**実装見積もり**: 2-3日

### フェーズ3: テストと検証

1. サンプルRGB+Depthデータで動作確認
2. 既存のMonodepth版との比較
3. エッジケースのテスト (異なる解像度、異なるカメラパラメータ)

**実装見積もり**: 1-2日

## 参考情報

### 関連ファイル一覧

- **エントリポイント**:
  - [run.bat](run.bat)
  - [src/sharp/cli/__init__.py](src/sharp/cli/__init__.py)
  - [src/sharp/cli/predict.py](src/sharp/cli/predict.py)

- **モデル定義**:
  - [src/sharp/models/predictor.py](src/sharp/models/predictor.py)
  - [src/sharp/models/monodepth.py](src/sharp/models/monodepth.py)
  - [src/sharp/models/initializer.py](src/sharp/models/initializer.py)
  - [src/sharp/models/composer.py](src/sharp/models/composer.py)

- **ユーティリティ**:
  - [src/sharp/utils/gaussians.py](src/sharp/utils/gaussians.py)
  - [src/sharp/utils/io.py](src/sharp/utils/io.py)
  - [src/sharp/utils/camera.py](src/sharp/utils/camera.py)

### キーとなるデータ構造

```python
# Monodepth出力
class MonodepthOutput(NamedTuple):
    disparity: torch.Tensor              # (B, C, H, W), C=1 or 2
    encoder_features: list[torch.Tensor] # Multi-scale features
    decoder_features: torch.Tensor       # Single-level feature
    output_features: list[torch.Tensor]  # Features for Gaussian predictor
    intermediate_features: list[torch.Tensor] = []

# Gaussian base values
class GaussianBaseValues(NamedTuple):
    mean_x_ndc: torch.Tensor        # (B, 1, N, H, W)
    mean_y_ndc: torch.Tensor        # (B, 1, N, H, W)
    mean_inverse_z_ndc: torch.Tensor # (B, 1, N, H, W)
    scales: torch.Tensor            # (B, 3, N, H, W)
    quaternions: torch.Tensor       # (B, 4, N, H, W)
    colors: torch.Tensor            # (B, 3, N, H, W)
    opacities: torch.Tensor         # Scalar

# 3D Gaussians
class Gaussians3D(NamedTuple):
    mean_vectors: torch.Tensor      # (B, N, 3)
    singular_values: torch.Tensor   # (B, N, 3)
    quaternions: torch.Tensor       # (B, N, 4)
    colors: torch.Tensor            # (B, N, 3)
    opacities: torch.Tensor         # (B, N)
```

## まとめ

- **現在の実装**: RGBのみから Monodepth → Gaussians
- **目標**: RGB + 事前準備したDepth → Gaussians
- **推奨アプローチ**: CLIオプションの追加 + Predictorの拡張
- **必要な入力**:
  - RGB画像 (既存と同じ)
  - Depth画像 (メートル単位、形状 `(1, H, W)`)
  - カメラパラメータ (EXIFまたは明示的指定)

**次のステップ**: 実装方針の決定と開発着手
