# 日本のナンバープレートOCR実装ガイド

PaddleOCRを使用して、パトカー車載カメラから取得した日本のナンバープレート画像を高精度に認識するための実装ガイドです。

## 目次

1. [概要](#概要)
2. [システム要件](#システム要件)
3. [セットアップ](#セットアップ)
4. [使用方法](#使用方法)
5. [ファインチューニング](#ファインチューニング)
6. [精度向上のヒント](#精度向上のヒント)
7. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### 対象画像の特徴
- パトカー車載カメラからの低解像度車両画像
- ナンバープレート部分を切り出し済み
- 専用の超解像処理を適用済み

### アプローチ
1. **カスタム辞書**: ナンバープレートで使用される文字のみに限定
2. **前処理**: 超解像処理後の画像を最適化
3. **ファインチューニング**: 日本語PP-OCRv3モデルをベースに学習
4. **後処理**: ナンバープレートフォーマットに基づく検証・補正

---

## システム要件

### ハードウェア
- **推奨**: GPU搭載マシン（NVIDIA GPU + CUDA）
- **最小**: CPU（推論速度は遅くなります）

### ソフトウェア
- Python 3.8～3.12
- PaddlePaddle 3.0以降
- PaddleOCR
- OpenCV

---

## セットアップ

### 1. PaddlePaddleのインストール

```bash
# CPU版
pip install paddlepaddle

# GPU版（推奨）
pip install paddlepaddle-gpu
```

### 2. PaddleOCRのインストール

```bash
pip install paddleocr
```

### 3. 必要なライブラリのインストール

```bash
pip install opencv-python numpy
```

### 4. ファイル構成の確認

```
PaddleOCR/
└── custom_configs/
    └── license_plate_japan/
        ├── README.md                    # このファイル
        ├── license_plate_dict.txt       # カスタム辞書
        ├── license_plate_rec.yml        # 学習設定ファイル
        ├── preprocessing.py             # 前処理スクリプト
        ├── postprocessing.py            # 後処理スクリプト
        └── inference.py                 # 推論スクリプト
```

---

## 使用方法

### クイックスタート（事前学習モデルを使用）

事前学習済みの日本語モデルを使って、すぐに推論を試すことができます。

```bash
# 単一画像の認識
python custom_configs/license_plate_japan/inference.py \
    /path/to/license_plate_image.jpg

# ディレクトリ内の全画像を一括処理
python custom_configs/license_plate_japan/inference.py \
    /path/to/images/ \
    --output_csv results.csv
```

### 推論オプション

```bash
python custom_configs/license_plate_japan/inference.py \
    /path/to/image.jpg \
    --rec_model_dir ./output/license_plate_japan_rec/best_accuracy \  # カスタムモデル
    --rec_char_dict_path custom_configs/license_plate_japan/license_plate_dict.txt \
    --min_confidence 0.7 \  # 最小信頼度スコア
    --no_preprocessing \    # 前処理を無効化
    --no_postprocessing \   # 後処理を無効化
    --no_gpu                # CPUを使用
```

### 前処理のみを実行

```bash
python custom_configs/license_plate_japan/preprocessing.py \
    input_image.jpg \
    output_image.jpg
```

### 後処理のテスト

```bash
# 後処理スクリプトを直接実行してテストケースを確認
python custom_configs/license_plate_japan/postprocessing.py
```

---

## ファインチューニング

### 1. データセットの準備

#### データセット構造

```
train_data/
└── license_plate/
    ├── images/
    │   ├── plate_0001.jpg
    │   ├── plate_0002.jpg
    │   └── ...
    ├── train_list.txt
    └── val_list.txt
```

#### ラベルファイルの形式

`train_list.txt`および`val_list.txt`の各行は以下の形式：

```
images/plate_0001.jpg	品川 330 あ 12-34
images/plate_0002.jpg	横浜 500 さ 56-78
images/plate_0003.jpg	大阪 300 ま 90-12
```

フォーマット: `画像パス<TAB>ナンバープレートテキスト`

#### データセット作成のヒント

- **最小データ量**: 500枚以上を推奨
- **推奨データ量**: 5,000枚以上で高精度を実現
- **多様性**: 様々な照明条件、角度、距離の画像を含める
- **バランス**: 地域名、分類番号、ひらがなが偏らないように

### 2. 事前学習モデルのダウンロード

```bash
# 日本語PP-OCRv3モデルをダウンロード
mkdir -p pretrain_models
cd pretrain_models

# 事前学習モデル
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar
tar -xf japan_PP-OCRv3_rec_train.tar

cd ..
```

### 3. 学習の実行

#### 単一GPU学習

```bash
python tools/train.py \
    -c custom_configs/license_plate_japan/license_plate_rec.yml \
    -o Global.pretrained_model=./pretrain_models/japan_PP-OCRv3_rec_train/best_accuracy
```

#### 複数GPU学習（推奨）

```bash
python -m paddle.distributed.launch --gpus '0,1,2,3' \
    tools/train.py \
    -c custom_configs/license_plate_japan/license_plate_rec.yml \
    -o Global.pretrained_model=./pretrain_models/japan_PP-OCRv3_rec_train/best_accuracy
```

#### 学習率の調整

GPU数やバッチサイズに応じて学習率を調整してください：

- **8-GPU, batch_size=1024**: learning_rate=0.001
- **単一GPU, batch_size=128**: learning_rate=0.0001～0.0002
- **単一GPU, batch_size=64**: learning_rate=0.00005～0.0001

設定ファイルで調整：

```yaml
Optimizer:
  lr:
    learning_rate: 0.0001  # 環境に応じて調整
```

### 4. 学習の監視

```bash
# ログファイルで損失と精度を確認
tail -f output/license_plate_japan_rec/train.log

# チェックポイントの確認
ls output/license_plate_japan_rec/
```

### 5. モデルの評価

```bash
python tools/eval.py \
    -c custom_configs/license_plate_japan/license_plate_rec.yml \
    -o Global.checkpoints=./output/license_plate_japan_rec/best_accuracy
```

### 6. 推論モデルのエクスポート

```bash
python tools/export_model.py \
    -c custom_configs/license_plate_japan/license_plate_rec.yml \
    -o Global.pretrained_model=./output/license_plate_japan_rec/best_accuracy \
       Global.save_inference_dir=./inference/license_plate_rec/
```

---

## 精度向上のヒント

### 1. データ品質の向上

- **高品質なアノテーション**: 正確なラベル付けが最重要
- **データクリーニング**: ノイズや誤ラベルを除去
- **データ拡張**: 既存データに変換を適用して増量

### 2. 前処理のカスタマイズ

`preprocessing.py`のパラメータを調整：

```python
preprocessor = LicensePlatePreprocessor(
    target_height=48,
    target_width=320,
    apply_denoising=True,      # ノイズ除去
    apply_sharpening=True,     # シャープニング
    apply_contrast_enhancement=True  # コントラスト強調
)
```

### 3. ハイパーパラメータのチューニング

`license_plate_rec.yml`で調整：

```yaml
# 学習率の調整
Optimizer:
  lr:
    learning_rate: 0.0001  # 低めから始める

# エポック数の増加
Global:
  epoch_num: 500  # 精度が向上するまで学習

# バッチサイズの調整
Train:
  loader:
    batch_size_per_card: 128  # GPUメモリに応じて調整
```

### 4. 後処理の最適化

`postprocessing.py`の信頼度閾値を調整：

```python
postprocessor = LicensePlatePostprocessor(
    min_confidence=0.7,  # より厳しい閾値
    enable_correction=True
)
```

### 5. アンサンブル手法

複数のモデルの結果を組み合わせて精度向上：

- 異なる学習データで学習した複数モデル
- 異なるハイパーパラメータのモデル
- 投票またはスコア平均で最終結果を決定

### 6. テストタイム拡張（TTA）

推論時に複数の変換を適用して結果を統合：

```python
# 例: 明るさ調整、コントラスト調整などを適用
results = []
for transform in transforms:
    augmented_image = transform(image)
    result = ocr.recognize(augmented_image)
    results.append(result)

# 最も信頼度の高い結果を選択
best_result = max(results, key=lambda x: x['confidence'])
```

---

## トラブルシューティング

### Q1: 認識精度が低い

**対策:**
1. データセットの品質を確認（正確なラベル付け）
2. データ量を増やす（最低500枚、推奨5,000枚以上）
3. 前処理パラメータを調整
4. 学習エポック数を増やす
5. 学習率を下げる

### Q2: 学習が収束しない

**対策:**
1. 学習率を下げる（例: 0.0001 → 0.00005）
2. バッチサイズを調整
3. 事前学習モデルから開始しているか確認
4. データセットのバランスを確認

### Q3: GPU メモリ不足

**対策:**
1. バッチサイズを減らす
2. 画像サイズを小さくする
3. グラディエント累積を使用

```yaml
Train:
  loader:
    batch_size_per_card: 64  # 128から64に減らす
```

### Q4: 特定の文字が誤認識される

**対策:**
1. その文字を多く含むデータを追加
2. 後処理で補正ルールを追加
3. 辞書を確認して該当文字が含まれているか確認

### Q5: 推論速度が遅い

**対策:**
1. GPUを使用（`--no_gpu`オプションを外す）
2. 軽量モデルを使用（PP-OCRv3 mobile版）
3. 前処理を簡略化
4. 画像サイズを小さくする

---

## 高度な使用例

### カスタム前処理パイプラインの追加

`preprocessing.py`を編集して独自の処理を追加：

```python
def _custom_processing(self, image: np.ndarray) -> np.ndarray:
    """カスタム処理"""
    # 独自の画像処理ロジック
    processed = ...
    return processed
```

### バッチ推論の並列化

複数GPUで並列推論：

```python
from multiprocessing import Pool

def process_batch(image_paths):
    ocr = LicensePlateOCR()
    return [ocr.recognize(path) for path in image_paths]

# 画像リストを分割して並列処理
with Pool(processes=4) as pool:
    results = pool.map(process_batch, batched_image_paths)
```

---

## 参考資料

### PaddleOCR公式ドキュメント
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [ファインチューニングガイド](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/finetune.md)
- [多言語モデル](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages.md)

### 日本のナンバープレートについて
- [国土交通省 - 自動車登録番号標](https://www.mlit.go.jp/jidosha/jidosha_fr1_000034.html)

---

## ライセンス

このプロジェクトはPaddleOCRのライセンスに従います（Apache License 2.0）。

---

## お問い合わせ

問題や質問がある場合は、PaddleOCRの公式GitHubリポジトリでIssueを作成してください。

---

## 更新履歴

- **2025-11-24**: 初版リリース
  - カスタム辞書、学習設定、前処理・後処理スクリプト、推論スクリプトを実装
