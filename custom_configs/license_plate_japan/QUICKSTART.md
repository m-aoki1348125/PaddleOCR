# 日本のナンバープレートOCR - クイックスタートガイド

事前学習モデル + 前後処理を使用した、ファインチューニング不要のナンバープレート認識システムです。

---

## 🚀 3ステップで開始

### ステップ1: 環境セットアップ

```bash
# PaddlePaddleのインストール（GPU版推奨）
pip install paddlepaddle-gpu

# CPUのみの場合
# pip install paddlepaddle

# PaddleOCRのインストール
pip install paddleocr

# その他の依存関係
pip install opencv-python numpy
```

### ステップ2: 事前学習モデルのダウンロード（オプション）

PaddleOCRは初回実行時に自動的にモデルをダウンロードしますが、事前にダウンロードすることも可能です：

```bash
cd /path/to/PaddleOCR

# モデルのダウンロード
bash custom_configs/license_plate_japan/download_models.sh
```

### ステップ3: ナンバープレート認識を実行

```bash
# 単一画像の認識
python custom_configs/license_plate_japan/run_ocr.py /path/to/plate.jpg

# ディレクトリ内の全画像を一括処理
python custom_configs/license_plate_japan/run_ocr.py /path/to/images/ --output_csv results.csv
```

**完了！** 🎉

---

## 📖 基本的な使い方

### 単一画像の認識

```bash
python custom_configs/license_plate_japan/run_ocr.py plate.jpg
```

**出力例:**
```
==================================================
🚗 日本のナンバープレートOCR
==================================================

PaddleOCR Root: /home/user/PaddleOCR
✓ 前処理を有効化
✓ 後処理を有効化
PaddleOCRを初期化中...
✓ PaddleOCR初期化完了

📷 画像を認識中: plate.jpg

  ✓ 前処理完了
  ✓ OCR実行完了: 品川 330 あ 12-34 (信頼度: 95.30%)
  ✓ 後処理完了: 有効性=True

==================================================
📋 認識結果
==================================================
✅ 認識成功
  ナンバープレート: 品川 330 あ 12-34
  信頼度: 95.30%
  地域: 品川
  分類番号: 330
  ひらがな: あ
  車両番号: 12-34
==================================================
```

### 複数画像の一括処理

```bash
python custom_configs/license_plate_japan/run_ocr.py images/ --output_csv results.csv
```

**出力CSV (`results.csv`):**
```csv
image_path,success,text,confidence,region,classification,hiragana,number,error
images/plate_001.jpg,True,品川 330 あ 12-34,0.953,品川,330,あ,12-34,
images/plate_002.jpg,True,横浜 500 さ 56-78,0.887,横浜,500,さ,56-78,
images/plate_003.jpg,False,,,0.0,,,,テキストが検出されませんでした
```

### JSON形式で出力

```bash
python custom_configs/license_plate_japan/run_ocr.py images/ --output_json results.json
```

---

## ⚙️ オプション

### 前処理・後処理の制御

```bash
# 前処理を無効化（超解像処理済み画像に対して）
python custom_configs/license_plate_japan/run_ocr.py plate.jpg --no_preprocessing

# 後処理を無効化（生のOCR結果が必要な場合）
python custom_configs/license_plate_japan/run_ocr.py plate.jpg --no_postprocessing

# 両方を無効化
python custom_configs/license_plate_japan/run_ocr.py plate.jpg --no_preprocessing --no_postprocessing
```

### GPU/CPU の選択

```bash
# GPU使用（デフォルト）
python custom_configs/license_plate_japan/run_ocr.py plate.jpg

# CPUを使用
python custom_configs/license_plate_japan/run_ocr.py plate.jpg --cpu
```

### 詳細ログの制御

```bash
# 詳細ログを表示（デフォルト）
python custom_configs/license_plate_japan/run_ocr.py plate.jpg

# 簡潔な出力のみ
python custom_configs/license_plate_japan/run_ocr.py plate.jpg --quiet
```

---

## 🔧 システム構成

### アーキテクチャ

```
入力画像
   ↓
[前処理]
 - ノイズ除去（非局所平均デノイジング）
 - コントラスト強調（CLAHE）
 - シャープニング（Unsharp Masking）
 - リサイズ（320x48、アスペクト比維持）
   ↓
[PaddleOCR]
 - 検出モデル: PP-OCRv3（日本語）
 - 認識モデル: PP-OCRv3（日本語）
 - カスタム辞書: ナンバープレート専用（約200文字）
   ↓
[後処理]
 - フォーマット解析（地域・分類番号・ひらがな・車両番号）
 - 有効性検証
 - 誤認識補正（O→0、I→1など）
   ↓
認識結果
```

### カスタム辞書

`license_plate_dict.txt` には以下の文字が含まれます：

- **数字**: 0-9
- **ひらがな**: あ、い、う、え、か、き... （お、し、へ、んを除く）
- **地名用漢字**: 品川、横浜、大阪、名古屋、札幌など

限定的な文字セットにより、認識精度が向上します。

---

## 📊 精度について

### 推奨される入力画像

- **解像度**: 300x150ピクセル以上
- **フォーマット**: JPG、PNG、BMP、TIFF
- **品質**: 超解像処理済みの高品質画像
- **角度**: 正面からの撮影（±10度以内）
- **照明**: 均一な照明、強い影がない

### 精度向上のヒント

1. **超解像処理**: 入力画像に専用の超解像処理を適用済みであることが前提
2. **前処理の調整**: `preprocessing.py` のパラメータを画像特性に応じて調整
3. **後処理の活用**: 自動補正により、一般的な誤認識を修正
4. **複数回実行**: 異なる前処理パラメータで複数回実行し、結果を比較

### 期待される精度

- **高品質画像**: 90%以上の認識率
- **中品質画像**: 70-90%の認識率
- **低品質画像**: 50-70%の認識率

**注**: さらに高精度が必要な場合は、ファインチューニングを検討してください（`README.md` 参照）。

---

## 🔍 トラブルシューティング

### Q1: モデルのダウンロードが遅い

**対策:**
```bash
# 事前にモデルをダウンロード
bash custom_configs/license_plate_japan/download_models.sh
```

### Q2: GPU が認識されない

**対策:**
```bash
# CUDA が正しくインストールされているか確認
nvidia-smi

# PaddlePaddle GPU版が正しくインストールされているか確認
python -c "import paddle; print(paddle.device.get_device())"

# CPU で実行
python custom_configs/license_plate_japan/run_ocr.py plate.jpg --cpu
```

### Q3: 認識精度が低い

**対策:**

1. 画像品質を確認
   ```bash
   # 前処理の効果を確認
   python custom_configs/license_plate_japan/preprocessing.py input.jpg output.jpg
   ```

2. 前処理パラメータを調整（`preprocessing.py` を編集）

3. より多くのテストを実行
   ```bash
   python custom_configs/license_plate_japan/run_ocr.py images/ --output_csv results.csv
   ```

4. ファインチューニングを検討（`README.md` 参照）

### Q4: 特定の文字が誤認識される

**対策:**

後処理で自動補正されますが、さらに補正ルールを追加する場合は `postprocessing.py` を編集：

```python
def _clean_text(self, text: str) -> str:
    # 既存の補正に加えて、カスタムルールを追加
    text = text.replace('0', 'O')  # 例: 0をOに
    return text
```

---

## 📚 詳細ドキュメント

- **完全ガイド**: `README.md` - セットアップからファインチューニングまでの詳細
- **前処理**: `preprocessing.py` - 画像前処理の実装
- **後処理**: `postprocessing.py` - 結果検証と補正の実装
- **推論**: `run_ocr.py` - エンドツーエンド推論スクリプト

---

## 💡 使用例

### 例1: パトカー車載カメラからの画像処理

```bash
# 1. 超解像処理済み画像ディレクトリを準備
# 2. 一括処理を実行
python custom_configs/license_plate_japan/run_ocr.py \
    /path/to/patrol_car_images/ \
    --output_csv patrol_results.csv \
    --output_json patrol_results.json

# 3. 結果を確認
cat patrol_results.csv
```

### 例2: リアルタイム処理（ループ実行）

```bash
#!/bin/bash
# リアルタイムで新しい画像を処理

WATCH_DIR="/path/to/watch"
OUTPUT_DIR="/path/to/output"

while true; do
    for img in ${WATCH_DIR}/*.jpg; do
        if [ -f "$img" ]; then
            python custom_configs/license_plate_japan/run_ocr.py "$img" \
                --output_json "${OUTPUT_DIR}/$(basename $img .jpg).json"
            mv "$img" "${OUTPUT_DIR}/"
        fi
    done
    sleep 5
done
```

### 例3: 前処理のみを実行（デバッグ用）

```bash
# 前処理の効果を視覚的に確認
python custom_configs/license_plate_japan/preprocessing.py \
    input.jpg \
    output_preprocessed.jpg
```

---

## 🎯 次のステップ

### より高精度が必要な場合

1. **データセットを準備** (500枚以上のナンバープレート画像)
2. **ファインチューニングを実行** (`README.md` の手順に従う)
3. **カスタムモデルで推論**
   ```bash
   python custom_configs/license_plate_japan/run_ocr.py plate.jpg \
       --rec_model_dir ./output/license_plate_japan_rec/best_accuracy
   ```

### カスタマイズ

- **辞書の編集**: `license_plate_dict.txt` に文字を追加/削除
- **前処理の調整**: `preprocessing.py` のパラメータを変更
- **後処理のルール追加**: `postprocessing.py` に検証・補正ロジックを追加

---

## ❓ サポート

問題が発生した場合は、以下を確認してください：

1. **ログの確認**: エラーメッセージを確認
2. **環境の確認**: Python、PaddlePaddle、PaddleOCRのバージョン
3. **ドキュメント**: `README.md` で詳細情報を確認

---

## 📝 ライセンス

このプロジェクトは PaddleOCR のライセンスに従います（Apache License 2.0）。

---

**🚗 日本のナンバープレートOCR - すぐに使える高精度認識システム**
