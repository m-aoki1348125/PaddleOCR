#!/bin/bash
# 日本語PP-OCRv3事前学習モデルのダウンロードスクリプト

set -e

echo "=========================================="
echo "日本語PP-OCRv3モデルのダウンロード"
echo "=========================================="
echo ""

# モデル保存ディレクトリの作成
MODEL_DIR="./pretrained_models"
mkdir -p ${MODEL_DIR}

cd ${MODEL_DIR}

# 検出モデル（軽量版）のダウンロード
echo "📥 テキスト検出モデルをダウンロード中..."
if [ ! -f "ch_PP-OCRv3_det_infer.tar" ]; then
    wget -q --show-progress https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
    echo "✅ 検出モデルのダウンロード完了"
else
    echo "ℹ️  検出モデルは既にダウンロード済みです"
fi

# アーカイブを展開
if [ ! -d "ch_PP-OCRv3_det_infer" ]; then
    echo "📦 検出モデルを展開中..."
    tar -xf ch_PP-OCRv3_det_infer.tar
    echo "✅ 検出モデルの展開完了"
fi

# 日本語認識モデル（推論用）のダウンロード
echo ""
echo "📥 日本語認識モデルをダウンロード中..."
if [ ! -f "japan_PP-OCRv3_rec_infer.tar" ]; then
    wget -q --show-progress https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar
    echo "✅ 認識モデルのダウンロード完了"
else
    echo "ℹ️  認識モデルは既にダウンロード済みです"
fi

# アーカイブを展開
if [ ! -d "japan_PP-OCRv3_rec_infer" ]; then
    echo "📦 認識モデルを展開中..."
    tar -xf japan_PP-OCRv3_rec_infer.tar
    echo "✅ 認識モデルの展開完了"
fi

cd ..

echo ""
echo "=========================================="
echo "✅ すべてのモデルのダウンロードが完了しました"
echo "=========================================="
echo ""
echo "ダウンロードされたモデル:"
echo "  - 検出モデル: ${MODEL_DIR}/ch_PP-OCRv3_det_infer/"
echo "  - 認識モデル: ${MODEL_DIR}/japan_PP-OCRv3_rec_infer/"
echo ""
echo "次のステップ:"
echo "  python custom_configs/license_plate_japan/run_ocr.py <画像パス>"
echo ""
