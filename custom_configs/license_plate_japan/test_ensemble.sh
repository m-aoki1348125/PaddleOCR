#!/bin/bash
# アンサンブルOCRのテストスクリプト

echo "=========================================="
echo "アンサンブルOCR テスト"
echo "=========================================="
echo ""

SAMPLE_DIR="custom_configs/license_plate_japan/sample_images"
RESULTS_FILE="ensemble_test_results.txt"

# 結果ファイルを初期化
> "$RESULTS_FILE"

echo "【超解像処理後の画像】"
echo "------------------------------------------"

# 超解像処理後の画像をテスト
for img in ${SAMPLE_DIR}/*_super_res.jpg; do
    if [ -f "$img" ]; then
        echo "テスト: $(basename $img)"
        echo "=== $(basename $img) ===" >> "$RESULTS_FILE"

        python custom_configs/license_plate_japan/ensemble_ocr.py "$img" --cpu --quiet 2>&1 | \
            grep -E "(最良戦略|ナンバープレート|信頼度|地域|分類番号|ひらがな|車両番号)" | \
            tee -a "$RESULTS_FILE"

        echo "" | tee -a "$RESULTS_FILE"
    fi
done

echo ""
echo "=========================================="
echo "テスト完了"
echo "結果を $RESULTS_FILE に保存しました"
echo "=========================================="
