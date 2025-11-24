#!/bin/bash
# 全てのサンプル画像をテストするスクリプト

echo "=========================================="
echo "サンプル画像のOCRテスト"
echo "=========================================="
echo ""

SAMPLE_DIR="custom_configs/license_plate_japan/sample_images"

# 超解像処理後の画像をテスト
echo "【超解像処理後の画像】"
echo "------------------------------------------"

for img in ${SAMPLE_DIR}/*_super_res.jpg; do
    if [ -f "$img" ]; then
        echo "テスト: $(basename $img)"
        python custom_configs/license_plate_japan/simple_test.py "$img" 2>&1 | grep -E "(結合テキスト|平均信頼度)"
        echo ""
    fi
done

echo ""
echo "【低解像度画像】"
echo "------------------------------------------"

# 低解像度画像をテスト
for img in ${SAMPLE_DIR}/*_low_res.jpg; do
    if [ -f "$img" ]; then
        echo "テスト: $(basename $img)"
        python custom_configs/license_plate_japan/simple_test.py "$img" 2>&1 | grep -E "(結合テキスト|平均信頼度|検出されませんでした)"
        echo ""
    fi
done

echo "=========================================="
echo "テスト完了"
echo "=========================================="
