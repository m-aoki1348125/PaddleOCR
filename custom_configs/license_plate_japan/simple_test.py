#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプルなナンバープレートOCRテストスクリプト
"""

import sys
from pathlib import Path

# PaddleOCRのパスを追加
PADDLEOCR_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PADDLEOCR_ROOT))

import cv2
from paddleocr import PaddleOCR

def main():
    if len(sys.argv) < 2:
        print("使用方法: python simple_test.py <画像パス>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("\n" + "="*60)
    print("シンプルなナンバープレートOCRテスト")
    print("="*60 + "\n")

    # PaddleOCRの初期化
    print("PaddleOCRを初期化中...")
    ocr = PaddleOCR(lang='japan', device='cpu')
    print("✓ 初期化完了\n")

    # 画像の読み込み
    print(f"画像を読み込み中: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ エラー: 画像が読み込めません")
        sys.exit(1)
    print(f"✓ 画像サイズ: {image.shape[1]}x{image.shape[0]}\n")

    # OCR実行
    print("OCRを実行中...")
    try:
        results = ocr.predict(image)
        print("✓ OCR完了\n")
    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)

    # 結果の表示
    print("="*60)
    print("認識結果")
    print("="*60)

    if not results or len(results) == 0:
        print("テキストが検出されませんでした")
        sys.exit(1)

    result = results[0]
    if 'rec_texts' in result:
        texts = result['rec_texts']
        scores = result['rec_scores']

        print(f"\n検出されたテキスト数: {len(texts)}\n")

        for i, (text, score) in enumerate(zip(texts, scores), 1):
            print(f"{i}. {text} (信頼度: {score:.2%})")

        # 全テキストを結合
        full_text = ' '.join(texts)
        avg_score = sum(scores) / len(scores)

        print(f"\n結合テキスト: {full_text}")
        print(f"平均信頼度: {avg_score:.2%}")
    else:
        print("結果からテキストが見つかりませんでした")
        print(f"結果の構造: {list(result.keys())}")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
