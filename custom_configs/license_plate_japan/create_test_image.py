#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テスト用のナンバープレート風画像を生成するスクリプト
"""

import cv2
import numpy as np
from pathlib import Path


def create_test_license_plate(
    text: str,
    output_path: str,
    width: int = 440,
    height: int = 220,
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (0, 0, 0)
):
    """
    テスト用のナンバープレート風画像を作成

    Args:
        text: 表示するテキスト（例: "品川 330 あ 12-34"）
        output_path: 出力画像のパス
        width: 画像の幅
        height: 画像の高さ
        bg_color: 背景色 (B, G, R)
        text_color: テキスト色 (B, G, R)
    """
    # 白い背景を作成
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # テキストを複数行に分割して描画
    lines = text.split('\n') if '\n' in text else [text]

    # フォント設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    # 各行を描画
    y_offset = height // 2 - (len(lines) - 1) * 30

    for i, line in enumerate(lines):
        # テキストサイズを取得
        (text_width, text_height), baseline = cv2.getTextSize(
            line, font, font_scale, thickness
        )

        # 中央に配置
        x = (width - text_width) // 2
        y = y_offset + i * 60

        # テキストを描画
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

    # 枠線を追加
    cv2.rectangle(image, (10, 10), (width-10, height-10), (0, 0, 0), 3)

    # 画像を保存
    cv2.imwrite(output_path, image)
    print(f"✅ テスト画像を作成しました: {output_path}")

    return image


def main():
    """メイン関数"""
    # テスト用ディレクトリの作成
    test_dir = Path(__file__).parent / "test_images"
    test_dir.mkdir(exist_ok=True)

    # テストケース
    test_cases = [
        ("品川 330 あ 12-34", "test_shinagawa.jpg"),
        ("横浜 500 さ 56-78", "test_yokohama.jpg"),
        ("大阪 300 ま 90-12", "test_osaka.jpg"),
        ("名古屋 100 き 34-56", "test_nagoya.jpg"),
        ("札幌 555 ら 78-90", "test_sapporo.jpg"),
    ]

    print("\n" + "="*50)
    print("🖼️  テスト用ナンバープレート画像の生成")
    print("="*50 + "\n")

    for text, filename in test_cases:
        output_path = str(test_dir / filename)
        create_test_license_plate(text, output_path)

    print("\n" + "="*50)
    print(f"✅ {len(test_cases)}枚のテスト画像を生成しました")
    print(f"保存先: {test_dir}")
    print("="*50 + "\n")

    print("次のステップ:")
    print(f"  python custom_configs/license_plate_japan/run_ocr.py {test_dir}")
    print()


if __name__ == "__main__":
    main()
