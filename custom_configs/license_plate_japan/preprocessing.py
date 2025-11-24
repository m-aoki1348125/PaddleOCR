#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本のナンバープレートOCR用前処理スクリプト

パトカー車載カメラから取得した低解像度画像に対して、
超解像処理後の画像をさらに最適化する前処理を実施します。
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class LicensePlatePreprocessor:
    """ナンバープレート画像の前処理クラス"""

    def __init__(
        self,
        target_height: int = 48,
        target_width: int = 320,
        apply_denoising: bool = True,
        apply_sharpening: bool = True,
        apply_contrast_enhancement: bool = True
    ):
        """
        Args:
            target_height: リサイズ後の高さ
            target_width: リサイズ後の幅
            apply_denoising: ノイズ除去を適用するか
            apply_sharpening: シャープニングを適用するか
            apply_contrast_enhancement: コントラスト強調を適用するか
        """
        self.target_height = target_height
        self.target_width = target_width
        self.apply_denoising = apply_denoising
        self.apply_sharpening = apply_sharpening
        self.apply_contrast_enhancement = apply_contrast_enhancement

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        ナンバープレート画像の前処理を実行

        Args:
            image: 入力画像 (BGR形式)

        Returns:
            前処理後の画像
        """
        # 1. ノイズ除去（超解像処理後の残留ノイズを除去）
        if self.apply_denoising:
            image = self._denoise(image)

        # 2. コントラスト強調（文字と背景の区別を明確に）
        if self.apply_contrast_enhancement:
            image = self._enhance_contrast(image)

        # 3. シャープニング（エッジを強調して文字を明瞭に）
        if self.apply_sharpening:
            image = self._sharpen(image)

        # 4. リサイズ（アスペクト比を維持しながらリサイズ）
        image = self._resize_keep_ratio(image)

        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        ノイズ除去処理

        非局所平均デノイジング（Non-Local Means Denoising）を使用
        超解像処理後の画像に適した軽度のデノイジング
        """
        # カラー画像用のデノイジング
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=3,  # フィルタ強度（低めに設定して詳細を保持）
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        コントラスト強調処理

        CLAHE (Contrast Limited Adaptive Histogram Equalization)を使用
        局所的なコントラストを適応的に強調
        """
        # LAB色空間に変換（輝度チャンネルのみを処理）
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHEを適用
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # LAB画像を再構成してBGRに戻す
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        シャープニング処理

        Unsharp Maskingを使用して文字のエッジを強調
        """
        # ガウシアンブラーでぼかし画像を作成
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)

        # Unsharp Masking: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        return sharpened

    def _resize_keep_ratio(
        self,
        image: np.ndarray,
        pad_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        アスペクト比を維持しながらリサイズ

        Args:
            image: 入力画像
            pad_color: パディングの色 (B, G, R)

        Returns:
            リサイズ後の画像
        """
        h, w = image.shape[:2]

        # アスペクト比を計算
        aspect_ratio = w / h
        target_aspect_ratio = self.target_width / self.target_height

        if aspect_ratio > target_aspect_ratio:
            # 幅が広い場合、幅を基準にリサイズ
            new_w = self.target_width
            new_h = int(self.target_width / aspect_ratio)
        else:
            # 高さが高い場合、高さを基準にリサイズ
            new_h = self.target_height
            new_w = int(self.target_height * aspect_ratio)

        # リサイズ
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # パディングを追加してターゲットサイズにする
        padded = np.full(
            (self.target_height, self.target_width, 3),
            pad_color,
            dtype=np.uint8
        )

        # 中央に配置
        y_offset = (self.target_height - new_h) // 2
        x_offset = (self.target_width - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return padded


def preprocess_image(
    image_path: str,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    画像ファイルを読み込んで前処理を実行

    Args:
        image_path: 入力画像のパス
        output_path: 出力画像のパス（Noneの場合は保存しない）

    Returns:
        前処理後の画像
    """
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像の読み込みに失敗しました: {image_path}")

    # 前処理を実行
    preprocessor = LicensePlatePreprocessor()
    processed = preprocessor.preprocess(image)

    # 出力パスが指定されていれば保存
    if output_path:
        cv2.imwrite(output_path, processed)

    return processed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python preprocessing.py <入力画像パス> [出力画像パス]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        processed_image = preprocess_image(input_path, output_path)
        print(f"前処理完了: {input_path}")
        if output_path:
            print(f"保存先: {output_path}")
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)
