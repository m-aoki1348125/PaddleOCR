#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前処理バリエーション実装

異なる前処理パターンを提供し、画像に応じて最適なものを選択できるようにします。
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class PreprocessingStrategy(Enum):
    """前処理戦略の種類"""
    STANDARD = "standard"  # 標準的な前処理
    AGGRESSIVE = "aggressive"  # より強力な前処理
    GENTLE = "gentle"  # 軽めの前処理
    HIGH_CONTRAST = "high_contrast"  # コントラスト重視
    SUPER_SHARP = "super_sharp"  # シャープネス重視
    DENOISE_FOCUSED = "denoise_focused"  # ノイズ除去重視


class OptimizedPreprocessor:
    """最適化された前処理クラス"""

    def __init__(
        self,
        strategy: PreprocessingStrategy = PreprocessingStrategy.STANDARD,
        target_height: int = 48,
        target_width: int = 320
    ):
        """
        Args:
            strategy: 前処理戦略
            target_height: リサイズ後の高さ
            target_width: リサイズ後の幅
        """
        self.strategy = strategy
        self.target_height = target_height
        self.target_width = target_width

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        戦略に応じた前処理を実行

        Args:
            image: 入力画像 (BGR形式)

        Returns:
            前処理後の画像
        """
        if self.strategy == PreprocessingStrategy.STANDARD:
            return self._standard_preprocessing(image)
        elif self.strategy == PreprocessingStrategy.AGGRESSIVE:
            return self._aggressive_preprocessing(image)
        elif self.strategy == PreprocessingStrategy.GENTLE:
            return self._gentle_preprocessing(image)
        elif self.strategy == PreprocessingStrategy.HIGH_CONTRAST:
            return self._high_contrast_preprocessing(image)
        elif self.strategy == PreprocessingStrategy.SUPER_SHARP:
            return self._super_sharp_preprocessing(image)
        elif self.strategy == PreprocessingStrategy.DENOISE_FOCUSED:
            return self._denoise_focused_preprocessing(image)
        else:
            return self._standard_preprocessing(image)

    def _standard_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """標準的な前処理"""
        # 1. 軽いデノイジング
        image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)

        # 2. CLAHE（適度なコントラスト強調）
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 3. Unsharp Masking
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        # 4. リサイズ
        image = self._resize_keep_ratio(image)

        return image

    def _aggressive_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """より強力な前処理（低品質画像向け）"""
        # 1. より強力なデノイジング
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # 2. より強いコントラスト強調
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 3. より強いシャープニング
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel)

        # 4. 二値化を適用（エッジ強調）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 5. リサイズ
        image = self._resize_keep_ratio(image)

        return image

    def _gentle_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """軽めの前処理（高品質画像向け）"""
        # 1. デノイジングなし、または最小限

        # 2. 軽いコントラスト強調
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 3. 軽いシャープニング
        blurred = cv2.GaussianBlur(image, (0, 0), 0.5)
        image = cv2.addWeighted(image, 1.2, blurred, -0.2, 0)

        # 4. リサイズ
        image = self._resize_keep_ratio(image)

        return image

    def _high_contrast_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """コントラスト重視の前処理"""
        # 1. ヒストグラム均等化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        # 2. 非常に強いCLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 3. ガンマ補正
        image = self._adjust_gamma(image, gamma=1.2)

        # 4. リサイズ
        image = self._resize_keep_ratio(image)

        return image

    def _super_sharp_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """シャープネス重視の前処理"""
        # 1. 軽いデノイジング
        image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)

        # 2. Laplacianシャープニング
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        image = cv2.add(image, cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR))

        # 3. Unsharp Maskingを強めに
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        image = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)

        # 4. リサイズ
        image = self._resize_keep_ratio(image)

        return image

    def _denoise_focused_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """ノイズ除去重視の前処理"""
        # 1. バイラテラルフィルタ（エッジ保存デノイジング）
        image = cv2.bilateralFilter(image, 9, 75, 75)

        # 2. 非局所平均デノイジング
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # 3. モルフォロジー演算でノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # 4. 軽いコントラスト強調
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 5. リサイズ
        image = self._resize_keep_ratio(image)

        return image

    def _adjust_gamma(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """ガンマ補正"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _resize_keep_ratio(
        self,
        image: np.ndarray,
        pad_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """アスペクト比を維持しながらリサイズ"""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        target_aspect_ratio = self.target_width / self.target_height

        if aspect_ratio > target_aspect_ratio:
            new_w = self.target_width
            new_h = int(self.target_width / aspect_ratio)
        else:
            new_h = self.target_height
            new_w = int(self.target_height * aspect_ratio)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        padded = np.full(
            (self.target_height, self.target_width, 3),
            pad_color,
            dtype=np.uint8
        )

        y_offset = (self.target_height - new_h) // 2
        x_offset = (self.target_width - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return padded


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("使用方法: python optimized_preprocessing.py <入力画像> [戦略]")
        print("\n利用可能な戦略:")
        for strategy in PreprocessingStrategy:
            print(f"  - {strategy.value}")
        sys.exit(1)

    input_path = sys.argv[1]
    strategy_name = sys.argv[2] if len(sys.argv) > 2 else "standard"

    # 戦略の選択
    try:
        strategy = PreprocessingStrategy(strategy_name)
    except ValueError:
        print(f"エラー: 無効な戦略 '{strategy_name}'")
        print("\n利用可能な戦略:")
        for s in PreprocessingStrategy:
            print(f"  - {s.value}")
        sys.exit(1)

    # 画像の読み込み
    image = cv2.imread(input_path)
    if image is None:
        print(f"エラー: 画像が読み込めません: {input_path}")
        sys.exit(1)

    # 前処理の実行
    print(f"\n前処理戦略: {strategy.value}")
    preprocessor = OptimizedPreprocessor(strategy=strategy)
    processed = preprocessor.preprocess(image)

    # 保存
    output_dir = Path("preprocessed_variations")
    output_dir.mkdir(exist_ok=True)

    input_stem = Path(input_path).stem
    output_path = output_dir / f"{input_stem}_{strategy.value}.jpg"
    cv2.imwrite(str(output_path), processed)

    print(f"✓ 保存完了: {output_path}")
