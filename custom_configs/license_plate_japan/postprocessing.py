#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本のナンバープレートOCR用後処理スクリプト

OCR結果を日本のナンバープレートフォーマットに基づいて検証・補正します。
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LicensePlateResult:
    """ナンバープレート認識結果"""
    text: str
    confidence: float
    is_valid: bool
    corrected_text: Optional[str] = None
    region: Optional[str] = None  # 地域名（例：「品川」）
    classification: Optional[str] = None  # 分類番号（例：「330」）
    hiragana: Optional[str] = None  # ひらがな1文字（例：「あ」）
    number: Optional[str] = None  # 車両番号（例：「12-34」）


class LicensePlatePostprocessor:
    """ナンバープレート認識結果の後処理クラス"""

    # ナンバープレートで使用されるひらがな（お、し、へ、んは除外）
    VALID_HIRAGANA = set(
        'あいうえかきくけこさすせそたちつてとなにぬねの'
        'はひふほまみむめもやゆよらりるろれを'
    )

    # 主要な地域名（一部）
    VALID_REGIONS = {
        '札幌', '函館', '室蘭', '帯広', '釧路', '北見', '旭川',
        '青森', '岩手', '宮城', '秋田', '山形', '福島',
        '水戸', '土浦', '宇都宮', '栃木', '群馬', '埼玉', '千葉',
        '品川', '練馬', '足立', '多摩', '八王子', '横浜', '相模', '湘南',
        '新潟', '長岡', '富山', '石川', '金沢', '福井',
        '山梨', '長野', '松本', '岐阜', '飛騨', '静岡', '浜松', '沼津',
        '名古屋', '尾張小牧', '一宮', '春日井', '豊田', '豊橋', '岡崎', '三河',
        '津', '四日市', '鈴鹿', '滋賀', '京都', '大阪', 'なにわ', '和泉',
        '堺', '神戸', '姫路', '奈良', '和歌山',
        '鳥取', '島根', '岡山', '倉敷', '広島', '福山', '山口', '下関',
        '徳島', '香川', '愛媛', '高知',
        '福岡', '北九州', '久留米', '筑豊', '佐賀', '長崎', '佐世保',
        '熊本', '大分', '宮崎', '鹿児島', '沖縄'
    }

    def __init__(
        self,
        min_confidence: float = 0.5,
        enable_correction: bool = True
    ):
        """
        Args:
            min_confidence: 最小信頼度スコア
            enable_correction: 自動補正を有効にするか
        """
        self.min_confidence = min_confidence
        self.enable_correction = enable_correction

    def process(
        self,
        text: str,
        confidence: float
    ) -> LicensePlateResult:
        """
        OCR結果を後処理

        Args:
            text: OCRで認識されたテキスト
            confidence: 信頼度スコア

        Returns:
            後処理結果
        """
        # 信頼度チェック
        if confidence < self.min_confidence:
            return LicensePlateResult(
                text=text,
                confidence=confidence,
                is_valid=False
            )

        # テキストをクリーンアップ
        cleaned_text = self._clean_text(text)

        # ナンバープレートのパースを試みる
        parsed = self._parse_license_plate(cleaned_text)

        if parsed:
            # パース成功
            region, classification, hiragana, number = parsed
            is_valid = self._validate_components(
                region, classification, hiragana, number
            )

            # フォーマットされたテキスト
            formatted_text = f"{region} {classification} {hiragana} {number}"

            return LicensePlateResult(
                text=text,
                confidence=confidence,
                is_valid=is_valid,
                corrected_text=formatted_text if self.enable_correction else None,
                region=region,
                classification=classification,
                hiragana=hiragana,
                number=number
            )
        else:
            # パース失敗
            return LicensePlateResult(
                text=text,
                confidence=confidence,
                is_valid=False
            )

    def _clean_text(self, text: str) -> str:
        """
        テキストのクリーンアップ

        Args:
            text: 入力テキスト

        Returns:
            クリーンアップされたテキスト
        """
        # 余分な空白を除去
        text = ' '.join(text.split())

        # よくある誤認識の補正
        if self.enable_correction:
            # 数字の誤認識補正
            text = text.replace('O', '0')  # アルファベットのOを0に
            text = text.replace('I', '1')  # アルファベットのIを1に
            text = text.replace('Z', '2')  # 場合によってZが2と誤認識
            text = text.replace('S', '5')  # 場合によってSが5と誤認識

        return text

    def _parse_license_plate(
        self,
        text: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """
        ナンバープレートテキストをパース

        日本のナンバープレートフォーマット:
        [地域名] [分類番号(3桁)] [ひらがな1文字] [車両番号(1-4桁、ハイフンあり)]
        例: 品川 330 あ 12-34

        Args:
            text: 入力テキスト

        Returns:
            (地域名, 分類番号, ひらがな, 車両番号) または None
        """
        # パターン1: スペース区切り（理想的なケース）
        # 例: "品川 330 あ 12-34"
        pattern1 = r'^([^\s\d]+)\s+(\d{3})\s+([^\s\d]{1})\s+([\d-]+)$'
        match = re.match(pattern1, text)
        if match:
            return match.groups()

        # パターン2: スペースなし
        # 例: "品川330あ12-34"
        pattern2 = r'^([^\d]+?)(\d{3})([^\d\s]{1})([\d-]+)$'
        match = re.match(pattern2, text)
        if match:
            return match.groups()

        # パターン3: 部分的なスペース
        # 例: "品川330 あ 12-34" や "品川 330あ12-34"
        pattern3 = r'^([^\d]+?)\s*(\d{3})\s*([^\d\s]{1})\s*([\d-]+)$'
        match = re.match(pattern3, text)
        if match:
            return match.groups()

        return None

    def _validate_components(
        self,
        region: str,
        classification: str,
        hiragana: str,
        number: str
    ) -> bool:
        """
        ナンバープレート各要素の検証

        Args:
            region: 地域名
            classification: 分類番号
            hiragana: ひらがな
            number: 車両番号

        Returns:
            全て有効ならTrue
        """
        # 地域名の検証（主要地域のみチェック、完全一致でなくてもOK）
        region_valid = any(r in region for r in self.VALID_REGIONS)

        # 分類番号の検証（3桁の数字）
        classification_valid = (
            len(classification) == 3 and
            classification.isdigit()
        )

        # ひらがなの検証
        hiragana_valid = (
            len(hiragana) == 1 and
            hiragana in self.VALID_HIRAGANA
        )

        # 車両番号の検証（1-4桁の数字、ハイフンあり/なし）
        number_pattern = r'^\d{1,2}-?\d{1,2}$'
        number_valid = bool(re.match(number_pattern, number))

        return (
            region_valid and
            classification_valid and
            hiragana_valid and
            number_valid
        )

    def batch_process(
        self,
        results: List[Tuple[str, float]]
    ) -> List[LicensePlateResult]:
        """
        複数のOCR結果を一括処理

        Args:
            results: [(text, confidence), ...] のリスト

        Returns:
            後処理結果のリスト
        """
        return [
            self.process(text, confidence)
            for text, confidence in results
        ]


def format_result(result: LicensePlateResult) -> str:
    """
    結果を人間が読みやすい形式にフォーマット

    Args:
        result: 後処理結果

    Returns:
        フォーマットされた文字列
    """
    lines = [
        f"認識テキスト: {result.text}",
        f"信頼度: {result.confidence:.2%}",
        f"有効性: {'有効' if result.is_valid else '無効'}"
    ]

    if result.corrected_text:
        lines.append(f"補正後: {result.corrected_text}")

    if result.is_valid:
        lines.extend([
            f"  地域: {result.region}",
            f"  分類番号: {result.classification}",
            f"  ひらがな: {result.hiragana}",
            f"  車両番号: {result.number}"
        ])

    return '\n'.join(lines)


if __name__ == "__main__":
    # テストケース
    postprocessor = LicensePlatePostprocessor()

    test_cases = [
        ("品川 330 あ 12-34", 0.95),
        ("横浜330さ1234", 0.88),
        ("大阪 500 ま 56-78", 0.92),
        ("invalid text", 0.65),
        ("品川 33O あ 12-34", 0.85),  # Oが含まれる（補正対象）
    ]

    print("=== ナンバープレート後処理テスト ===\n")
    for text, conf in test_cases:
        result = postprocessor.process(text, conf)
        print(format_result(result))
        print("-" * 50)
