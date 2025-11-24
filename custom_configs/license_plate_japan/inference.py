#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本のナンバープレートOCR推論スクリプト

事前学習済みモデルまたはファインチューニング済みモデルを使用して、
ナンバープレート画像からテキストを認識します。
"""

import os
import sys
import cv2
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# PaddleOCRのパスを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Error: PaddleOCRがインストールされていません。")
    print("インストール方法: pip install paddleocr")
    sys.exit(1)

# カスタムモジュールのインポート
from preprocessing import LicensePlatePreprocessor
from postprocessing import LicensePlatePostprocessor, format_result


class LicensePlateOCR:
    """日本のナンバープレート認識クラス"""

    def __init__(
        self,
        rec_model_dir: Optional[str] = None,
        rec_char_dict_path: Optional[str] = None,
        use_gpu: bool = True,
        use_preprocessing: bool = True,
        use_postprocessing: bool = True,
        min_confidence: float = 0.5
    ):
        """
        Args:
            rec_model_dir: 認識モデルのディレクトリパス
            rec_char_dict_path: カスタム辞書のパス
            use_gpu: GPUを使用するか
            use_preprocessing: 前処理を使用するか
            use_postprocessing: 後処理を使用するか
            min_confidence: 最小信頼度スコア
        """
        self.use_preprocessing = use_preprocessing
        self.use_postprocessing = use_postprocessing

        # 前処理クラスの初期化
        if use_preprocessing:
            self.preprocessor = LicensePlatePreprocessor()

        # 後処理クラスの初期化
        if use_postprocessing:
            self.postprocessor = LicensePlatePostprocessor(
                min_confidence=min_confidence
            )

        # PaddleOCRの初期化
        ocr_params = {
            'use_angle_cls': False,  # 角度分類は不要（ナンバープレートは正面）
            'lang': 'japan',
            'use_gpu': use_gpu,
            'show_log': False
        }

        # カスタムモデルを使用する場合
        if rec_model_dir:
            ocr_params['rec_model_dir'] = rec_model_dir

        # カスタム辞書を使用する場合
        if rec_char_dict_path:
            ocr_params['rec_char_dict_path'] = rec_char_dict_path

        try:
            self.ocr = PaddleOCR(**ocr_params)
        except Exception as e:
            print(f"PaddleOCRの初期化に失敗しました: {e}")
            raise

    def recognize(
        self,
        image_path: str,
        return_details: bool = False
    ) -> Dict:
        """
        ナンバープレート画像を認識

        Args:
            image_path: 入力画像のパス
            return_details: 詳細情報を返すか

        Returns:
            認識結果の辞書
        """
        # 画像の読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")

        # 前処理
        if self.use_preprocessing:
            image = self.preprocessor.preprocess(image)

        # OCR実行
        results = self.ocr.ocr(image, cls=False)

        if not results or not results[0]:
            return {
                'success': False,
                'message': 'テキストが検出されませんでした',
                'text': None,
                'confidence': 0.0
            }

        # 最も信頼度の高い結果を選択
        best_result = max(results[0], key=lambda x: x[1][1])
        text, confidence = best_result[1]

        # 後処理
        if self.use_postprocessing:
            post_result = self.postprocessor.process(text, confidence)

            return {
                'success': post_result.is_valid,
                'text': post_result.corrected_text or text,
                'original_text': text,
                'confidence': confidence,
                'is_valid': post_result.is_valid,
                'region': post_result.region,
                'classification': post_result.classification,
                'hiragana': post_result.hiragana,
                'number': post_result.number,
                'details': post_result if return_details else None
            }
        else:
            return {
                'success': True,
                'text': text,
                'confidence': confidence
            }

    def batch_recognize(
        self,
        image_paths: List[str],
        output_csv: Optional[str] = None
    ) -> List[Dict]:
        """
        複数の画像を一括認識

        Args:
            image_paths: 入力画像のパスリスト
            output_csv: CSV出力パス（Noneの場合は出力しない）

        Returns:
            認識結果のリスト
        """
        results = []

        for i, image_path in enumerate(image_paths):
            print(f"処理中 ({i+1}/{len(image_paths)}): {image_path}")

            try:
                result = self.recognize(image_path)
                result['image_path'] = image_path
                results.append(result)

                # 結果を表示
                if result['success']:
                    print(f"  認識結果: {result['text']} (信頼度: {result['confidence']:.2%})")
                else:
                    print(f"  認識失敗: {result.get('message', 'Unknown error')}")

            except Exception as e:
                print(f"  エラー: {e}")
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'message': str(e)
                })

        # CSV出力
        if output_csv:
            self._save_to_csv(results, output_csv)

        return results

    def _save_to_csv(self, results: List[Dict], output_path: str):
        """
        結果をCSVファイルに保存

        Args:
            results: 認識結果のリスト
            output_path: 出力CSVファイルのパス
        """
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'image_path', 'success', 'text', 'confidence',
                'region', 'classification', 'hiragana', 'number'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow({
                    'image_path': result.get('image_path', ''),
                    'success': result.get('success', False),
                    'text': result.get('text', ''),
                    'confidence': result.get('confidence', 0.0),
                    'region': result.get('region', ''),
                    'classification': result.get('classification', ''),
                    'hiragana': result.get('hiragana', ''),
                    'number': result.get('number', ''),
                })

        print(f"\n結果をCSVに保存しました: {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='日本のナンバープレートOCR推論スクリプト'
    )
    parser.add_argument(
        'input',
        help='入力画像パスまたはディレクトリ'
    )
    parser.add_argument(
        '--rec_model_dir',
        help='認識モデルのディレクトリパス'
    )
    parser.add_argument(
        '--rec_char_dict_path',
        default='custom_configs/license_plate_japan/license_plate_dict.txt',
        help='カスタム辞書のパス'
    )
    parser.add_argument(
        '--output_csv',
        help='CSV出力パス'
    )
    parser.add_argument(
        '--no_preprocessing',
        action='store_true',
        help='前処理を無効化'
    )
    parser.add_argument(
        '--no_postprocessing',
        action='store_true',
        help='後処理を無効化'
    )
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help='GPUを使用しない'
    )
    parser.add_argument(
        '--min_confidence',
        type=float,
        default=0.5,
        help='最小信頼度スコア'
    )

    args = parser.parse_args()

    # OCRクラスの初期化
    print("ナンバープレートOCRを初期化中...")
    ocr = LicensePlateOCR(
        rec_model_dir=args.rec_model_dir,
        rec_char_dict_path=args.rec_char_dict_path,
        use_gpu=not args.no_gpu,
        use_preprocessing=not args.no_preprocessing,
        use_postprocessing=not args.no_postprocessing,
        min_confidence=args.min_confidence
    )

    # 入力パスの処理
    input_path = Path(args.input)

    if input_path.is_file():
        # 単一ファイル
        print(f"\n画像を認識中: {input_path}")
        result = ocr.recognize(str(input_path), return_details=True)

        if result['success']:
            print(f"\n認識成功!")
            print(f"  テキスト: {result['text']}")
            print(f"  信頼度: {result['confidence']:.2%}")
            print(f"  地域: {result['region']}")
            print(f"  分類番号: {result['classification']}")
            print(f"  ひらがな: {result['hiragana']}")
            print(f"  車両番号: {result['number']}")
        else:
            print(f"\n認識失敗: {result.get('message', 'Unknown error')}")

    elif input_path.is_dir():
        # ディレクトリ内の全画像
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = [
            str(p) for p in input_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            print(f"エラー: {input_path} に画像ファイルが見つかりませんでした")
            return

        print(f"\n{len(image_paths)}枚の画像を処理します...\n")
        results = ocr.batch_recognize(image_paths, args.output_csv)

        # サマリーを表示
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"\n=== 処理完了 ===")
        print(f"成功: {success_count}/{len(results)}")

    else:
        print(f"エラー: {input_path} は有効なファイルまたはディレクトリではありません")


if __name__ == "__main__":
    main()
