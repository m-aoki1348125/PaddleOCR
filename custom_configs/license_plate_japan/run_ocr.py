#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本のナンバープレートOCR - エンドツーエンド推論スクリプト

事前学習モデル + 前後処理を使用した、すぐに使えるナンバープレート認識スクリプト
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import json

# PaddleOCRのパスを追加
SCRIPT_DIR = Path(__file__).parent.resolve()
PADDLEOCR_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PADDLEOCR_ROOT))

print(f"PaddleOCR Root: {PADDLEOCR_ROOT}")

# カスタムモジュールのインポート
try:
    from custom_configs.license_plate_japan.preprocessing import LicensePlatePreprocessor
    from custom_configs.license_plate_japan.postprocessing import LicensePlatePostprocessor, format_result
except ImportError as e:
    print(f"Error: カスタムモジュールのインポートに失敗しました: {e}")
    print("現在のディレクトリから実行してください")
    sys.exit(1)

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Error: PaddleOCRがインストールされていません。")
    print("インストール方法: pip install paddleocr")
    sys.exit(1)


class SimpleLicensePlateOCR:
    """シンプルなナンバープレート認識クラス（事前学習モデル使用）"""

    def __init__(
        self,
        use_gpu: bool = True,
        enable_preprocessing: bool = True,
        enable_postprocessing: bool = True,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            use_gpu: GPUを使用するか
            enable_preprocessing: 前処理を有効化
            enable_postprocessing: 後処理を有効化
            det_model_dir: カスタム検出モデルディレクトリ
            rec_model_dir: カスタム認識モデルディレクトリ
            verbose: 詳細ログを表示
        """
        self.enable_preprocessing = enable_preprocessing
        self.enable_postprocessing = enable_postprocessing
        self.verbose = verbose

        # 前処理クラス
        if enable_preprocessing:
            self.preprocessor = LicensePlatePreprocessor(
                target_height=48,
                target_width=320,
                apply_denoising=True,
                apply_sharpening=True,
                apply_contrast_enhancement=True
            )
            if verbose:
                print("✓ 前処理を有効化")

        # 後処理クラス
        if enable_postprocessing:
            self.postprocessor = LicensePlatePostprocessor(
                min_confidence=0.5,
                enable_correction=True
            )
            if verbose:
                print("✓ 後処理を有効化")

        # PaddleOCRの初期化
        if verbose:
            print("PaddleOCRを初期化中...")

        ocr_params = {
            'use_angle_cls': False,
            'lang': 'japan',
            'use_gpu': use_gpu,
            'show_log': False,
            'rec_char_dict_path': str(SCRIPT_DIR / 'license_plate_dict.txt')
        }

        # カスタムモデルを指定
        if det_model_dir:
            ocr_params['det_model_dir'] = det_model_dir
        if rec_model_dir:
            ocr_params['rec_model_dir'] = rec_model_dir

        try:
            self.ocr = PaddleOCR(**ocr_params)
            if verbose:
                print("✓ PaddleOCR初期化完了")
        except Exception as e:
            print(f"Error: PaddleOCRの初期化に失敗しました: {e}")
            raise

    def recognize(self, image_path: str) -> Dict:
        """
        ナンバープレート画像を認識

        Args:
            image_path: 入力画像のパス

        Returns:
            認識結果の辞書
        """
        # 画像の読み込み
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'画像の読み込みに失敗: {image_path}',
                'image_path': image_path
            }

        # 前処理
        if self.enable_preprocessing:
            image = self.preprocessor.preprocess(image)
            if self.verbose:
                print("  ✓ 前処理完了")

        # OCR実行
        try:
            results = self.ocr.ocr(image, cls=False)
        except Exception as e:
            return {
                'success': False,
                'error': f'OCR実行エラー: {e}',
                'image_path': image_path
            }

        if not results or not results[0]:
            return {
                'success': False,
                'error': 'テキストが検出されませんでした',
                'image_path': image_path
            }

        # 最も信頼度の高い結果を選択
        best_result = max(results[0], key=lambda x: x[1][1])
        text, confidence = best_result[1]

        if self.verbose:
            print(f"  ✓ OCR実行完了: {text} (信頼度: {confidence:.2%})")

        # 後処理
        if self.enable_postprocessing:
            post_result = self.postprocessor.process(text, confidence)

            if self.verbose:
                print(f"  ✓ 後処理完了: 有効性={post_result.is_valid}")

            return {
                'success': post_result.is_valid,
                'image_path': image_path,
                'text': post_result.corrected_text or text,
                'original_text': text,
                'confidence': confidence,
                'is_valid': post_result.is_valid,
                'region': post_result.region,
                'classification': post_result.classification,
                'hiragana': post_result.hiragana,
                'number': post_result.number
            }
        else:
            return {
                'success': True,
                'image_path': image_path,
                'text': text,
                'confidence': confidence
            }

    def batch_recognize(self, image_paths: List[str]) -> List[Dict]:
        """
        複数の画像を一括認識

        Args:
            image_paths: 入力画像のパスリスト

        Returns:
            認識結果のリスト
        """
        results = []
        total = len(image_paths)

        print(f"\n{total}枚の画像を処理します...\n")

        for i, image_path in enumerate(image_paths, 1):
            print(f"[{i}/{total}] {Path(image_path).name}")

            result = self.recognize(image_path)
            results.append(result)

            if result['success']:
                print(f"  ✅ 認識成功: {result['text']}")
            else:
                print(f"  ❌ 認識失敗: {result.get('error', 'Unknown')}")

            print()

        return results


def save_results_to_csv(results: List[Dict], output_path: str):
    """結果をCSVファイルに保存"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'image_path', 'success', 'text', 'confidence',
            'region', 'classification', 'hiragana', 'number', 'error'
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
                'error': result.get('error', '')
            })

    print(f"✅ 結果をCSVに保存しました: {output_path}")


def save_results_to_json(results: List[Dict], output_path: str):
    """結果をJSONファイルに保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 結果をJSONに保存しました: {output_path}")


def print_summary(results: List[Dict]):
    """結果のサマリーを表示"""
    total = len(results)
    success = sum(1 for r in results if r.get('success', False))
    failed = total - success

    print("\n" + "="*50)
    print("📊 認識結果サマリー")
    print("="*50)
    print(f"総数:     {total}枚")
    print(f"成功:     {success}枚 ({success/total*100:.1f}%)")
    print(f"失敗:     {failed}枚 ({failed/total*100:.1f}%)")

    if success > 0:
        avg_confidence = sum(
            r.get('confidence', 0) for r in results if r.get('success', False)
        ) / success
        print(f"平均信頼度: {avg_confidence:.2%}")

    print("="*50 + "\n")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='日本のナンバープレートOCR - エンドツーエンド推論',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一画像の認識
  python run_ocr.py plate.jpg

  # ディレクトリ内の全画像を処理
  python run_ocr.py images/ --output_csv results.csv

  # 前処理・後処理を無効化
  python run_ocr.py plate.jpg --no_preprocessing --no_postprocessing

  # CPUを使用
  python run_ocr.py plate.jpg --cpu
        """
    )

    parser.add_argument(
        'input',
        help='入力画像パスまたはディレクトリ'
    )
    parser.add_argument(
        '--output_csv',
        help='CSV出力パス'
    )
    parser.add_argument(
        '--output_json',
        help='JSON出力パス'
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
        '--cpu',
        action='store_true',
        help='CPUを使用（GPUを使用しない）'
    )
    parser.add_argument(
        '--det_model_dir',
        help='カスタム検出モデルディレクトリ'
    )
    parser.add_argument(
        '--rec_model_dir',
        help='カスタム認識モデルディレクトリ'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='詳細ログを表示しない'
    )

    args = parser.parse_args()

    print("\n" + "="*50)
    print("🚗 日本のナンバープレートOCR")
    print("="*50 + "\n")

    # OCRクラスの初期化
    ocr = SimpleLicensePlateOCR(
        use_gpu=not args.cpu,
        enable_preprocessing=not args.no_preprocessing,
        enable_postprocessing=not args.no_postprocessing,
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
        verbose=not args.quiet
    )

    # 入力パスの処理
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ エラー: {input_path} が見つかりません")
        sys.exit(1)

    # 単一ファイル
    if input_path.is_file():
        print(f"📷 画像を認識中: {input_path.name}\n")

        result = ocr.recognize(str(input_path))

        print("\n" + "="*50)
        print("📋 認識結果")
        print("="*50)

        if result['success']:
            print(f"✅ 認識成功")
            print(f"  ナンバープレート: {result['text']}")
            print(f"  信頼度: {result['confidence']:.2%}")
            print(f"  地域: {result.get('region', 'N/A')}")
            print(f"  分類番号: {result.get('classification', 'N/A')}")
            print(f"  ひらがな: {result.get('hiragana', 'N/A')}")
            print(f"  車両番号: {result.get('number', 'N/A')}")
        else:
            print(f"❌ 認識失敗")
            print(f"  エラー: {result.get('error', 'Unknown')}")

        print("="*50 + "\n")

        # JSON出力
        if args.output_json:
            save_results_to_json([result], args.output_json)

    # ディレクトリ
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [
            str(p) for p in input_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            print(f"❌ エラー: {input_path} に画像ファイルが見つかりませんでした")
            sys.exit(1)

        # 一括処理
        results = ocr.batch_recognize(image_paths)

        # サマリー表示
        print_summary(results)

        # CSV出力
        if args.output_csv:
            save_results_to_csv(results, args.output_csv)

        # JSON出力
        if args.output_json:
            save_results_to_json(results, args.output_json)

    else:
        print(f"❌ エラー: {input_path} は有効なファイルまたはディレクトリではありません")
        sys.exit(1)


if __name__ == "__main__":
    main()
