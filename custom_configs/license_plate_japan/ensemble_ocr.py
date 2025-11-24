#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アンサンブルOCR推論スクリプト

複数の前処理戦略を試行し、最も信頼度の高い結果を選択することで
認識精度を向上させます。
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import cv2

# PaddleOCRのパスを追加
SCRIPT_DIR = Path(__file__).parent.resolve()
PADDLEOCR_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PADDLEOCR_ROOT))

from paddleocr import PaddleOCR
from optimized_preprocessing import OptimizedPreprocessor, PreprocessingStrategy
from postprocessing import LicensePlatePostprocessor


class EnsembleOCR:
    """アンサンブルOCRクラス"""

    def __init__(
        self,
        strategies: List[PreprocessingStrategy] = None,
        use_gpu: bool = False,
        verbose: bool = True
    ):
        """
        Args:
            strategies: 試行する前処理戦略のリスト
            use_gpu: GPUを使用するか
            verbose: 詳細ログを表示
        """
        self.verbose = verbose

        # デフォルトの戦略
        if strategies is None:
            strategies = [
                PreprocessingStrategy.STANDARD,
                PreprocessingStrategy.HIGH_CONTRAST,
                PreprocessingStrategy.SUPER_SHARP,
                PreprocessingStrategy.AGGRESSIVE
            ]

        self.strategies = strategies

        # 前処理クラスのリストを作成
        self.preprocessors = {
            strategy: OptimizedPreprocessor(strategy=strategy)
            for strategy in strategies
        }

        # 後処理クラス
        self.postprocessor = LicensePlatePostprocessor(
            min_confidence=0.3,  # アンサンブルでは低めの閾値
            enable_correction=True
        )

        # PaddleOCRの初期化
        if verbose:
            print("PaddleOCRを初期化中...")

        self.ocr = PaddleOCR(
            lang='japan',
            device='gpu' if use_gpu else 'cpu',
            use_textline_orientation=False
        )

        if verbose:
            print("✓ 初期化完了\n")

    def recognize(self, image_path: str) -> Dict:
        """
        複数の前処理戦略でOCRを実行し、最良の結果を返す

        Args:
            image_path: 入力画像のパス

        Returns:
            最良の認識結果
        """
        # 画像の読み込み
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'画像の読み込みに失敗: {image_path}',
                'image_path': image_path
            }

        if self.verbose:
            print(f"画像: {Path(image_path).name}")
            print(f"元のサイズ: {image.shape[1]}x{image.shape[0]}\n")

        # 各前処理戦略で推論
        results = []
        for strategy in self.strategies:
            if self.verbose:
                print(f"  試行中: {strategy.value}")

            # 前処理
            preprocessor = self.preprocessors[strategy]
            processed_image = preprocessor.preprocess(image.copy())

            # OCR実行
            try:
                ocr_results = self.ocr.predict(processed_image)

                if ocr_results and len(ocr_results) > 0:
                    result_dict = ocr_results[0]

                    if 'rec_texts' in result_dict and result_dict['rec_texts']:
                        texts = result_dict['rec_texts']
                        scores = result_dict['rec_scores']

                        # 全テキストを結合
                        text = ' '.join(texts)
                        confidence = sum(scores) / len(scores)

                        # 後処理
                        post_result = self.postprocessor.process(text, confidence)

                        results.append({
                            'strategy': strategy.value,
                            'text': text,
                            'confidence': confidence,
                            'is_valid': post_result.is_valid,
                            'corrected_text': post_result.corrected_text,
                            'region': post_result.region,
                            'classification': post_result.classification,
                            'hiragana': post_result.hiragana,
                            'number': post_result.number
                        })

                        if self.verbose:
                            print(f"    結果: {text} (信頼度: {confidence:.2%}, 有効: {post_result.is_valid})")

            except Exception as e:
                if self.verbose:
                    print(f"    エラー: {e}")
                continue

        if not results:
            return {
                'success': False,
                'error': 'すべての戦略で認識失敗',
                'image_path': image_path,
                'all_results': []
            }

        # 最良の結果を選択
        best_result = self._select_best_result(results)

        if self.verbose:
            print(f"\n✓ 最良の戦略: {best_result['strategy']}")
            print(f"  テキスト: {best_result.get('corrected_text') or best_result['text']}")
            print(f"  信頼度: {best_result['confidence']:.2%}\n")

        return {
            'success': True,
            'image_path': image_path,
            'best_strategy': best_result['strategy'],
            'text': best_result.get('corrected_text') or best_result['text'],
            'original_text': best_result['text'],
            'confidence': best_result['confidence'],
            'is_valid': best_result['is_valid'],
            'region': best_result.get('region'),
            'classification': best_result.get('classification'),
            'hiragana': best_result.get('hiragana'),
            'number': best_result.get('number'),
            'all_results': results
        }

    def _select_best_result(self, results: List[Dict]) -> Dict:
        """
        複数の結果から最良のものを選択

        選択基準:
        1. 後処理で有効と判定されたもの
        2. 信頼度が最も高いもの
        """
        # まず、有効な結果を優先
        valid_results = [r for r in results if r['is_valid']]

        if valid_results:
            # 有効な結果の中で最も信頼度が高いものを選択
            return max(valid_results, key=lambda x: x['confidence'])
        else:
            # 有効な結果がない場合は、単純に信頼度が最も高いものを選択
            return max(results, key=lambda x: x['confidence'])


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='アンサンブルOCR推論 - 複数の前処理戦略を試行して最良の結果を選択'
    )
    parser.add_argument('input', help='入力画像パス')
    parser.add_argument('--cpu', action='store_true', help='CPUを使用')
    parser.add_argument('--quiet', action='store_true', help='詳細ログを表示しない')
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=[s.value for s in PreprocessingStrategy],
        help='使用する戦略を指定'
    )

    args = parser.parse_args()

    # 戦略の選択
    if args.strategies:
        strategies = [PreprocessingStrategy(s) for s in args.strategies]
    else:
        strategies = None  # デフォルトを使用

    print("\n" + "="*60)
    print("🔬 アンサンブルOCR推論")
    print("="*60 + "\n")

    # アンサンブルOCRの初期化
    ensemble = EnsembleOCR(
        strategies=strategies,
        use_gpu=not args.cpu,
        verbose=not args.quiet
    )

    # 認識実行
    result = ensemble.recognize(args.input)

    # 結果の表示
    print("="*60)
    print("📋 認識結果")
    print("="*60)

    if result['success']:
        print(f"\n✅ 認識成功")
        print(f"  最良戦略: {result['best_strategy']}")
        print(f"  ナンバープレート: {result['text']}")
        print(f"  信頼度: {result['confidence']:.2%}")

        if result['is_valid']:
            print(f"\n  詳細:")
            print(f"    地域: {result.get('region', 'N/A')}")
            print(f"    分類番号: {result.get('classification', 'N/A')}")
            print(f"    ひらがな: {result.get('hiragana', 'N/A')}")
            print(f"    車両番号: {result.get('number', 'N/A')}")

        # 全結果の表示
        if not args.quiet and result['all_results']:
            print(f"\n  全戦略の結果 ({len(result['all_results'])}件):")
            for i, r in enumerate(result['all_results'], 1):
                print(f"    {i}. [{r['strategy']}] {r['text']} "
                      f"(信頼度: {r['confidence']:.2%}, 有効: {r['is_valid']})")
    else:
        print(f"\n❌ 認識失敗")
        print(f"  エラー: {result.get('error', 'Unknown')}")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
