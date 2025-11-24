#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低解像度ナンバープレート画像の生成スクリプト

パトカー車載カメラからの低解像度画像と、
超解像処理後の画像の両方を生成します。
"""

from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import numpy as np
except ImportError:
    print("Error: Pillowがインストールされていません。")
    print("インストール方法: pip install Pillow")
    exit(1)


class LicensePlateSampleGenerator:
    """ナンバープレートサンプル画像生成クラス"""

    def __init__(self):
        # 日本のナンバープレート標準サイズ比率（横:縦 = 2.2:1）
        self.aspect_ratio = 2.2

    def create_high_res_plate(
        self,
        text: str,
        width: int = 440,
        bg_color: tuple = (255, 255, 255),
        text_color: tuple = (0, 0, 0)
    ) -> Image.Image:
        """
        高解像度のナンバープレート画像を作成

        Args:
            text: ナンバープレートのテキスト（例: "品川 330\nあ 12-34"）
            width: 画像の幅
            bg_color: 背景色 (R, G, B)
            text_color: テキスト色 (R, G, B)

        Returns:
            PIL Image
        """
        # 高さを計算
        height = int(width / self.aspect_ratio)

        # 白い背景を作成
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image)

        # 枠線を描画
        border_width = 5
        draw.rectangle(
            [border_width, border_width, width-border_width, height-border_width],
            outline=text_color,
            width=border_width
        )

        # テキストを描画（複数行対応）
        lines = text.split('\n')

        try:
            # システムフォントを試す
            font_size = int(height / (len(lines) + 1))
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            # フォントが見つからない場合はデフォルト
            font = ImageFont.load_default()

        # 各行を描画
        y_offset = height // (len(lines) + 1)
        for i, line in enumerate(lines):
            # テキストのバウンディングボックスを取得
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 中央に配置
            x = (width - text_width) // 2
            y = y_offset * (i + 1) - text_height // 2

            draw.text((x, y), line, fill=text_color, font=font)

        return image

    def simulate_low_resolution(
        self,
        image: Image.Image,
        scale_factor: float = 0.15
    ) -> Image.Image:
        """
        低解像度をシミュレート

        Args:
            image: 入力画像
            scale_factor: 縮小率（0.1 = 10%のサイズ）

        Returns:
            低解像度画像
        """
        # 小さいサイズに縮小
        small_width = int(image.width * scale_factor)
        small_height = int(image.height * scale_factor)

        low_res = image.resize((small_width, small_height), Image.BILINEAR)

        # ノイズを追加
        low_res_array = np.array(low_res)
        noise = np.random.normal(0, 10, low_res_array.shape)
        noisy = np.clip(low_res_array + noise, 0, 255).astype(np.uint8)
        low_res = Image.fromarray(noisy)

        # ぼかしを追加
        low_res = low_res.filter(ImageFilter.GaussianBlur(radius=0.5))

        return low_res

    def simulate_super_resolution(
        self,
        low_res_image: Image.Image,
        target_width: int = 320
    ) -> Image.Image:
        """
        超解像処理をシミュレート

        Args:
            low_res_image: 低解像度画像
            target_width: 目標の幅

        Returns:
            超解像処理後の画像
        """
        # アスペクト比を維持してリサイズ
        aspect = low_res_image.height / low_res_image.width
        target_height = int(target_width * aspect)

        # Lanczosリサンプリングで高品質にリサイズ
        super_res = low_res_image.resize(
            (target_width, target_height),
            Image.LANCZOS
        )

        # 軽いシャープニング
        super_res = super_res.filter(ImageFilter.SHARPEN)

        return super_res

    def generate_sample_set(
        self,
        text: str,
        output_dir: Path,
        basename: str
    ):
        """
        1つのナンバープレートについて3種類の画像を生成

        Args:
            text: ナンバープレートのテキスト
            output_dir: 出力ディレクトリ
            basename: ファイル名のベース
        """
        # 1. 高解像度画像を生成
        high_res = self.create_high_res_plate(text)
        high_res_path = output_dir / f"{basename}_high_res.jpg"
        high_res.save(high_res_path, quality=95)
        print(f"  ✓ 高解像度: {high_res_path.name}")

        # 2. 低解像度画像を生成
        low_res = self.simulate_low_resolution(high_res, scale_factor=0.15)
        low_res_path = output_dir / f"{basename}_low_res.jpg"
        low_res.save(low_res_path, quality=70)
        print(f"  ✓ 低解像度: {low_res_path.name} ({low_res.width}x{low_res.height})")

        # 3. 超解像処理後の画像を生成
        super_res = self.simulate_super_resolution(low_res, target_width=320)
        super_res_path = output_dir / f"{basename}_super_res.jpg"
        super_res.save(super_res_path, quality=90)
        print(f"  ✓ 超解像処理: {super_res_path.name} ({super_res.width}x{super_res.height})")


def main():
    """メイン関数"""
    print("\n" + "="*60)
    print("🖼️  低解像度ナンバープレートサンプル画像の生成")
    print("="*60 + "\n")

    # 出力ディレクトリの作成
    script_dir = Path(__file__).parent
    sample_dir = script_dir / "sample_images"
    sample_dir.mkdir(exist_ok=True)

    # サンプルデータ
    samples = [
        {
            "text": "品川 330\nあ 12-34",
            "basename": "plate_shinagawa",
            "description": "品川 330 あ 12-34"
        },
        {
            "text": "横浜 500\nさ 56-78",
            "basename": "plate_yokohama",
            "description": "横浜 500 さ 56-78"
        },
        {
            "text": "大阪 300\nま 90-12",
            "basename": "plate_osaka",
            "description": "大阪 300 ま 90-12"
        },
        {
            "text": "名古屋 100\nき 34-56",
            "basename": "plate_nagoya",
            "description": "名古屋 100 き 34-56"
        },
        {
            "text": "札幌 555\nら 78-90",
            "basename": "plate_sapporo",
            "description": "札幌 555 ら 78-90"
        },
    ]

    # 画像生成
    generator = LicensePlateSampleGenerator()

    for sample in samples:
        print(f"📷 生成中: {sample['description']}")
        generator.generate_sample_set(
            text=sample['text'],
            output_dir=sample_dir,
            basename=sample['basename']
        )
        print()

    print("="*60)
    print(f"✅ {len(samples)}セット（{len(samples)*3}枚）の画像を生成しました")
    print(f"保存先: {sample_dir}")
    print("="*60 + "\n")

    print("各ナンバープレートについて、3種類の画像を生成:")
    print("  1. *_high_res.jpg  - オリジナルの高解像度画像")
    print("  2. *_low_res.jpg   - パトカー車載カメラからの低解像度画像（シミュレート）")
    print("  3. *_super_res.jpg - 超解像処理後の画像（シミュレート）")
    print()

    print("次のステップ:")
    print(f"  # 低解像度画像で認識を試す")
    print(f"  python custom_configs/license_plate_japan/run_ocr.py {sample_dir}/*_low_res.jpg")
    print()
    print(f"  # 超解像処理後の画像で認識を試す（推奨）")
    print(f"  python custom_configs/license_plate_japan/run_ocr.py {sample_dir}/*_super_res.jpg")
    print()
    print(f"  # 全画像を一括処理")
    print(f"  python custom_configs/license_plate_japan/run_ocr.py {sample_dir} --output_csv results.csv")
    print()


if __name__ == "__main__":
    main()
