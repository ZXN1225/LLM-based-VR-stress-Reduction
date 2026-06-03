import argparse
import os
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--supir_dir", required=True, help="SUPIR repo directory")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--edm_steps", type=int, default=30)
    parser.add_argument("--s_cfg", type=float, default=4.0)
    parser.add_argument("--s_stage2", type=float, default=1.0)
    parser.add_argument("--s_stage1", type=int, default=-1)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    supir_dir = Path(args.supir_dir).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if not supir_dir.exists():
        raise FileNotFoundError(f"SUPIR directory not found: {supir_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_dir = tmp / "input"
        save_dir = tmp / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        temp_input = input_dir / input_path.name
        shutil.copy2(input_path, temp_input)

        import subprocess

        cmd = [
            "python",
            str(supir_dir / "test.py"),
            "--img_dir", str(input_dir),
            "--save_dir", str(save_dir),
            "--upscale", str(args.scale),
            "--SUPIR_sign", "Q",
            "--seed", str(args.seed),
            "--edm_steps", str(args.edm_steps),
            "--s_cfg", str(args.s_cfg),
            "--s_stage2", str(args.s_stage2),
            "--s_stage1", str(int(args.s_stage1)),
            "--no_llava",
            "--loading_half_params",
            "--use_tile_vae",
            "--ae_dtype", "fp32",
            "--diff_dtype", "fp16",
            "--color_fix_type", "Wavelet",
        ]

        print("[SUPIR single] Running:")
        print(" ".join(cmd))

        subprocess.run(
            cmd,
            cwd=str(supir_dir),
            check=True
        )

        outputs = list(save_dir.glob("*.png")) + list(save_dir.glob("*.jpg"))
        if not outputs:
            raise RuntimeError("SUPIR finished but no output image was generated.")

        newest = max(outputs, key=lambda p: p.stat().st_mtime)
        shutil.copy2(newest, output_path)

        print(f"[SUPIR single] Saved: {output_path}")


if __name__ == "__main__":
    main()