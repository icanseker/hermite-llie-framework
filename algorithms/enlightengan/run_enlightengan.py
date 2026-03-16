"""
run_enlightengan.py — EnlightenGAN inference, CLI wrapper for MATLAB.

Called by run_EnlightenGAN.m via system():
    python run_enlightengan.py <input_path> <output_path>

Reads the image at <input_path>, runs EnlightenGAN ONNX inference,
writes the enhanced result to <output_path>.
Exit code 0 = success, 1 = failure.

Uses the enlighten_inference package located next to this script in
algorithms/enlightengan/enlighten_inference/. The package is from:
    https://github.com/arsenyinfo/EnlightenGAN-inference
"""

import sys
import os
import cv2


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_enlightengan.py <input_path> <output_path>",
              file=sys.stderr)
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Add this script's directory to sys.path so the local
    # enlighten_inference/ package folder is found first.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    try:
        from enlighten_inference import EnlightenOnnxModel
    except ImportError:
        print(
            "Error: enlighten_inference package not found.\n"
            "Expected location: algorithms/enlightengan/enlighten_inference/",
            file=sys.stderr
        )
        sys.exit(1)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read image: {input_path}", file=sys.stderr)
        sys.exit(1)

    model    = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    enhanced = model.predict(img)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, enhanced)
    sys.exit(0)


if __name__ == "__main__":
    main()
