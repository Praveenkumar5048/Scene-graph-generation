#!/usr/bin/env python3
"""Simple wrapper to run inference using the repo's `inference.py`.

Usage example:
  python run_inference_simple.py --input test_image.png --resume ckpt/checkpoint0149.pth --output inference_result.png

The script imports `inference.get_args_parser` to obtain defaults, sets the input/output/resume/device
fields and calls `inference.main(args)`.
"""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description="Run RelTR inference (simple wrapper)")
    # If you prefer not to pass CLI args, these defaults will be used.
    parser.add_argument('--input', '-i', default='test_images/image5.png', help='Path to input image (default: test_image.png)')
    parser.add_argument('--output', '-o', default='output/image5.png', help='Path to save output image (default: inference_result.png)')
    parser.add_argument('--resume', '-r', default='ckpt/checkpoint0149.pth', help='Path to checkpoint')
    parser.add_argument('--device', '-d', default='cpu', help='Device to run on (cpu or cuda)')
    args = parser.parse_args()

    try:
        inf = importlib.import_module('inference')
    except Exception as e:
        print('Failed to import local inference.py:', e)
        sys.exit(2)

    # get the full parser from inference to pick up defaults
    try:
        inf_parser = inf.get_args_parser()
    except AttributeError:
        print('inference.get_args_parser not found. Are you running from repo root?')
        sys.exit(2)

    inf_args = inf_parser.parse_args([])  # use defaults

    # override the values we care about
    inf_args.img_path = args.input
    inf_args.output = args.output
    inf_args.resume = args.resume
    inf_args.device = args.device

    # If the user didn't explicitly request a graph image, default to saving
    # a graph image next to the visualization output. We do NOT auto-create a
    # scene-graph JSON unless explicitly requested.
    if not getattr(inf_args, 'graph_image', ''):
        if inf_args.output:
            inf_args.graph_image = inf_args.output.rsplit('.', 1)[0] + '_graph.png'
        else:
            inf_args.graph_image = 'scene_graph.png'

    # call the inference main
    try:
        inf.main(inf_args)
    except Exception as e:
        print('Error while running inference:', e)
        raise


if __name__ == '__main__':
    main()
