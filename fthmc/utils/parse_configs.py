"""
parse_configs.py

Implements a method for parsing configuration objects from JSON file.
"""
from __future__ import absolute_import, division, print_function, annotations
import argparse
import json


def parse_configs():
    """Parse configs from JSON file."""
    parser = argparse.ArgumentParser(
        f'Normalizing flow model on 2D U(1) lattice gauge model.'
    )
    parser.add_argument('--json_file',
                        dest='json_file',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to JSON file containing configuration.')
    args = parser.parse_args()
    with open(args.json_file, 'rt') as f:
        targs = argparse.Namespace()
        targs.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=targs)

    return args
