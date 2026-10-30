import argparse
import os
from pathlib import Path


def check_dataset_structure(data_dir: Path):
    expected_subdirs = ['train', 'val', 'test']
    missing = []

    for subdir in expected_subdirs:
        path = data_dir / subdir
        if not path.exists() or not path.is_dir():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            'Missing expected dataset directories:\n' + '\n'.join(missing)
        )

    print(f'Found dataset structure under {data_dir}')
    for subdir in expected_subdirs:
        normal_dir = data_dir / subdir / 'NORMAL'
        pneumonia_dir = data_dir / subdir / 'PNEUMONIA'
        print(f'- {subdir}:')
        print(f'  - NORMAL: {normal_dir.exists()}')
        print(f'  - PNEUMONIA: {pneumonia_dir.exists()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check local chest X-ray dataset structure.')
    parser.add_argument('--data-dir', '-d', default='data', help='Path to the local dataset root folder')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f'Local data folder not found: {data_dir}')

    check_dataset_structure(data_dir)
