#!/usr/bin/env python3
"""
Zenodo Packaging Script for SAR Water Detection v2
==================================================
Creates a reproducible bundle for Zenodo submission:
  - Trained model weights
  - Source code
  - Dockerfile + requirements
  - Sample chips (first 5)
  - Paper PDF
  - CITATION.cff
  - README

Usage:
    python package_zenodo.py --output zenodo_bundle.zip
"""
import os
import sys
import zipfile
import argparse
import hashlib
from pathlib import Path
from datetime import datetime


# File manifest - what goes into the Zenodo bundle
MANIFEST = {
    'README.md': 'README.md',
    'CITATION.cff': 'CITATION.cff',
    'src/': 'src/',
    'reproducibility/': 'reproducibility/',
    'scripts/': 'scripts/',
    'docs/': 'docs/',
}


def file_hash(filepath: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def create_citation_cff(bundle_dir: Path, version: str = 'v2.0'):
    """Create a CITATION.cff file."""
    content = f"""cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "Pune SAR Water Detection v2"
version: {version}
date-released: "{datetime.now().strftime('%Y-%m-%d')}"
authors:
  - family-names: "Parekh"
    given-names: "Neeraj"
license: MIT
repository-code: "https://github.com/neeraj-parekh/pune-sar-water-detection"
type: software
keywords:
  - SAR
  - water detection
  - Sentinel-1
  - deep learning
  - U-Net
  - topology preservation
"""
    with open(bundle_dir / 'CITATION.cff', 'w') as f:
        f.write(content)


def create_zenodo_readme(bundle_dir: Path, version: str = 'v2.0'):
    """Create a top-level README for the Zenodo bundle."""
    content = f"""# Pune SAR Water Detection — Zenodo Bundle ({version})

## Contents

- `src/` — Source code (training, inference, losses, model, dataset)
- `reproducibility/` — Dockerfile, pinned requirements
- `scripts/` — Baseline comparisons (Otsu, MNDWI)
- `docs/` — Bug inventory, verification report, run instructions
- `models/` — Trained model weights (best_v2.pth)
- `sample_chips/` — 5 sample chips for quick testing

## Reproducibility

1. Build Docker image: `docker build -t sar-water-v2 reproducibility/`
2. Run training: `docker run -v $(pwd)/data:/data sar-water-v2 python src/train.py --fresh --fold 0`
3. Run inference: `docker run -v $(pwd)/data:/data sar-water-v2 python src/inference.py --model models/best_v2.pth --input /data/test.tif --output /data/pred.tif`

## Citation

See CITATION.cff

## License

MIT License. See LICENSE file.
"""
    with open(bundle_dir / 'README.md', 'w') as f:
        f.write(content)


def package_zenodo(output_path: str, src_dir: str, version: str = 'v2.0',
                   include_models: bool = True) -> dict:
    """Create the Zenodo bundle."""
    src_dir = Path(src_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files_added = []
    total_size = 0

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Add source code
        for src_rel in ['src', 'reproducibility', 'scripts', 'docs']:
            src_path = src_dir / src_rel
            if not src_path.exists():
                continue
            for f in src_path.rglob('*'):
                if f.is_file() and '__pycache__' not in str(f):
                    arcname = f.relative_to(src_dir)
                    zf.write(f, arcname)
                    files_added.append(str(arcname))
                    total_size += f.stat().st_size

        # Add models (if requested)
        if include_models:
            models_dir = src_dir / 'models'
            if models_dir.exists():
                for f in models_dir.glob('*.pth'):
                    arcname = f.relative_to(src_dir)
                    zf.write(f, arcname)
                    files_added.append(str(arcname))
                    total_size += f.stat().st_size

        # Create CITATION.cff and README inside the zip
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            create_citation_cff(tmp, version)
            zf.write(tmp / 'CITATION.cff', 'CITATION.cff')
            create_zenodo_readme(tmp, version)
            zf.write(tmp / 'README.md', 'README.md')
            files_added.append('CITATION.cff')
            files_added.append('README.md')

    # Compute bundle hash
    bundle_hash = file_hash(output_path)
    bundle_size_mb = output_path.stat().st_size / 1e6

    manifest = {
        'bundle': str(output_path),
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'n_files': len(files_added),
        'raw_size_mb': total_size / 1e6,
        'bundle_size_mb': bundle_size_mb,
        'sha256': bundle_hash,
        'files': files_added,
    }

    # Save manifest next to bundle
    manifest_path = output_path.with_suffix('.manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='zenodo_bundle.zip', help='Output zip path')
    parser.add_argument('--src-dir', default='.', help='Source directory')
    parser.add_argument('--version', default='v2.0', help='Version string')
    parser.add_argument('--no-models', action='store_true', help='Skip model weights')
    args = parser.parse_args()

    manifest = package_zenodo(args.output, args.src_dir, args.version, not args.no_models)

    print(f"\n=== Zenodo Bundle Created ===")
    print(f"Path:      {manifest['bundle']}")
    print(f"Size:      {manifest['bundle_size_mb']:.1f} MB (raw: {manifest['raw_size_mb']:.1f} MB)")
    print(f"Files:     {manifest['n_files']}")
    print(f"SHA256:    {manifest['sha256']}")
    print(f"Manifest:  {manifest['bundle'].replace('.zip', '.manifest.json')}")
