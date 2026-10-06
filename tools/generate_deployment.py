"""Generate matching Python config, firmware frontend tables and model bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.deployment_contract import CONTRACT, CONTRACT_SHA256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firmware-root', type=Path, help='Optional local LED_TEST checkout to synchronize.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--label-map', type=Path)
    parser.add_argument('--feature', choices=('MFCC', 'LOGMEL', 'PCEN'))
    parser.add_argument('--source-commit', default='unknown')
    parser.add_argument('--check', action='store_true', help='Verify checkout synchronization without writing.')
    args = parser.parse_args()
    synchronized = ['shared/deployment_contract.json', 'shared/deployment_contract.py',
                    'tools/generate_mfcc_frontend_tables.py', 'tools/generate_model_bundle.py']
    if args.firmware_root:
        target = args.firmware_root.resolve()
        if not (target / 'CMakeLists.txt').is_file():
            parser.error('--firmware-root must be an existing LED_TEST checkout')
        differences = [name for name in synchronized if not (target / name).exists()
                       or (target / name).read_bytes() != (ROOT / name).read_bytes()]
        if args.check:
            if differences:
                print(json.dumps({'synchronized': False, 'different_files': differences}))
                return 1
        else:
            for name in synchronized:
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target / name)
    if args.check:
        print(json.dumps({'synchronized': True, 'frontend_contract_sha256': CONTRACT_SHA256}))
        return 0
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    python_config = '# Generated; canonical source is shared/deployment_contract.json.\n'
    python_config += 'CONTRACT = ' + repr(CONTRACT) + '\nCONTRACT_SHA256 = ' + repr(CONTRACT_SHA256) + '\n'
    (out / 'frontend_contract.py').write_text(python_config, encoding='utf-8')
    (out / 'label_map.json').write_text(json.dumps({x: i for i, x in enumerate(CONTRACT['labels'])}, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'tools/generate_mfcc_frontend_tables.py'),
                    '--output-dir', str(out / 'frontend')], check=True)
    if args.model:
        command = [sys.executable, str(ROOT / 'tools/generate_model_bundle.py'),
                   '--tflite', str(args.model.resolve()), '--output-dir', str(out / 'model'),
                   '--source-commit', args.source_commit]
        for option, value in [('metadata', args.metadata), ('label-map', args.label_map), ('feature', args.feature)]:
            if value is not None:
                command.extend(['--' + option, str(value.resolve()) if isinstance(value, Path) else value])
        subprocess.run(command, check=True)
        bundle = json.loads((out / 'model/model_bundle.json').read_text(encoding='utf-8'))
        shutil.copyfile(args.model.resolve(), out / 'model/model.tflite')
        shutil.copyfile(Path(bundle['source_metadata']), out / 'model/metadata.json')
        shutil.copyfile(out / 'label_map.json', out / 'model/label_map.json')
    # Deployment carries the same authority and generators, including remote use.
    for name in synchronized:
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in out.rglob('*') if p.is_file() and p.name != 'deployment_manifest.json'}
    try:
        source_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        source_dirty = None
    manifest = {'schema_version': 1, 'source_dirty': source_dirty, 'frontend_contract_sha256': CONTRACT_SHA256,
                'source_commit': args.source_commit, 'artifact_sha256': files}
    (out / 'deployment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(out), 'frontend_contract_sha256': CONTRACT_SHA256}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
