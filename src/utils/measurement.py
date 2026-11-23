"""Read NashConv results, including execution settings saved separately by older runs."""
import json
from pathlib import Path


def read_measurement(path):
    result = json.loads(Path(path).read_text())
    if 'backend' not in result:
        selected = result.get('adv', {}).get('selected', {})
        if 'run_dir' in selected:
            config = Path(selected['run_dir']) / 'config.json'
            if config.is_file():
                args = json.loads(config.read_text())
                result.update(backend=args.get('env_backend', 'mpe'), device=args['device'], threads=args['threads'])
    return result
