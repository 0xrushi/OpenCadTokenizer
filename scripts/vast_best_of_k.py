"""Run the best-of-k generation variant, then score it with the unchanged benchmark scorer."""
import argparse
import json
from pathlib import Path

from cadtokenizer.bench_best_of_k import run_best_of_k
from cadtokenizer.bench_score import benchmark_score
from cadtokenizer.common import read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--benchmark-config', required=True)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    config, settings = read_config(args.config), read_config(args.benchmark_config)
    predictions = run_best_of_k(config, settings, k=args.k, temperature=args.temperature)
    report = benchmark_score(config, settings, predictions_path=predictions)
    root = Path(settings['output'])
    write_json(root / 'evaluation_complete.json', {
        'status': 'complete', 'variant': f'best_of_{args.k}', 'tasks': {task: value['table_metrics'] for task, value in report['tasks'].items()}})
    print(json.dumps({'best_of_k_complete': str(root)}), flush=True)


if __name__ == '__main__':
    main()
