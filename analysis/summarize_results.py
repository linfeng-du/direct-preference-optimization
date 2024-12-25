import json
from pathlib import Path

import numpy as np


def parse_results_from_log(log_path):
    results = {
        'training': [],
        'inference': [],
        'validation_reward': [],
        'validation_ranking': [],
        'test_reward': [],
        'test_ranking': [],
        'test_unseen_reward': [],
        'test_unseen_ranking': []
    }
    validation_results = {}
    with open(log_path, 'r') as file:
        for line in file:
            if 'Training throughput:' in line:
                results['training'].append(line.split(' ')[-2])
            elif 'Inference throughput:' in line:
                results['inference'].append(line.split(' ')[-2])
            elif 'validation results after' in line:
                validation_n_examples = line.split(' ')[-3]
                validation_result = ''.join(file.readline() for _ in range(10))
                validation_result = json.loads(validation_result)
                validation_results[validation_n_examples] = validation_result
            elif 'test results after' in line:
                test_n_examples = line.split(' ')[-3]
                test_result = ''.join(file.readline() for _ in range(10))
                test_result = json.loads(test_result)
                results['test_reward'].append(test_result['reward_test/accuracy'])
                results['test_ranking'].append(test_result['logp_test/accuracy'])
            elif 'test_unseen results after' in line:
                test_n_examples = line.split(' ')[-3]
                test_unseen_result = ''.join(file.readline() for _ in range(10))
                test_unseen_result = json.loads(test_unseen_result)
                results['test_unseen_reward'].append(test_unseen_result['reward_test_unseen/accuracy'])
                results['test_unseen_ranking'].append(test_unseen_result['logp_test_unseen/accuracy'])

    results['validation_reward'].append(validation_results[test_n_examples]['reward_validation/accuracy'])
    results['validation_ranking'].append(validation_results[test_n_examples]['logp_validation/accuracy'])

    results = {
        key: [
            float(v)
            if key in {'training', 'inference'} else
            float(v) * 100 for v in value
        ]
        for key, value in results.items()
    }
    return results


settings = set()

for log_path in Path('./outputs').rglob('*train.log'):
    setting = log_path.parent.parent
    if setting not in settings:
        results = {
            'training': [],
            'inference': [],
            'validation_reward': [],
            'validation_ranking': [],
            'test_reward': [],
            'test_ranking': [],
            'test_unseen_reward': [],
            'test_unseen_ranking': []
        }
        for log_path in Path(setting).rglob('*train.log'):
            result = parse_results_from_log(log_path)
            for key, value in results.items():
                value.extend(result[key])

        print('=' * 40, setting, '=' * 40)
        results = {
            key:
            f'{round(np.mean(value), 2)}'
            if key in {'training', 'inference'} else
            f'{round(np.mean(value), 2)} \u00b1 {round(np.std(value), 2)}'
            for key, value in results.items()
        }
        print(json.dumps(results, indent=4, ensure_ascii=False))

        settings.add(setting)
