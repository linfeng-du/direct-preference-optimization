import sys

import numpy as np
from transformers import AutoTokenizer

from tqdm import tqdm

sys.path.append('src')
from preference_datasets.persona import load_persona


tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')


def get_token_count(text):
    tokenized = tokenizer(text, add_special_tokens=False)
    return len(tokenized['input_ids'])


def main():
    n_examples = 0
    prompts = set()
    for split in ['train', 'validation', 'test', 'test_unseen']:
        persona_length = {}
        prompt_length = {}
        chosen_length = []
        rejected_length = []

        dataset = load_persona(split, prepend_persona=False)

        for prompt, prompt_data in tqdm(dataset.items(), desc=split):
            prompts.add(prompt)

            if prompt not in prompt_length:
                prompt_length[prompt] = get_token_count(prompt)

            for i, (chosen, rejected) in enumerate(prompt_data['pairs']):
                n_examples += 1

                persona = prompt_data['persona'][i]
                chosen = prompt_data['responses'][chosen]
                rejected = prompt_data['responses'][rejected]

                if persona not in persona_length:
                    persona_length[persona] = get_token_count(persona)

                chosen_length.append(get_token_count(chosen))
                rejected_length.append(get_token_count(rejected))

        persona_length = round(np.mean(list(persona_length.values())), 2)
        prompt_length = round(np.mean(list(prompt_length.values())), 2)
        chosen_length = round(np.mean(chosen_length), 2)
        rejected_length = round(np.mean(rejected_length), 2)

        print(f'Average persona length: {persona_length}')
        print(f'Average prompt length: {prompt_length}')
        print(f'Average chosen response length: {chosen_length}')
        print(f'Average rejected response length: {rejected_length}')

    print(f'Number of response pairs: {n_examples}')
    print(f'Number of unique prompts: {len(prompts)}')



if __name__ == '__main__':
    main()
