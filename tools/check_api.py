"""Verify advertised model and one real non-streaming inference response."""
import argparse
import json
import time
import urllib.request

MODEL = 'DeepSeek-v4.1-Flash-EXL3'


def check(base):
    base = base.rstrip('/')
    with urllib.request.urlopen(base + '/models', timeout=10) as response:
        models = json.load(response)['data']
    if MODEL not in [entry['id'] for entry in models]:
        raise ValueError('Expected model is not advertised: ' + repr([entry['id'] for entry in models]))
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'What is 17 times 19? Reply with only the integer.'}],
            'temperature': 0, 'max_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=90) as response:
        result = json.load(response)
    choice = result['choices'][0]
    content = choice['message'].get('content', '').strip()
    if content != '323' or choice['finish_reason'] != 'stop':
        raise ValueError('Inference did not pass the arithmetic check: ' + repr(choice))
    return {'model': MODEL, 'reply': content, 'elapsed_seconds': round(time.monotonic() - start, 3), 'passed': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8888/v1', help='API base including /v1')
    args = parser.parse_args()
    print(json.dumps(check(args.base_url), indent=2))
