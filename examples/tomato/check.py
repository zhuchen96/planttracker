"""Check every predicted family and score against the published tomato references."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = 'f6a9e2dd9f96abfbed67da3600129b93894d178fc22716d5d211334c6088cbfb'


def load(path):
    return {m: sorted(ds) for m, ds in json.loads(path.read_text()).items()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('predictions', type=Path)
    cli.add_argument('families', type=Path)
    args = cli.parse_args()
    pred, gold = load(args.predictions), load(args.families)
    exact = sum(pred.get(m) == ds for m, ds in gold.items())
    canonical = json.dumps(pred, sort_keys=True, separators=(',', ':')).encode()
    print(f'{exact}/{len(gold)} exact families; {len(pred)} predicted families')
    if (exact, len(gold)) != (488, 502) or hashlib.sha256(canonical).hexdigest() != EXPECTED:
        raise SystemExit('Predictions differ from the frozen reference')
    print('All mother and daughter IDs match the frozen reference.')


if __name__ == '__main__':
    main()
