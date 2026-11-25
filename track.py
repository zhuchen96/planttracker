"""Predict the daughters of each mother in one consecutive pair of prepared frames."""
import argparse
import fcntl
import json
import pickle
from tempfile import TemporaryDirectory

from pathlib import Path
from release_io import ROOT, digest, independent, new_directory, sequence_name
import inference


def run(frames, output, sequence, time, models):
    seq = sequence_name(sequence)
    if time < 0:
        raise ValueError('Time must be nonnegative')
    for path in (frames, models, ROOT / 'models'):
        independent(output, path)
    inputs = {}
    for t in (time, time + 1):
        for name in ('seg.tif', 'props.pkl'):
            rel = Path(seq) / f'T{t}' / name
            inputs[str(rel)] = digest(frames / rel)
    for name in ('set_ranker.pkl', 'final_ranker.pkl'):
        inputs[name] = digest(models / name)
    folder = output / 'predictions' / seq / f'T{time}'
    folder.parent.mkdir(parents=True, exist_ok=True)
    key = json.dumps(inputs, sort_keys=True)
    with (folder.parent / f'.T{time}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if folder.exists():
            if (folder / 'inputs.json').read_text() != key:
                raise ValueError('Input frames or checkpoints changed. Use a fresh output directory.')
            if not (folder / 'full.json').is_file():
                raise ValueError('Completed prediction file missing. Use a fresh output directory.')
            return folder / 'full.json'
        with new_directory(folder) as dest, TemporaryDirectory(dir=output) as tmp:
            work = Path(tmp)
            inference.MODELS = models
            inference.DATA = inference.W.R = inference.wgen.G = frames
            inference.OUTPUT = work / 'predictions'
            inference.wgen.OUT = work / 'geometry'
            inference.wgen.OUT.mkdir()
            inference.run((seq, time))
            pair = inference.OUTPUT / seq / f'T{time}'
            pred = pickle.loads((pair / 'full.pkl').read_bytes())
            result = {str(m): list(map(int, ds)) for m, ds in pred.items()}
            for artifact in pair.iterdir():
                artifact.rename(dest / artifact.name)
            (dest / 'full.json').write_text(json.dumps(result, sort_keys=True) + '\n')
            (dest / 'inputs.json').write_text(key)
    return folder / 'full.json'


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('frames', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('sequence')
    cli.add_argument('time', type=int)
    cli.add_argument('--models', type=Path, default=ROOT / 'models')
    print(run(**vars(cli.parse_args())))
