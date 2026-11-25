"""Fit both lineage scoring checkpoints from annotated, prepared frame pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import pickle

import numpy as np
import tifffile
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path
from release_io import ROOT, digest, independent, new_directory, sequence_name
from em import fit_warp
from perturb import PerturbedWarp
from ranker import TwoStage
from setmatch import SET_FEATS, build_sets, competition_features
from training_registration import estimate

VARIANTS = ('estimated', 'annotation', 'pert1', 'pert2', 'pert3')


def properties(paths):
    return [pickle.loads((path / 'props.pkl').read_bytes()) for path in paths]


def validate(manifest, frames):
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError('Cross-fitting requires at least two annotated transitions')
    seen, jobs = set(), []
    for entry in entries:
        seq = sequence_name(entry['sequence'])
        a, b = entry['earlier'], entry['later']
        if type(a) is not int or type(b) is not int or a < 0 or b != a + 1 or (seq, a) in seen:
            raise ValueError(f'Invalid or duplicate interval {seq} {a} {b}')
        seen.add((seq, a))
        paths = [frames / seq / f'T{t}' for t in (a, b)]
        A, B = properties(paths)
        for path, props in zip(paths, (A, B)):
            for filename in ('seg.tif', 'image.tif'):
                with tifffile.TiffFile(path / filename) as tif:
                    if tuple(tif.series[0].shape) != tuple(props['shape']):
                        raise ValueError(f'Frame shape mismatch in {path}')
        labels = json.loads((manifest.parent / entry['families']).read_text())
        fam = {int(m): ds for m, ds in labels.items()}
        if len(labels) != len(fam) or any(str(int(m)) != m for m in labels):
            raise ValueError('Use canonical positive integer strings for mother IDs')
        available_mothers, available_daughters = set(A['labels']), set(B['labels'])
        used = set()
        for m, ds in fam.items():
            if m not in available_mothers:
                raise ValueError(f'Unknown mother {m}')
            if not isinstance(ds, list) or not ds or any(type(d) is not int for d in ds):
                raise ValueError(f'Expected daughter ID list for mother {m}')
            if len(set(ds)) != len(ds) or used.intersection(ds) or not set(ds).issubset(available_daughters):
                raise ValueError(f'Invalid or reused daughter for mother {m}')
            used.update(ds)
        if len(fam) < 4 or np.linalg.matrix_rank(A['cen'][list(fam)] - A['cen'][list(fam)].mean(0)) < 3:
            raise ValueError(f'{seq} T{a}: need four noncoplanar annotated mother centers')
        daughter_centers = np.array([np.average(B['cen'][ds], axis=0, weights=B['vol'][ds])
                                     for ds in fam.values()])
        if np.linalg.matrix_rank(daughter_centers - daughter_centers.mean(0)) < 3:
            raise ValueError(f'{seq} T{a}: daughter-group centers are coplanar')
        recipe = entry.get('registration')
        if recipe is not None and (not isinstance(recipe, dict)
                or recipe.get('route') not in ('similarity', 'anisotropic')
                or type(recipe.get('candidate')) is not int or not 0 <= recipe['candidate'] < 60):
            raise ValueError('Expected a registration route and a zero-based candidate index below 60')
        jobs.append((paths, fam, recipe))
    return entries, jobs


def make_tables(task):
    index, (paths, fam, recipe), workspace = task
    props = properties(paths)
    masks = [tifffile.imread(p / 'seg.tif') for p in paths]
    images = [tifffile.imread(p / 'image.tif') for p in paths]
    print(f'Pair {index}: estimating training registration', flush=True)
    estimated = estimate(masks, images, props, recipe)
    del images
    annotation = fit_warp(*props, fam, smoothing=100)
    warps = [estimated, annotation]
    for k, (xy, z) in enumerate(((2, 1), (3.5, 1.5), (5, 2.5)), 1):
        warps.append(PerturbedWarp(annotation, masks[1].shape, xy, z, seed=100*k+index))
    for name, warp in zip(VARIANTS, warps):
        rows, X, y, _, _ = build_sets(index, *masks, *props, warp, gt_fam=fam)
        if len(np.unique(y[y >= 0])) != 2:
            raise ValueError(f'Pair {index}, {name}: need both positive and negative annotated proposals')
        with (workspace / f'{name}-{index}.pkl').open('wb') as handle:
            pickle.dump(dict(rows=rows, X=X, y=y), handle, protocol=4)
        print(f'Pair {index}, {name}: {len(y)} proposals, {sum(y == 1)} positives', flush=True)


def fit_classifier(arrays, labels):
    X, y = np.concatenate(arrays), np.concatenate(labels)
    keep = y >= 0
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=.05, max_leaf_nodes=31,
                    l2_regularization=1., random_state=0).fit(X[keep], y[keep])


def fit_models(jobs, workspace):
    def load(name, i):
        with (workspace / f'{name}-{i}.pkl').open('rb') as handle:
            return pickle.load(handle)
    ids = list(range(len(jobs)))
    props = {i: properties(jobs[i][0]) for i in ids}
    tables = [load('estimated', i) for i in ids]
    def first(indices):
        return fit_classifier([tables[i]['X'] for i in indices], [tables[i]['y'] for i in indices])
    stage1 = first(ids)
    competition = []
    for i, table in enumerate(tables):
        other = first([j for j in ids if j != i])
        scores = other.predict_proba(table['X'])[:, 1]
        competition.append(competition_features(table['rows'], table['X'], scores, *props[i]))
    stage2 = fit_classifier(competition, [table['y'] for table in tables])
    early = dict(stage1=stage1, stage2=stage2, features=SET_FEATS)
    del competition
    augmented = list(zip(tables, ids))
    augmented.extend((load(name, i), i) for name in VARIANTS[1:] for i in ids)
    ranker = TwoStage(augmented, props)
    clf = fit_classifier([t['X'][t['y'] >= 0] for t, _ in augmented],
                         [t['y'][t['y'] >= 0] for t, _ in augmented])
    return early, dict(lambdarank=ranker, classifier=clf)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('frames', 'manifest', 'output'):
        cli.add_argument(arg, type=Path)
    cli.add_argument('--workers', type=int, default=1, help='Number of frame pairs prepared concurrently')
    cli.add_argument('--keep-tables', action='store_true', help='Retain candidate tables for audits')
    args = cli.parse_args()
    for path in (args.frames, args.manifest, ROOT / 'models'):
        independent(args.output, path)
    entries, jobs = validate(args.manifest, args.frames)
    if args.workers < 1:
        cli.error('workers must be positive')
    with new_directory(args.output) as out:
        workspace = out / 'tables'
        workspace.mkdir()
        tasks = [(i, job, workspace) for i, job in enumerate(jobs)]
        with ProcessPoolExecutor(min(args.workers, len(jobs))) as pool:
            list(pool.map(make_tables, tasks))
        print('Fitting classifiers and grouped rankers', flush=True)
        for name, model in zip(('set_ranker.pkl','final_ranker.pkl'), fit_models(jobs, workspace)):
            with (out / name).open('wb') as handle:
                pickle.dump(model, handle)
        record = dict(transitions=entries, variants=VARIANTS,
                      families_per_transition=[len(job[1]) for job in jobs],
                      sha256={p.name: digest(p) for p in out.glob('*.pkl')})
        (out / 'training.json').write_text(json.dumps(record, indent=2) + '\n')
        if not args.keep_tables:
            import shutil
            shutil.rmtree(workspace)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
