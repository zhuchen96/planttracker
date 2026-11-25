"""Prepare one ZYX frame already sampled on the tracker's working grid."""
import argparse
import pickle
import shutil
import numpy as np
import tifffile
from pathlib import Path
from release_io import independent, new_directory, sequence_name
from prep import frame_props


def prepare(frames, sequence, time, raw, mask):
    if time < 0:
        raise ValueError('Time must be nonnegative')
    destination = frames / sequence_name(sequence) / f'T{time}'
    for path in (raw, mask):
        independent(destination, path)
    with new_directory(destination) as folder:
        seg = tifffile.imread(mask)
        img = tifffile.imread(raw)
        if img.ndim != 3 or img.shape != seg.shape:
            raise ValueError('Raw and mask must be equally shaped ZYX volumes')
        if seg.dtype.kind not in 'ui' or seg.min() < 0 or not 0 < seg.max() < 100000:
            raise ValueError('Masks require integer cell IDs from 1 through 99999 and background 0')
        if img.dtype.kind not in 'uif' or not np.isfinite(img).all() or img.min() < 0:
            raise ValueError('Membrane intensities must be finite, nonnegative numbers')
        shutil.copyfile(raw, folder / 'image.tif')
        if seg.dtype == np.uint64:
            tifffile.imwrite(folder / 'seg.tif', seg.astype(np.uint32), photometric='minisblack')
        else:
            shutil.copyfile(mask, folder / 'seg.tif')
        del img, seg
        props = frame_props(time, folder / 'seg.tif', folder / 'image.tif')
        (folder / 'props.pkl').write_bytes(pickle.dumps(props, protocol=4))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('frames', type=Path)
    cli.add_argument('sequence')
    cli.add_argument('time', type=int)
    cli.add_argument('raw', type=Path)
    cli.add_argument('mask', type=Path)
    prepare(**vars(cli.parse_args()))
