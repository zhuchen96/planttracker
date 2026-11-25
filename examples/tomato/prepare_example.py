"""Fetch or use the original MorphoGraphX archive and prepare the two tomato frames."""
import argparse
import csv
import json
import shutil
import struct
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
import pickle

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from release_io import digest, new_directory
from prep import frame_props

URL = 'https://nbicloud.sharepoint.com/:u:/s/repository/001/EaYNRd_omBdCmyzOWYiEUIUB067xSPc6-knnJiGlCVfNoA?download=1'
SHA256 = '9f3a6a2c9c7125ba18bd175faf83f7ee2c8b569a563514f0f9466df4b51db0fa'
GRID = np.array([1.2, .6, .6])


def mesh(path):
    with path.open('rb') as f:
        ver=f.read(9)
        if ver==b'MGXM 2.0 ':
            assert f.read(8)==b'\x04\x00\x00\x00MGXM'
            count,=struct.unpack('<I',f.read(4))
            dtype=np.dtype(dict(names=['id','xyz','label','signal'],formats=['<i4',('<f8',3),'<i4','<f4'],offsets=[40,0,32,28],itemsize=46))
            vertices=np.fromfile(f,dtype,count=count)
            assert np.array_equal(vertices['id'],np.arange(count)),path
            return vertices
        assert ver in (b'MGXM 1.2 ',b'MGXM 1.3 '),ver
        iscells,scale,transform,lo,hi,n=struct.unpack('<3BffI',f.read(15))
        assert (iscells,scale,transform)==(0,1,0)
        f.read(n)
        header,count,extra,edge=struct.unpack('<4I',f.read(16))
        f.read(header)
        assert extra>=8
        dtype=np.dtype(dict(names=['id','xyz','label','signal'],formats=['<u4',('<f4',3),'<i4','<f4'],offsets=[0,4,16,20],itemsize=16+extra))
        vertices=np.fromfile(f,dtype,count=count)
        assert len(vertices)==count and len(np.unique(vertices['id']))==count
        if ver==b'MGXM 1.2 ':
            vertices['signal']=lo+vertices['signal']*(hi-lo)
        return vertices

def frame(path, t, out):
    out.mkdir(parents=True)
    v=mesh(path)
    xyz=v['xyz'][:,::-1].astype('d')
    valid=v['label']>0
    labs=v['label'][valid]
    assert 0<labs.max()<65536
    resolution_scale=1.
    spacing=GRID*resolution_scale
    origin=np.floor(xyz.min(0)/spacing).astype(int)-3
    inds=np.rint(xyz/spacing-origin).astype(int)
    shape=inds.max(0)+4
    boundary=np.zeros(shape,dtype=bool)
    boundary[tuple(inds.T)]=True
    support=ndimage.distance_transform_edt(~boundary,sampling=spacing)<=2.45*resolution_scale
    points=np.argwhere(support)
    seg=np.zeros(shape,'uint16'); img=np.zeros(shape,'uint16')
    alltree=cKDTree(xyz); labtree=cKDTree(xyz[valid])
    signal=np.clip(v['signal'],0,65535)
    for i in range(0,len(points),100000):
        p=points[i:i+100000]; q=(p+origin)*spacing
        dist,j=alltree.query(q,distance_upper_bound=1.5*resolution_scale)
        keep=np.isfinite(dist)
        coords=tuple(p[keep].T)
        img[coords]=signal[j[keep]]
        dist2,k=labtree.query(q[keep],distance_upper_bound=2.4*resolution_scale)
        kvalid=np.isfinite(dist2)
        seg[tuple(p[keep][kvalid].T)]=labs[k[kvalid]]
    tifffile.imwrite(out/'seg.tif',seg,photometric='minisblack',compression='zlib',compressionargs={'level':1})
    tifffile.imwrite(out/'image.tif',img,photometric='minisblack',compression='zlib',compressionargs={'level':1})
    props=frame_props(t,out/'seg.tif',out/'image.tif')
    props['lay']={int(i):1 for i in props['labels']}
    (out/'props.pkl').write_bytes(pickle.dumps(props,protocol=4))
    meta=dict(source=path.name,shape=shape.tolist(),native_ids=np.unique(labs).tolist(),retained_labels=len(props['labels']),prepared_spacing=list(spacing),crop_origin=origin.tolist(),vertices=len(v),radius_um=1.5*resolution_scale)
    (out/'meta.json').write_text(json.dumps(meta))
    print('tomato',t,shape,len(meta['native_ids']),meta['retained_labels'],flush=True)

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('destination', type=Path)
    cli.add_argument('--archive', type=Path, help='An existing, unmodified tomato archive')
    args = cli.parse_args()
    with new_directory(args.destination) as out, tempfile.TemporaryDirectory(dir=out) as scratch:
        temp = Path(scratch)
        archive = args.archive
        if archive is None:
            archive = temp / 'tomato.tar.gz'
            with urllib.request.urlopen(URL, timeout=180) as source, archive.open('wb') as target:
                shutil.copyfileobj(source, target)
        if digest(archive) != SHA256:
            raise ValueError('Unexpected MorphoGraphX archive checksum')
        meshes = [f'meristem_T{t}_fine.mgxm' for t in (0, 1)]
        parents = 'meristem_T0-T1_parents.csv'
        with tarfile.open(archive) as tar:
            for name in [*meshes, parents]:
                member = tar.getmember('Kierzkowski_2012/' + name)
                if not member.isfile():
                    raise ValueError(name)
                with tar.extractfile(member) as src, (temp / name).open('wb') as dest:
                    shutil.copyfileobj(src, dest)
        for t, name in enumerate(meshes):
            frame(temp / name, t, out / 'frames' / 'tomato' / f'T{t}')
        gold = {}
        with (temp / parents).open() as handle:
            for row in list(csv.reader(handle))[1:]:
                if row:
                    d, m = map(int, row[:2])
                    if d > 0 and m > 0:
                        gold.setdefault(str(m), []).append(d)
        (out / 'families.json').write_text(json.dumps(gold, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
