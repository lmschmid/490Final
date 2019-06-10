import sys
import os
import shutil
import numpy as np
from pathlib import Path


if len(sys.argv) != 2:
    print("Usage: python PrepareData.py <labels_file>")


def generateIDMapping(labelsFile):
    idMap = {}

    with open(labelsFile) as f:
        for line in f:
            if line.rstrip() != 0 and line.rstrip() != "id,breed":
                id, breed = line.rstrip().split(
                    ',')[0], line.rstrip().split(',')[1]

                idMap[id] = breed

    return idMap


def makeTrainDirectories(idMap, source, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)

    for filename in os.listdir(source):
        id = filename.split('.')[0]
        breed = idMap[id]

        srcPath = source+"/"+filename
        destPath = dest+"/"+breed+"/"+filename

        path = Path(dest+"/"+breed)
        path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(srcPath, destPath)


def makeValidationDirectories():
    for dirname in os.listdir('train'):
        samples = np.asarray(os.listdir('train/'+dirname))
        valSamples = np.random.choice(samples, len(samples)//4, replace=False)

        if os.path.exists('validation/'+dirname):
            shutil.rmtree('validation/'+dirname)
        for sample in valSamples:
            path = Path('validation/'+dirname)
            path.mkdir(parents=True, exist_ok=True)

            shutil.move('train/'+dirname+'/'+sample,
                        'validation/'+dirname+'/'+sample)


def makeTestDirectory(source, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)

    dest = dest + "/images"
    path = Path(dest)
    path.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(source):
        srcPath = source+"/"+filename
        destPath = dest+"/"+filename
        shutil.copyfile(srcPath, destPath)


idMap = generateIDMapping(sys.argv[1])

makeTrainDirectories(idMap, 'data/train', 'train')
makeValidationDirectories()
makeTestDirectory('data/test', 'test')
