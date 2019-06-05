import sys
import os
from pathlib import Path
import shutil
from keras.preprocessing.image import ImageDataGenerator


if len(sys.argv) != 2:
    print("Usage: python PrepareData.py <labels_file>")


def generateBreedMapping(labelsFile):
    breedMap = {}

    with open(labelsFile) as f:
        for line in f:
            if line.rstrip() != 0 and line.rstrip() != "id,breed":
                id, breed = line.rstrip().split(
                    ',')[0], line.rstrip().split(',')[1]

                if not breed in breedMap:
                    breedMap[breed] = [id]
                else:
                    breedMap[breed].append(id)

    return breedMap


def generateIDMapping(labelsFile):
    idMap = {}

    with open(labelsFile) as f:
        for line in f:
            if line.rstrip() != 0 and line.rstrip() != "id,breed":
                id, breed = line.rstrip().split(
                    ',')[0], line.rstrip().split(',')[1]

                idMap[id] = breed

    return idMap



def makeBreedDirectories(idMap, source, dest):
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


idMap = generateIDMapping(sys.argv[1])

makeBreedDirectories(idMap, 'data/train', 'train')
