import os


def checkPath(saveRoot):
    if not os.path.exists(saveRoot):
        os.makedirs(saveRoot)


def createSavePath(saveRoot):
    checkPath(saveRoot)
    idx = len(os.listdir(saveRoot))
    savePath = f"{saveRoot}/exp{idx + 1}"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    return savePath

