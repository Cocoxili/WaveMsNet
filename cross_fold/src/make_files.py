import os

if __name__ == '__main__':
    print('-------------------------------------------------------------')
    audioPath = '../../ESC-10'
    evaluatePath = '../evaluate-setup-ESC10'
    metaFile = 'ESC10.audiolist'
    classNum = 10


    if not os.path.exists(evaluatePath):
        os.makedirs(evaluatePath)

    f = open(metaFile, 'w')
    dirDict = {}
    classID = 0
    for directory in sorted(os.listdir(audioPath)):
        absDirectory = os.path.join(audioPath, directory)
        if not (os.path.isdir(absDirectory) and os.path.basename(absDirectory)[0:3].isdigit()):
            continue
        #print directory
        audioID = classID
        fileList = []
        for audioFile in sorted(os.listdir(absDirectory)):
            if not audioFile.endswith('.ogg'):
                continue
            print(directory, audioFile, audioID)

            f.writelines(directory + '/' + audioFile + '\n')
            fileList.append((directory, audioFile, audioID))
        dirDict[directory] = fileList
        classID += 1
    f.close()
    assert classID == classNum

    for idx, fold in enumerate(((5, 1), (1, 2), (2, 3), (3, 4), (4, 5))):
        validFold = fold[0]
        testFold = fold[1]
        ftrain = open(os.path.join(evaluatePath, 'fold{:d}_train.txt').format(idx), 'w')
        fvalid = open(os.path.join(evaluatePath, 'fold{:d}_valid.txt').format(idx), 'w')
        ftest = open(os.path.join(evaluatePath, 'fold{:d}_test.txt').format(idx), 'w')
        for dirKey in dirDict:
            for directory, audioFile, audioID in dirDict[dirKey]:
                fwstr = directory+'/'+audioFile+'\t'+str(audioID)+'\n'
                if audioFile[0] == str(validFold):
                    fvalid.write(fwstr)
                elif audioFile[0] == str(testFold):
                    ftest.write(fwstr)
                else:
                    ftrain.write(fwstr)
        ftrain.close()
        fvalid.close()
        ftest.close()
