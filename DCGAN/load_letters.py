import sys
sys.path.append('..')

import numpy as np
import os
import cv2


data_dir = 'Letters_v2/'

def letters():
    list_letters = os.listdir(data_dir)
    trX = cv2.imread(os.path.join(data_dir, 'a/11.png'), 0)
    trX = np.reshape(trX, (1, 28*28))
    
    trY = [0]
    
    teX = cv2.imread(os.path.join(data_dir, 'a/11.png'), 0)
    teX = np.reshape(teX, (1, 28*28))
    
    teY = [0]
    
    for i in range(len(list_letters)):
        path = data_dir + list_letters[i] + '/'
        img = os.listdir(path) 
        for j in range(0, len(img)):  
            letter = cv2.imread(os.path.join(path, img[j]), 0)
            letter = np.reshape(letter, (1, 28*28))
            trX = np.concatenate((trX, letter), axis=0 )
            trY.append(i)
            if j> 2000:
                teX = np.concatenate((trX, letter), axis=0 )
                teY.append(i)
        print('ok for letter : {}'.format(list_letters[i]))
    
    trX = trX.astype(float)
    teX = teX.astype(float)
                
    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY
    
    
def letters_set():
    trX, teX, trY, teY = letters()

    train_inds = np.arange(len(trX))
    np.random.shuffle(train_inds)
    trX = trX[train_inds]
    trY = trY[train_inds]
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY


