import numpy as np
import  load_CIFAR10
import NearestNeighbor

# download the dataset of training and testing

Xtr,Ytr, Xte,Yte=load_CIFAR10('E:/mycode/CS231N/cifar-10-python/')
Xtr_rows=Xtr.reshape(Xtr.shape[0],32*32*3)
Xte_rows=Xte.reshape(Xte.shape[0],32*32*3)

# classsification with knn

nn=NearestNeighbor()
nn.train(Xtr_rows , Ytr)
Yte_predict=nn.predict(Xte_rows)
print 'accuracy:%f'%(np.mean(Yte_predict==Yte))
