import numpy as np
import  load_CIFAR10
import NearestNeighbor

Xtr,Ytr, Xte,Yte=load_CIFAR10('E:/mycode/CS231N/cifar-10-python/')
Xtr_rows=Xtr.reshape(Xtr.shape[0],32*32*3)
Xte_rows=Xte.reshape(Xte.shape[0],32*32*3)

## validation
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xte_rows[:1000, :]
Yval = Ytr[:1000]
Xte_rows = Xte_rows[1000:,:]
Ytr = Ytr[1000:]

# find the best hyperparameters k
validation_accuracies = []
for k in [1,3,5,10,20,50,100]:


    nn=NearestNeighbor()
    nn.train(Xtr_rows , Ytr)
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = np.mean(Yval_predict == Yval)
    print 'accuracy:%f'% (acc,)

    # keep track of what works on the validation set
    validation_accuracies.append((k,acc))
Yte_predict=nn.predict(Xte_rows)

#print 'accuracy:%f'%(np.mean(Yte_predict==Yte))


