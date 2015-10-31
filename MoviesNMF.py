from Movies import*
from sklearn.decomposition import NMF




rdata = sp.loadtxt('u.data',int)
v = sp.array([])
m = sp.array([],dtype=int)

cols = max(rdata[:,1]) # another way of doing it...
rows = max(rdata[:,0])
vxm = sp.zeros((rows,cols))
for rating in rdata:
    vxm[rating[0]-1, rating[1]-1] = rating[2]
# x =5
# nnmf = NMF(n_components=x)
#
# W = nnmf.fit_transform(vxm)
# H = nnmf.components_

dimList = []
for num in [1,2,3,4,5,10,40,100,200,400,800]:
    sumNonzeros = (vxm !=0).sum()
    numTest = int(.15*sumNonzeros)  # proportion of values to remove

    elementList = []

    nonZeroTuple = sp.nonzero(vxm)

    for x in range(int(numTest)):
        rInt = sp.random.randint(0,nonZeroTuple[0].size)
        randrow = nonZeroTuple[0][rInt]
        randcolumn = nonZeroTuple[1][rInt]

        valElementIndex = [randrow,randcolumn]
        elementList.append(valElementIndex)

    modvxm = sp.copy(vxm) # The matrix with the randomly deleted values

    for x in elementList:
        modvxm[x[0],x[1]] = 0

    #self.fillAverages(vxm = self.modvxm)
    vxmc = sp.copy(modvxm)
    for i in range(vxmc.shape[0]):
        row = vxmc[i,:]
        row[row==0] = sp.mean(row[row!=0])
        vxmc[i,:] = row

    #INSERT creation of W and H, then take the dot product. = newmodvxm
    nnmf = NMF(n_components=num)

    W = nnmf.fit_transform(vxmc)
    H = nnmf.components_
    newmodvxm = sp.dot(W,H)

    sqDiff = 0
    for x in elementList:
        sqDiff += sp.square(newmodvxm[x[0],x[1]] - vxm[x[0],x[1]])
    rmse = sp.sqrt(sqDiff/len(elementList))

    print(num,': ',rmse)
    dimList.append(rmse)