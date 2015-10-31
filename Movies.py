from sklearn.decomposition import TruncatedSVD,PCA
import scipy as sp
from random import *
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as a3

class mPredictor():

    def __init__(self,file,dims,):
        self.dims = dims
        self.rdata = sp.loadtxt(file,int)
        self.v = sp.array([])
        self.m = sp.array([],dtype=int)
        self.vxm = sp.empty(self.dims)

    def buildMatrix(self):

        cols = max(self.rdata[:,1]) # another way of doing it...
        rows = max(self.rdata[:,0])
        self.vxm = sp.zeros((rows,cols))
        for rating in self.rdata:
            self.vxm[rating[0]-1, rating[1]-1] = rating[2]


    def fillAverages(self,vxm):

        vxmc = sp.copy(vxm)
        for i in range(vxmc.shape[0]):
            row = vxmc[i,:]
            row[row==0] = sp.mean(row[row!=0])
            vxmc[i,:] = row
        return vxmc

    def predict(self,cut,export=0, vxm=None):

        if vxm is None:
            vxm = self.vxm
            self.buildMatrix()
            print(vxm)
            vxm = self.fillAverages(vxm)


        U, s, V = sp.linalg.svd(vxm, full_matrices = True )

        U = U[:,0:cut]
        V = V[0:cut,:]
        S = sp.diag(s)
        S = S[:cut,:cut]

        new_vxm = sp.dot(U, sp.dot(S, V))
        if export == 0:
            return new_vxm
        else:
            return new_vxm

    def errorApproximation(self, ratio, dim=20):

        self.buildMatrix()

        sumNonzeros = (self.vxm !=0).sum()
        numTest = int(ratio*sumNonzeros)

        elementList = []

        nonZeroTuple = sp.nonzero(self.vxm)

        for x in range(int(numTest)):
            rInt = sp.random.randint(0,nonZeroTuple[0].size)
            randrow = nonZeroTuple[0][rInt]
            randcolumn = nonZeroTuple[1][rInt]

            valElementIndex = [randrow,randcolumn]
            elementList.append(valElementIndex)

        self.modvxm = sp.copy(self.vxm)

        for x in elementList:
            self.modvxm[x[0],x[1]] = 0

        self.modvmx = self.fillAverages(vxm = self.modvxm)
        self.newmodvxm = self.predict(dim,vxm=self.modvxm)

        sqDiff = 0
        for x in elementList:
            sqDiff += sp.square(self.newmodvxm[x[0],x[1]] - self.vxm[x[0],x[1]])
        self.rmse = sp.sqrt(sqDiff/len(elementList))

        # print(self.vxm[elementList[0][0],elementList[1][0]])
        # print(self.modvxm[elementList[0][0],elementList[1][0]])
        # print(self.newmodvxm[elementList[0][0],elementList[1][0]])
        # print(self.rmse)

if __name__ == "__main__":
    mat = mPredictor('u.data',[943,1682])
    mat.buildMatrix()
    vxm = mat.vxm
    pc = PCA(n_components=3)
    z = pc.fit_transform(vxm)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z[:,0],z[:,1], z[:,2])
    plt.show()

    #svd = TruncatedSVD(n_components=3)
    #svd.fit(vxm)
    #selector = RFE(svd,n_features_to_select=2)
    # x = mat.predict(cut=20)

    mat.errorApproximation(.15,dim=5)
    print(mat.rmse)
    dimList = []
    for x in [1,2,3,4,5,10,40,100,200,400,800]:
        mat = mPredictor('u.data',[943,1682])
        mat.errorApproximation(.15,dim=x)
        print(x,': ',mat.rmse)
        dimList.append(mat.rmse)
