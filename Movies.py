from sklearn.decomposition import TruncatedSVD
import scipy as sp

rdata = sp.loadtxt('u.data',int )

a = sp.zeros([100000,4])
v = sp.array([])
m = sp.array([],dtype=int)
vxm = sp.empty([943,1682])


for x in rdata[:,1]: # Builds the movie vector
    if x not in m:
        m = sp.append(m,x)

for x in rdata[:,0]: # Builds the viewer vector
    if x not in v:
        v = sp.append(v,x)

for x in rdata: # Creates the matrix containing the ratings of the movies by the viewers
    r = x[2]
    mov = sp.nonzero(m==x[1])
    view = sp.nonzero(v==x[0])
    vxm[view,mov] = r

for x in vxm.T: # For unrated slots, it takes the movie's average rating and replaces the zeros with that
    avg = sp.average(x[x>0])
    for y,z in zip(x, range(len(x))):
        if y == 0:
            x[z]=avg

U,s,V = sp.linalg.svd(vxm, full_matrices = True )

U = U[:,0:20]
V = V[0:20,:]
S = sp.diag(s)
S = S[:20,:20]

new_vxm = sp.dot(U, sp.dot(S,V))


