import numpy as np

class PCA:
    def __init__(self, iters=32, epsilon=1e-5):
        super().__init__()
        self.iters = iters
        self.epsilon = epsilon

    def process(self, data):
        M = self.__cov( data )
        U = np.array( [[1,0,0],[0,1,0],[0,0,1]] )

        for iter in range( self.iters ):
            Q, R = np.linalg.qr( M )

            #-- QR Iteration
            M = R @ Q           # Eigen Values
            U = U @ Q           # Eigen Vectors

            # off Diagonal Error
            e = ( abs( M[0][1] ) +
                  abs( M[0][2] ) +
                  abs( M[1][2] )            
            )

            if e <= self.epsilon:
                break
        
        return np.array( [U[0][0], U[1][0], U[2][0]] )
        

    def __orthoNormalize(self, M):
        X, Y, Z = M[:,0], M[:,1], M[:,2]
        z = np.cross( X, Y );       y = np.cross( Z, X);

        x = (X / np.linalg.norm( X )).reshape(-1,1) 
        y = (X / np.linalg.norm( y )).reshape(-1,1) 
        z = (X / np.linalg.norm( z )).reshape(-1,1) 

        m = np.concatenate( (x, y, z), axis=1 )

        return m

    def __cov(self, data):
        d = np.array( data )
        o = d.mean( axis=0 )
        x = d - o
        cov = np.transpose( data ) @ data

        return cov



