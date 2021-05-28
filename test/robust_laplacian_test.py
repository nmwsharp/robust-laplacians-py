import unittest
import os
import sys
import os.path as path
import numpy as np
import scipy

# Path to where the bindings live

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
if os.name == 'nt': # if Windows
    # handle default location where VS puts binary
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "Debug")))
else:
    # normal / unix case
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))


import robust_laplacian as rl


def generate_verts(n_pts=999):
    np.random.seed(777)        
    return np.random.rand(n_pts, 3)

def generate_faces(n_pts=999):
    # n_pts should be a multiple of 3 for indexing to work out
    np.random.seed(777)        
    rand_faces = np.random.randint(0, n_pts, size=(2*n_pts,3))
    coverage_faces = np.arange(n_pts).reshape(-1, 3)
    faces = np.vstack((rand_faces, coverage_faces))
    return faces

def is_symmetric(A, eps=1e-6):
    resid = A - A.T
    return np.all(np.abs(resid.data) < eps)

def is_nonnegative(A, eps=1e-6):
    return np.all(A.data > -eps)

class TestCore(unittest.TestCase):

    def test_mesh_laplacian(self):

        V = generate_verts()
        F = generate_faces()

        L, M = rl.mesh_laplacian(V, F)

        # Validate mass matrix    
        self.assertTrue(is_nonnegative(M))
        self.assertTrue(is_symmetric(M))
        self.assertEqual(M.sum(), M.diagonal().sum())
        
        # Validate Laplacian 
        self.assertTrue(is_symmetric(L))
        off_L = scipy.sparse.diags(L.diagonal()) - L
        self.assertTrue(is_nonnegative(off_L)) # positive edge weights
        self.assertGreater(L.sum(), -1e-5)


        # Trigger validation errors
        # rl.mesh_laplacian("cat", F)  
        # rl.mesh_laplacian(V, "cat")  
        # rl.mesh_laplacian(V.flatten(), F)  
        # rl.mesh_laplacian(V, F.flatten())  
    
    def test_mesh_laplacian_unused_verts(self):

        V = generate_verts()
        F = generate_faces()
        
        # add an unused vertex at the beginning and end
        V = np.vstack((
                    np.array([[0., 0., 0.,]]),
                    V, 
                    np.array([[0., 0., 0.,]])
                )) 
        F = F + 1

        L, M = rl.mesh_laplacian(V, F)

        # Validate mass matrix    
        self.assertTrue(is_nonnegative(M))
        self.assertTrue(is_symmetric(M))
        self.assertEqual(M.sum(), M.diagonal().sum())
        
        # Validate Laplacian 
        self.assertTrue(is_symmetric(L))
        off_L = scipy.sparse.diags(L.diagonal()) - L
        self.assertTrue(is_nonnegative(off_L)) # positive edge weights
        self.assertGreater(L.sum(), -1e-5)

   
    def test_point_cloud_laplacian(self):

        V = generate_verts()

        L, M = rl.point_cloud_laplacian(V)

        # Validate mass matrix    
        self.assertTrue(is_nonnegative(M))
        self.assertTrue(is_symmetric(M))
        self.assertEqual(M.sum(), M.diagonal().sum())
        
        # Validate Laplacian 
        self.assertTrue(is_symmetric(L))
        off_L = scipy.sparse.diags(L.diagonal()) - L
        self.assertTrue(is_nonnegative(off_L)) # positive edge weights
        self.assertGreater(L.sum(), -1e-5)

        # Trigger validation errors
        # rl.point_cloud_laplacian("cat")  
        # rl.point_cloud_laplacian(V.flatten())  

if __name__ == '__main__':
    unittest.main()
