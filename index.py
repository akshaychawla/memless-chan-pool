from copy import deepcopy
import numpy as np 

def getOffset(indices, dimensions):
    """
    Given a list of indices and dimensions, return the offset for an array
    """
    offset = 0.0
    for indexIdx in range(len(indices)): 
        
        product = deepcopy(indices[indexIdx])
        for dimIdx in range(indexIdx + 1, len(dimensions)): 
            product *= dimensions[dimIdx] 

        offset += product
    return int(offset)

if __name__ == "__main__":

    A = [1,2,3,4,5,6] # 2d array stored in row-major order
    dimensions = [2,3] 
    indices = [0,1]

    offset = getOffset(indices, dimensions)
    print("indices {} | Offset {} | Value {}".format(indices, offset, A[offset]))


    # For 3d numpy array 
    A = np.arange(24).reshape((2,3,4))
    A_flat = A.ravel().tolist()
    offset = getOffset([1,2,3], [2,3,4])
    print(A_flat[offset])
    print(A[1,2,3])




