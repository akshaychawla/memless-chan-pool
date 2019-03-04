
__device__
unsigned long int getOffset_DEVICE(int * indices, int * dimensions, int num_dimensions){ 
        /*
         * Function: getOffset
         * -------------------
         *  compute the offset from the base address of an array for given indices corresponding
         *  to dimensions. 
         * indices: pointer to array of indices e.g A[1,2,3] 
         * dimensions: pointer to array of dimensions e.g dimensionality of A = 2x3x4
         * num_dimensions: number of dimensions e.g dimensions(A) = 3
         *
         * returns:
         * -------
         *  offset: the offset from the base memory address. (NOTE: independant of datatype)
         */

        unsigned long int offset = 0.0; 
        for(int indexIdx=0; indexIdx<num_dimensions; indexIdx++){
                unsigned long int product = indices[indexIdx]; 
                for(int dimIdx=indexIdx+1; dimIdx<num_dimensions; dimIdx ++){
                        product = product * dimensions[dimIdx]; 
                }
                offset = offset + product; 
        }

        return offset; 
}

extern "C" __global__ 
void my_add(float * x1, float * x2, float * y){

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    printf("THREADIDX %d blockDim.x %d , blockIdx.x %d threadIdx.x %d \n", tid, blockDim.x, blockIdx.x, threadIdx.x);
    y[tid] = x1[tid] + x2[tid];
}

extern "C" __global__ 
void add_2d(float *x1, int * dimensions, float *y){
    // printf("BlockIdx info x: %d y: %d \n", blockIdx.x, blockIdx.y);
    // printf("ThreadIdx info x: %d y: %d z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    int row = threadIdx.x; 
    int col = threadIdx.y; 
    int indices[] = {0,row,col};
    // int dimensions[] = {2,3,4};
    printf("dimensions %d %d %d \n", dimensions[0], dimensions[1], dimensions[2]);
    int num_dimensions = 3;
    int offset_value = getOffset_DEVICE(indices, dimensions, num_dimensions);
    // printf("%d\n", offset_value);
    float x1_val = *(x1+offset_value);
    printf("Row: %d | Col: %d | Value: %f \n", row, col, x1_val);
}

extern "C" __global__ 
void KERNEL_max(float * d_out, int * d_out_DIMS, float * d_in, int * d_in_DIMS)
{
	//printf("Block x: %d , y %d\n",blockIdx.x, blockIdx.y);
	int col= blockIdx.x*blockDim.x + threadIdx.x; 
	int row= blockIdx.y*blockDim.y + threadIdx.y; 
    int indices[] = {0,row,col};
    printf("Grid Dimensions %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
	
	// bounds checking 
	if ( (indices[1]>=d_in_DIMS[1]) || (indices[2]>=d_in_DIMS[2]) ){
		return;
	}

	float max_value = *(d_in + getOffset_DEVICE(indices, d_in_DIMS, 3));
	int DEPTH = d_in_DIMS[0];
	for(int i=0; i<DEPTH; i++){
		int indices2[] = {i, row, col};
		if ( max_value < *(d_in+getOffset_DEVICE(indices2, d_in_DIMS, 3)) ){
		       max_value = *(d_in+getOffset_DEVICE(indices2, d_in_DIMS, 3));
		}
	}
	// store the max value in the (row,col) location of d_out
	int indices3[] = {row,col};
	*(d_out + getOffset_DEVICE(indices3, d_out_DIMS, 2)) = max_value;
}
