#include<stdio.h> 
#include<stdlib.h>
#include<time.h>
#include"utils.h"

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

__global__
void KERNEL_max(float * d_out, int * d_out_DIMS, float * d_in, int * d_in_DIMS)
{
	//printf("Block x: %d , y %d\n",blockIdx.x, blockIdx.y);
	int col = blockIdx.x*blockDim.x + threadIdx.x; 
	int row = blockIdx.y*blockDim.y + threadIdx.y; 
	int indices[] = {0,row,col};
	
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

int main()
{

	srand(time(0));
	const int HEIGHT=6; 
	const int WIDTH=7; 
	const int DEPTH=2; 

        // input array (host)
	int h_in_DIMS[] = {DEPTH, HEIGHT, WIDTH}; 
	float * h_in = new3dArray(DEPTH, HEIGHT, WIDTH); 
	randFill3d(h_in, h_in_DIMS); 
	printArr3d(h_in, h_in_DIMS);

	// output array (host)
	int h_out_DIMS[] = {HEIGHT, WIDTH}; 
	float * h_out = new2dArray(HEIGHT, WIDTH); 
	randFill2d(h_out, h_out_DIMS); 

	// max value 
	//KERNEL_max(h_out, h_out_DIMS, h_in, h_in_DIMS, 2,3); 

	// Split the computation up into multiple tiles 
	const int MAX_TILE_DIM = 4; 
	int NUM_TILES_X = ceil(float(WIDTH) / MAX_TILE_DIM); 
	int NUM_TILES_Y = ceil(float(HEIGHT) / MAX_TILE_DIM); 
	dim3 grid(NUM_TILES_X, NUM_TILES_Y);
	printf("\nX_TILES:%d Y_TILEs: %d \n",NUM_TILES_X, NUM_TILES_Y);

	
	float * d_in; 
	float * d_out; 
	int * d_out_DIMS;
	int * d_in_DIMS;	
        cudaMalloc((void **) &d_in, DEPTH*HEIGHT*WIDTH*sizeof(float));
        cudaMalloc((void **) &d_out, HEIGHT*WIDTH*sizeof(float));
	cudaMalloc((void **) &d_out_DIMS, 2*sizeof(int));
	cudaMalloc((void **) &d_in_DIMS, 3*sizeof(int));
        cudaMemcpy(d_in, h_in, DEPTH*HEIGHT*WIDTH*sizeof(float), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_in_DIMS, h_in_DIMS, 3*sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_out_DIMS, h_out_DIMS, 2*sizeof(int), cudaMemcpyHostToDevice); 

	//dim3 block(WIDTH, HEIGHT);
	dim3 block(MAX_TILE_DIM, MAX_TILE_DIM);

	KERNEL_max<<<grid, block>>>(d_out, d_out_DIMS, d_in, d_in_DIMS);

        cudaMemcpy(h_out, d_out, HEIGHT*WIDTH*sizeof(float), cudaMemcpyDeviceToHost);

	printf("\n Final channel pooled output -->\n");
	printArr2d(h_out, h_out_DIMS);

	cudaFree(d_in); 
	cudaFree(d_out);
	

	
    return 0;
} 


