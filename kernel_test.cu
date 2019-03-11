
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
    printf("Grid Dimensions %d %d %d BlockIdx: %d %d %d ThreadIdx: %d %d %d \n", gridDim.x, gridDim.y, gridDim.z,
    blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
	
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



/* 
This function performs channel pooling over multiple sections 
*/
extern "C" __global__ 
void KERNEL_max_multi(float * d_out, int * d_out_DIMS, float * d_in, int * d_in_DIMS, int * channel_idx_sets, int * channel_idx_sets_DIMS, 
    int * max_channels, int * max_channels_DIMS, int MAX_CHANNELS_PER_SET)
{
	//printf("Block x: %d , y %d\n",blockIdx.x, blockIdx.y);
	int col= blockIdx.x*blockDim.x + threadIdx.x; 
    int row= blockIdx.y*blockDim.y + threadIdx.y; 
    int channel_set_idx = threadIdx.z;
    int batch_idx = blockIdx.z;
	
	// bounds checking 
	if ( (row>=d_in_DIMS[2]) || (col>=d_in_DIMS[3]) ){ return;
	}

    // Find maximum value along channels subset indexed by (channel_set_idx, :) subarray in channel_idx_sets 
    float max_val = 0.0; //min value based on FLT_MIN
    int max_channel = -1;
    int first_pass = 1;
    for(int channel_index=0; channel_index<MAX_CHANNELS_PER_SET; channel_index++){

        // 1. extract channel value from 2d array channel_idx_sets 
        int indices_0[] = {channel_set_idx, channel_index};
        int current_channel = *(channel_idx_sets + getOffset_DEVICE(indices_0, channel_idx_sets_DIMS, 2));

        // Special case: use -1 padding for unequal channel sets.
        if (current_channel==-1){
            continue;
        }

        // 2. Extract the value at this location (current_channel, row, col)
        int indices2[] = {batch_idx, current_channel, row, col}; 
        float current_value = *(d_in+getOffset_DEVICE(indices2,d_in_DIMS,4)); 

        if(first_pass==1){
            max_val = current_value;
            max_channel = current_channel;
            first_pass = 0;
        }
        else {
            if(current_value > max_val){
                max_val = current_value;
                max_channel = current_channel;
            }

        }
    }
    // Replace the value at location (channel_set_idx, row, col) in  d_out
    int indices_out[] = {batch_idx, channel_set_idx, row, col}; 
    *(d_out + getOffset_DEVICE(indices_out, d_out_DIMS, 4)) = max_val;

    // Store the max_channel at the corresponding location in tensor max_channels (NUM_CHANNEL_SETS x HEIGHT x WIDTH)
    int indices_mcs[] = {batch_idx, channel_set_idx, row, col};
    *(max_channels + getOffset_DEVICE(indices_mcs, max_channels_DIMS,4)) = max_channel;
}

/* 
This function performs BACKWARD PASS of channel pooling over multiple sections 
*/
extern "C" __global__ 
void KERNEL_max_multi_BACKWARD(float * GRAD_d_out, int * GRAD_d_out_DIMS, float * GRAD_d_in, int * GRAD_d_in_DIMS,
    int * max_channels, int * max_channels_DIMS)
{

	int col= blockIdx.x*blockDim.x + threadIdx.x; 
    int row= blockIdx.y*blockDim.y + threadIdx.y; 
    int channel_set_idx = threadIdx.z;
    int batch_idx = blockIdx.z;
	
	// bounds checking 
	if ( (row>=d_in_DIMS[2]) || (col>=d_in_DIMS[3]) ){
		return;
	}

    // find the gradient value at location (batch, channel_set_idx, row, col) in GRAD_d_out
    int indices_out[] = {batch_idx, channel_set_idx, row, col}; 
    float local_grad = *(GRAD_d_out + getOffset_DEVICE(indices_out, GRAD_d_out_DIMS, 4));

    // find the channel that had the maximum value for this (batchsize, channel_set_indices, row, col )
    int indices_mcs[] = {batch_idx, channel_set_idx, row, col};
    int max_channel = *(max_channels + getOffset_DEVICE(indices_mcs, max_channels_DIMS,4));

    // rout local_grad to location (batch,max_channel,row,col) in GRAD_d_in
    int indices_GRAD_d_in[] = {batch_idx, max_channel, row, col};
    *(GRAD_d_in + getOffset_DEVICE(indices_GRAD_d_in, GRAD_d_in_DIMS, 4)) += local_grad;

}
