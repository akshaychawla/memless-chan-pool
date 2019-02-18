#include<stdio.h> 
#include<stdlib.h>


 /*the gpu kernel, to be launched on threads + blocks + grid heirarchy*/
__global__ 
void KERNEL_max(float *d_out, float *d_in, int DIM)
{ 
        int idx = threadIdx.x; //threadIdx is a struct with 3 members, x,y, and z.  

        int max_value = *(d_in + idx*DIM + 0);  
        for(int j=0; j<DIM; j++){

                if (max_value<*(d_in + idx*DIM + j))
                {
                        max_value = *(d_in + idx*DIM + j);
                }
        }

        d_out[idx] = max_value;
}

float random_number(int min, int max)
{
        float num = rand() % (max + 1 - min) + min; 
        return num;
}

int main()
{

        const int NUM_SAMPLES=10; 
        const int DIM=16; 

        // 1. HOST CODE
        // input array (host)
        float * h_in = (float *)malloc(NUM_SAMPLES*DIM*sizeof(float)); 
        for(int i=0; i<NUM_SAMPLES; i++){
                for(int j=0; j<DIM; j++){
                        *(h_in + i*DIM + j) = random_number(1,100); 
                }
        }

        // print the input array 
        for(int i=0; i<NUM_SAMPLES; i++){
                printf("\nRow %d \n", i);
                for(int j=0; j<DIM; j++){
                        printf(" %f ", *(h_in + i*DIM + j)); 
                }
                printf("\n");
        }
        // output array (host)
        float * h_out = (float *)malloc(NUM_SAMPLES*sizeof(float));
        for(int i=0; i<NUM_SAMPLES; i++){
                h_out[i] = 0.0; 
        }

        
        // 2. DEVICE CODE
        float * d_in; 
        float * d_out; 

        // copy h_in to d_in 
        cudaMalloc((void **) &d_in, NUM_SAMPLES*DIM*sizeof(float));
        cudaMalloc((void **) &d_out, NUM_SAMPLES*sizeof(float));
        cudaMemcpy(d_in, h_in, NUM_SAMPLES*DIM*sizeof(float), cudaMemcpyHostToDevice); 

        // launch the kernel 
        KERNEL_max<<<1, NUM_SAMPLES>>>(d_out, d_in, DIM);

        // copy the output back to host memory 
        cudaMemcpy(h_out, d_out, NUM_SAMPLES*sizeof(float), cudaMemcpyDeviceToHost);

        // print the resultant 
        for(int i=0; i<NUM_SAMPLES; i++){
                printf("Row %d : %f \n", i, h_out[i]); 
        }
        
        /*// free GPU memory allocation */
        cudaFree(d_in); 
        cudaFree(d_out); 


    return 0;
} 


