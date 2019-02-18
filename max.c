#include<stdio.h> 
#include<stdlib.h>

struct tIdx 
{
        int x,y,z; 
};

 /*the gpu kernel, to be launched on threads + blocks + grid heirarchy*/
void KERNEL_max(float *d_out, float *d_in, int NUM_SAMPLES, int DIM, struct tIdx threadIdx)
{ 
        int idx = threadIdx.x; //threadIdx is a struct with 3 members, x,y, and z.  

        int max_value = *(d_in);  
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

        struct tIdx threadIdx; 
        threadIdx.x = 6; 
        threadIdx.y = 1; 
        threadIdx.z = 1;
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

        
        KERNEL_max(h_out, h_in, NUM_SAMPLES, DIM, threadIdx);

        // 2. DEVICE CODE
        /*float * d_in; */
        /*float * d_out; */

        /*// copy h_in to d_in */
        /*cudaMalloc((void **) &d_in, NUM_SAMPLES*DIM*sizeof(float));*/
        /*cudaMalloc((void **) &d_out, NUM_SAMPLES*sizeof(float));*/
        /*cudaMemcpy(d_in, h_in, NUM_SAMPLES*DIM*sizeof(float), cudaMemcpyHostToDevice); */

        /*// launch the kernel */
        /*KERNEL_max<<<1, NUM_SAMPLES>>>(d_out, d_in, NUM_SAMPLES, DIM);*/

        /*// copy the output back to host memory */
        /*cudaMemcpy(h_out, d_out, NUM_SAMPLES*sizeof(float), cudaMemcpyDeviceToHost);*/

        // print the resultant 
        for(int i=0; i<NUM_SAMPLES; i++){
                printf("%f \n", h_out[i]); 
        }
        
        /*// free GPU memory allocation */
        /*cudaFree(d_in); */
        /*cudaFree(d_out); */


    return 0;
} 


