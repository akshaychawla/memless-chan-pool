#include<stdio.h>
#include<stdlib.h>
#include<time.h>

float random_number(int min, int max){
        /* 
         * Function: random_number 
         * -----------------------
         *  return a random number between min and max as a float 
         */
        float num = rand() % (max + 1 - min) + min; 
        return num;
}

unsigned long int getOffset(int * indices, int * dimensions, int num_dimensions){ 
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

void randFill3d(float * arr, int * dimensions){
        /*
         * Function: randFill3d 
         * --------------------
         *  fills a 3d array with random numbers between 1,100 
         */
        int CHANNELS = dimensions[0]; 
        int HEIGHT = dimensions[1]; 
        int WIDTH = dimensions[2];
        for(int chan=0; chan<CHANNELS; chan++){
                for(int row=0; row<HEIGHT; row++){
                        for(int col=0; col<WIDTH; col++){
                                int indices[3] = {chan, row, col};
                                *(arr + getOffset(indices, dimensions, 3)) = random_number(1,100);
                        }
                }
        }
}

void printArr3d(float * arr, int * dimensions){
        /* To print a 3d array - used multiple times 
         */
        int CHANNELS = dimensions[0]; 
        int HEIGHT = dimensions[1]; 
        int WIDTH = dimensions[2];
        for(int chan=0; chan<CHANNELS; chan++){
                printf("\nChannel %d", chan);
                for(int row=0; row<HEIGHT; row++){
                        printf("\n");
                        for(int col=0; col<WIDTH; col++){
                                int indices[3] = {chan, row, col};
                                printf(" %f ", *(arr + getOffset(indices, dimensions, 3)));
                        }
                }
        }
}

int main(){

        // Test this! 
        float arr[4][3] = { {1,2,3}, {4,5,6}, {7,8,9}, {10,11,12} }; // 4x3 array 
        float * arr_ptr = &arr[0][0];
        int indices[] = {2,1}; 
        int dimensions[] = {4,3};
        int num_dimensions = 2;

        unsigned long int offset = getOffset( indices, dimensions, num_dimensions );
        printf("value at ( %d , %d ) is %f \n", indices[0], indices[1], arr[indices[0]][indices[1]]);
        printf("value at ( %d , %d ) is %f \n", indices[0], indices[1], *(arr_ptr + offset) );

        // Test fillArr and printArr 
        srand(time(0));
        const int HEIGHT=2;
        const int WIDTH=3; 
        const int DEPTH=4; 

        float * multiArr = (float *)malloc(HEIGHT*WIDTH*DEPTH*sizeof(float));
        int dimensions2[] = {DEPTH, HEIGHT, WIDTH};
        randFill3d(multiArr, dimensions2); 
        printArr3d(multiArr, dimensions2);
        

        return 0;
}
