#include<stdio.h>

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

int main(){

        // Test this! 
        int arr[4][3] = { {1,2,3}, {4,5,6}, {7,8,9}, {10,11,12} }; // 4x3 array 
        int * arr_ptr = &arr[0][0];
        int indices[] = {2,1}; 
        int dimensions[] = {4,3};
        int num_dimensions = 2;

        unsigned long int offset = getOffset( indices, dimensions, num_dimensions );
        printf("value at ( %d , %d ) is %d \n", indices[0], indices[1], arr[indices[0]][indices[1]]);
        printf("value at ( %d , %d ) is %d \n", indices[0], indices[1], *(arr_ptr + offset) );

        return 0;
}
