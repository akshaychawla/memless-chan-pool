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

void randFill(float * arr, int * dimensions, int num_dimensions){
	/* 
	 * Must be a recursable function 
	 */ 
	
	// 1. Base condition  
	if(num_dimensions==1){
		printf("Reached base condition\n");
		for(int i=0; i<dimensions[0]; i++){
			*(arr + i) = random_number(1,100);
		}
	}
	// 2. Recursive condition 
	else { 
		int first_dim = dimensions[0]; 
		int second_dim = dimensions[1];
		printf("First dim : %d | Second dim : %d", first_dim, second_dim);
		int * new_dims = dimensions + 1; 
		for(int dim=0; dim<first_dim; dim++){
			randFill( arr+second_dim , new_dims, num_dimensions-1);
		}
	}
}	


void randFill2d(float * arr, int * dimensions){
        /*
         * Function: randFill3d 
         * --------------------
         *  fills a 3d array with random numbers between 1,100 
         */
        int HEIGHT = dimensions[0]; 
        int WIDTH = dimensions[1];
	for(int row=0; row<HEIGHT; row++){
                        for(int col=0; col<WIDTH; col++){
                                int indices[2] = {row, col};
                                *(arr + getOffset(indices, dimensions, 2)) = random_number(1,100);
                    	}
	}
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

void printArr2d(float * arr, int * dimensions){
	int HEIGHT = dimensions[0]; 
	int WIDTH  = dimensions[1]; 

	printf("\n2d Array: \n");
	for(int r=0; r<HEIGHT; r++){
		printf("\n"); 
		for(int c=0; c<WIDTH; c++){
			int indices[] = {r,c}; 
			printf(" %.1f ", *(arr + getOffset(indices, dimensions, 2)));
		}
	}
	printf("\n");
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
                                printf(" %.1f ", *(arr + getOffset(indices, dimensions, 3)));
                        }
                }
		printf("\n");
        }
}

float * new3dArray(int DEPTH, int HEIGHT, int WIDTH){
	float * arr = (float*)malloc(DEPTH*HEIGHT*WIDTH*sizeof(float)); 
	return arr; 
}

float * new2dArray(int HEIGHT, int WIDTH){
	float * arr = (float*)malloc(HEIGHT*WIDTH*sizeof(float)); 
	return arr; 
}

/*
int main()
{
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
*/
