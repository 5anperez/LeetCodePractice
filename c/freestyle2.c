/*
    ~ PROBLEM: 645. Set Mismatch

    ~ PROMPT: Implement a C function that accepts three arguments. The first is an array "int *data" such that "data's" elements consist of every member in the set [1, n]. The second arg is the size "int *dataSize" and the third is "int *returnSize", so name the function "int* getWrongAndCorrect(int *data, int dataSize, int *returnSize)". This data was transmitted over the air and experienced noise. Therefore, the function must find the error and return an array with the incorrect element and the correct element. Also note that the data is guaranteed to have atleast 2 elements and no more than one thousand elts.
*/


/*  OG  */
// #include <stdio.h>
// #include <stdlib.h>

// int* getWrongAndCorrect(int *data, int dataSize, int *returnSize) 
// {
//     int *result = (int*)malloc(2 * sizeof(int));

//      // To store the count of each number
//     int *count = (int*)calloc(dataSize + 1, sizeof(int));
    
//     // Initialize return size
//     *returnSize = 2;

//     // Count occurrences of each number
//     for (int i = 0; i < dataSize; i++) 
//         count[data[i]]++;

//     int incorrect = -1;
//     int correct = -1;

//     // Find the incorrect and correct numbers
//     for (int i = 1; i <= dataSize; i++) {
//         if (count[i] == 2) 
//             incorrect = i; 
//         else if (count[i] == 0) 
//             correct = i;
//     }

//     result[0] = incorrect;
//     result[1] = correct;

//     free(count); // Free the allocated memory for count array
//     return result;
// }









/*  MA  */
// #include <stdlib.h>
// #include <stdio.h>

// int* getWrongAndCorrect(int *data, int dataSize, int *returnSize) {
//     // Input validation
//     if (dataSize < 2 || dataSize > 1000) {
//         *returnSize = 0;  // Indicate invalid input
//         return NULL;
//     }

//     // Calculate expected sum of numbers from 1 to n
//     long long int expectedSum = (long long int)dataSize * (dataSize + 1) / 2;

//     // Calculate actual sum of elements in 'data'
//     long long int actualSum = 0;
//     for (int i = 0; i < dataSize; i++) {
//         actualSum += data[i];
//     }

//     // Determine the wrong number and its correct replacement
//     int wrongNumber = actualSum - expectedSum;
//     int correctNumber = wrongNumber - dataSize; 

//     // Allocate memory for the return array
//     int *result = malloc(2 * sizeof(int));
//     *returnSize = 2; 

//     // Populate the return array
//     result[0] = wrongNumber;
//     result[1] = correctNumber;

//     return result;
// }







/*  MA fixed  */
#include <stdlib.h>
#include <stdio.h>

int* getWrongAndCorrect(int *data, int dataSize, int *returnSize) {
    // Input validation
    if (dataSize < 2 || dataSize > 1000) {
        *returnSize = 0;  // Indicate invalid input
        return NULL;
    }

    // Calculate expected sum and sum of squares of numbers from 1 to n
    long long int expectedSum = (long long int)dataSize * (dataSize + 1) / 2;
    long long int expectedSumSquares = (long long int)dataSize * (dataSize + 1) * (2 * dataSize + 1) / 6;

    // Calculate actual sum and sum of squares of elements in 'data'
    long long int actualSum = 0;
    long long int actualSumSquares = 0;
    for (int i = 0; i < dataSize; i++) {
        actualSum += data[i];
        actualSumSquares += (long long int)data[i] * data[i];
    }

    // Calculate the difference between expected and actual sums
    // 1. wrong - correct
    long long int sumDifference = actualSum - expectedSum;  
    // 2. wrong^2 - correct^2
    long long int squareSumDifference = actualSumSquares - expectedSumSquares;  

    // Use the differences to find the wrong and correct numbers
    // 3. wrong + correct
    long long int sumOfNumbers = squareSumDifference / sumDifference;  

    int wrongNumber = (sumDifference + sumOfNumbers) / 2;
    int correctNumber = sumOfNumbers - wrongNumber;

    // Allocate memory for the return array
    int *result = malloc(2 * sizeof(int));
    *returnSize = 2; 

    // Populate the return array
    result[0] = wrongNumber;
    result[1] = correctNumber;

    return result;
}



// Test the function
int main() {
    int data[] = {1, 2, 2, 4};
    int dataSize = sizeof(data) / sizeof(data[0]);
    int returnSize;

    int *result = getWrongAndCorrect(data, dataSize, &returnSize);

    printf("Expected incorrect element: 2\n");
    printf("Expected correct element: 3\n");

    printf("Actual incorrect element: %d\n", result[0]);
    printf("Actual correct element: %d\n", result[1]);

    free(result); // Free the allocated memory for result array

    return 0;
}























































/*
    ~ PROBLEM: N/A (forgot to get)

    ~ PROMPT: N/A
*/

// #include <stdio.h>
// #include <math.h>
// #include <stdbool.h>

// bool isPy(int c) 
// {
//     // Check if the input is a non-negative integer
//     if (c < 0) 
//         return false;
    
//     // Iterate through possible values of a
//     for (int a = 0; a * a <= c; a++) 
//     {
//         // Get the difference
//         int b_squared = c - a * a;

//         // Find its root
//         int b = (int)sqrt(b_squared);
        
//         // Decide
//         if (b * b == b_squared)
//             return true;
//     }
    
//     return false;
// }

// int main() 
// {
//     // Test cases
//     printf("Expected: 1, Actual: %d\n", isPy(5));  // 1 (true)
//     printf("Expected: 0, Actual: %d\n", isPy(3));  // 0 (false)
//     printf("Expected: 0, Actual: %d\n", isPy(2147482647)); // 1 (true)
//     printf("Expected: 1, Actual: %d\n", isPy(50)); // 1 (true)

//     return 0;
// }
