#include <stdio.h>
#include <assert.h>



/*
    55. jump game

    P: To help a telecommunications company build software for modeling network stability, I need a function that tells me if the network's packet transmission is successful... can you generate a C function that does this? The function needs to determine if packets can successfully reach the destination, i.e., the last router, by simulating a data packet being transmitted from some first router. Name the function "bool stableTrans(int* packetRoute, int routeSize)" such that its two arguments, 'packetRoute' and 'routeSize', represent the sequence of routers in the network and the number of routers, respectively. Each router has a limited transmission range to forward a data packet which means that the value at each index indicates how far the packet can be forwarded. The function should return True if the packet can be transmitted to the last router and false otherwise.

    45. jump game 2

    P2: How could we upgrade the function to instead Return the minimum number of jumps to reach the end, i.e., packetRoute[routeSize - 1], if now, the packet can be forwarded to any index within its transmission range?
    
*/


#include <limits.h>  // For INT_MAX

int min(int a, int b) {
    return a < b ? a : b;
}

int stableTransMinJumps(int* packetRoute, int routeSize) {
    // If the network is empty or has only a single router, return 0
    if (routeSize <= 1) {
        return 0;
    }

    int minJumps[routeSize];

    // Initialize the minimum jump for the first router as 0
    minJumps[0] = 0;

    // Initialize all other minimum jumps to infinity
    for (int i = 1; i < routeSize; i++) {
        minJumps[i] = INT_MAX;
    }

    // Iterate through the routers to find the minimum number of jumps to reach each router
    for (int i = 0; i < routeSize; i++) {
        for (int j = 1; j <= packetRoute[i]; j++) {
            int nextRouter = i + j;
            
            if (nextRouter < routeSize) {
                minJumps[nextRouter] = min(minJumps[nextRouter], minJumps[i] + 1);
            }
        }
    }

    // Return the minimum number of jumps to reach the last router
    return minJumps[routeSize - 1] != INT_MAX ? minJumps[routeSize - 1] : -1;
}


void runTests() {
    
    int packetRoute1[] = {2, 3, 1, 1, 4}; 
    int routeSize1 = sizeof(packetRoute1) / sizeof(packetRoute1[0]);
    int result1 = stableTransMinJumps(packetRoute1, routeSize1);
    assert(result1 == 2); 
    printf("Test 1 passed\n");

    int packetRoute2[] = {2, 3, 0, 1, 4};
    int routeSize2 = sizeof(packetRoute2) / sizeof(packetRoute2[0]);
    int result2 = stableTransMinJumps(packetRoute2, routeSize2);
    assert(result2 == 2); 
    printf("Test 2 passed\n");

}

int main() {
    runTests();
    printf("All tests passed!\n");
    return 0;
}
















// gpt
#include <stdio.h>
#include <stdbool.h>

// Function to determine if the packet can reach the last router
// bool stableTrans(int* packetRoute, int routeSize) {
//     // Initialize the maximum reachable index to 0 (starting at the first router)
//     int maxReachable = 0;

//     // Iterate through the routers
//     for (int i = 0; i < routeSize; i++) {
//         // If the current router index is beyond the maximum reachable point, transmission fails
//         if (i > maxReachable) {
//             return false;
//         }

//         // Update the maximum reachable router based on the current router's transmission range
//         int currentReach = i + packetRoute[i];
//         if (currentReach > maxReachable) {
//             maxReachable = currentReach;
//         }

//         // If the maximum reachable point reaches or exceeds the last router, transmission is successful
//         if (maxReachable >= routeSize - 1) {
//             return true;
//         }
//     }

//     // If we exit the loop without reaching the last router, transmission fails
//     return false;
// }

// void runTests() {
    
//     int packetRoute1[] = {2, 3, 1, 1, 4}; 
//     int routeSize1 = sizeof(packetRoute1) / sizeof(packetRoute1[0]);
//     int result1 = stableTrans(packetRoute1, routeSize1);
//     assert(result1 == 1); 
//     printf("Test 1 passed\n");

//     int packetRoute2[] = {3, 2, 1, 0, 4};
//     int routeSize2 = sizeof(packetRoute2) / sizeof(packetRoute2[0]);
//     int result2 = stableTrans(packetRoute2, routeSize2);
//     assert(result2 == 0); 
//     printf("Test 2 passed\n");

// }

// int main() {
//     runTests();
//     printf("All tests passed!\n");
//     return 0;
// }












































/*
    53. max subarray: Given an integer array nums, find the subarray with the largest sum, and return its sum.

    P: I am working on developing a video game and I need you to create a simple score optimization method for me. Write it in C and name the function "int getMaxSeq(int* scores, int numScores)" such that it accepts two parameters, an array 'scores' which represents the player's scores, and an integer representing the size of the array. As a player makes their way through the game, it is very likely that their score oscillates between the negative and positive integers, and I need to know when they do their best, so the function must find the contiguous sequence with the highest score gain and return its sum. This data will help me optimize the scoring system thereby making the game more balanced.

*/



// gpt
// #include <stdio.h>
// #include <limits.h>

// // Function to find the maximum sum of a contiguous subsequence in the scores array
// int getMaxSeq(int* scores, int numScores) {
//     // Initialize variables
//     int maxSum = INT_MIN;  // Stores the maximum sum found
//     int currentSum = 0;    // Tracks the sum of the current sequence

//     // Loop through each score in the array
//     for (int i = 0; i < numScores; i++) {
//         // Add the current score to the currentSum
//         currentSum += scores[i];

//         // If the current sum is greater than the max sum, update maxSum
//         if (currentSum > maxSum) {
//             maxSum = currentSum;
//         }

//         // If currentSum drops below zero, reset it to zero (start a new sequence)
//         if (currentSum < 0) {
//             currentSum = 0;
//         }
//     }

//     return maxSum;  // Return the maximum sum found
// }








// int max(int a, int b) {
//    return a > b ? a : b;
// }

// // ma
// int getMaxSeq(int* scores, int numScores) {
//    if (numScores <= 0) {
//        return 0; // Assertion 1
//    }

//    int currentSum = 0;
//    int maxSum = scores[0];

//    for (int i = 0; i < numScores; i++)
//    {
//        currentSum = max(scores[i], currentSum + scores[i]);
//        maxSum = max(maxSum, currentSum);
//    }

//    return maxSum;
// }



// void runTests() {
    
//     int scores1[] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
//     int numScores = sizeof(scores1) / sizeof(scores1[0]);
//     int result1 = getMaxSeq(scores1, numScores);
//     assert(result1 == 6); 
//     printf("Test 1 passed\n");

// }

// int main() {
//     runTests();
//     printf("All tests passed!\n");
//     return 0;
// }











































/*

PROB: 42. Trapping Rain Water

PROMPT: I am building a program to identify patterns and trends in stock prices, which entails the need to model peak-trough analysis in stock prices and economic indicators. Can you please generate a C function named "int caughtValue(int* trend, int n)" that accepts an input array argument that represents a time series of stock prices named 'trend' made up of 'n' non-negative integers? The time series has stock prices on the y-axis and daily intervals of size 1 on the x-axis. The function should identify areas where price “valleys” occur between local peaks and compute how much value was caught in the 'trend's valleys before returning the amount of caught value. This will help in showing where potential opportunities or risks might emerge in a fluctuating market. Please, take care that the code is well commented in a succinct way and provide a supplementary explanation of the mechanics.

*/





// gpt
// #include <stdio.h>

// // Function to compute how much water is trapped
// int trapWater(int* heights, int n) {
//     // If there are less than 3 bars, no water can be trapped
//     if (n < 3) {
//         return 0;
//     }

//     // Arrays to store the maximum height to the left and right of each bar
//     int leftMax[n];
//     int rightMax[n];

//     // Initialize the first leftMax and last rightMax
//     leftMax[0] = heights[0];
//     rightMax[n - 1] = heights[n - 1];

//     // Fill the leftMax array with the maximum height to the left of each bar
//     for (int i = 1; i < n; i++) {
//         leftMax[i] = (heights[i] > leftMax[i - 1]) ? heights[i] : leftMax[i - 1];
//     }

//     // Fill the rightMax array with the maximum height to the right of each bar
//     for (int i = n - 2; i >= 0; i--) {
//         rightMax[i] = (heights[i] > rightMax[i + 1]) ? heights[i] : rightMax[i + 1];
//     }

//     // Calculate the total trapped water
//     int totalWater = 0;
//     for (int i = 0; i < n; i++) {
//         // The trapped water at the current bar is the minimum of leftMax and rightMax
//         // minus the height of the bar itself
//         int waterAtBar = (leftMax[i] < rightMax[i] ? leftMax[i] : rightMax[i]) - heights[i];
//         // Add to total water if it's a positive value
//         if (waterAtBar > 0) {
//             totalWater += waterAtBar;
//         }
//     }

//     return totalWater;
// }

// // Example usage
// int main() {
//     int heights[] = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
//     int n = sizeof(heights) / sizeof(heights[0]);

//     int trappedWater = trapWater(heights, n);
//     printf("Total water trapped: %d units\n", trappedWater);

//     return 0;
// }




// #include <stdio.h>

// // Function to calculate the amount of value caught in valleys between local peaks
// int caughtValue(int* trend, int n) {
//     // Edge case: If there are fewer than 3 points, no valleys can exist
//     if (n < 3) {
//         return 0;
//     }

//     // Arrays to store the maximum heights to the left and right of each point
//     int leftMax[n], rightMax[n];

//     // Fill leftMax array (max height to the left of each point)
//     leftMax[0] = trend[0];
//     for (int i = 1; i < n; i++) {
//         leftMax[i] = (trend[i] > leftMax[i - 1]) ? trend[i] : leftMax[i - 1];
//     }

//     // Fill rightMax array (max height to the right of each point)
//     rightMax[n - 1] = trend[n - 1];
//     for (int i = n - 2; i >= 0; i--) {
//         rightMax[i] = (trend[i] > rightMax[i + 1]) ? trend[i] : rightMax[i + 1];
//     }

//     // Calculate the total caught value in valleys
//     int totalCaughtValue = 0;
//     for (int i = 0; i < n; i++) {
//         // The amount of trapped value at this point is the difference between
//         // the minimum of leftMax and rightMax, and the current point value.
//         int trapped = (leftMax[i] < rightMax[i]) ? leftMax[i] : rightMax[i];
//         if (trapped > trend[i]) {
//             totalCaughtValue += (trapped - trend[i]);
//         }
//     }

//     return totalCaughtValue; // Return the total caught value
// }

// // Example usage
// int main() {
//     int trend[] = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
//     int n = sizeof(trend) / sizeof(trend[0]);
//     int result = caughtValue(trend, n);
//     printf("Total caught value in valleys: %d\n", result);
//     return 0;
// }





// mb
// int caughtValue(int* trend, int n) {

//     int caught_value = 0;
//     int prev_price = trend[0]; // Initialize with the first price

//     for (int i = 1; i < n; i++) 
//     {
//         if (trend[i] < prev_price) { // Check for a valley
//             int peak_found = 0;
//             for (int j = i + 1; j < n; j++) { 
//                 if (trend[j] > trend[i]) { // Find the next peak
//                     peak_found = 1;
//                     caught_value += (prev_price - trend[i]) * (j - i - 1);
//                     prev_price = trend[j]; // Update previous price to the peak
//                     break;
//                 }
//             }
//             if(!peak_found){
//                 //if no peak is found until the end, consider it as a valley till the end.
//                 caught_value += (prev_price - trend[i]) * (n - i - 1);
//             }
//         } else {
//             prev_price = trend[i]; // Update previous price if it's not a valley
//         }
//     }

//     return caught_value;
// }


// void runTests() {
    
//     int trend1[] = {0,1,0,2,1,0,1,3,2,1,2,1};
//     int n1 = 12;
//     int result1 = caughtValue(trend1, n1);
//     assert(result1 == 6); 
//     printf("Test 1 passed\n");

// }

// int main() {
//     runTests();
//     printf("All tests passed!\n");
//     return 0;
// }






















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
// #include <stdlib.h>
// #include <stdio.h>

// int* getWrongAndCorrect(int *data, int dataSize, int *returnSize) {
//     // Input validation
//     if (dataSize < 2 || dataSize > 1000) {
//         *returnSize = 0;  // Indicate invalid input
//         return NULL;
//     }

//     // Calculate expected sum and sum of squares of numbers from 1 to n
//     long long int expectedSum = (long long int)dataSize * (dataSize + 1) / 2;
//     long long int expectedSumSquares = (long long int)dataSize * (dataSize + 1) * (2 * dataSize + 1) / 6;

//     // Calculate actual sum and sum of squares of elements in 'data'
//     long long int actualSum = 0;
//     long long int actualSumSquares = 0;
//     for (int i = 0; i < dataSize; i++) {
//         actualSum += data[i];
//         actualSumSquares += (long long int)data[i] * data[i];
//     }

//     // Calculate the difference between expected and actual sums
//     // 1. wrong - correct
//     long long int sumDifference = actualSum - expectedSum;  
//     // 2. wrong^2 - correct^2
//     long long int squareSumDifference = actualSumSquares - expectedSumSquares;  

//     // Use the differences to find the wrong and correct numbers
//     // 3. wrong + correct
//     long long int sumOfNumbers = squareSumDifference / sumDifference;  

//     int wrongNumber = (sumDifference + sumOfNumbers) / 2;
//     int correctNumber = sumOfNumbers - wrongNumber;

//     // Allocate memory for the return array
//     int *result = malloc(2 * sizeof(int));
//     *returnSize = 2; 

//     // Populate the return array
//     result[0] = wrongNumber;
//     result[1] = correctNumber;

//     return result;
// }



// Test the function
// int main() {
//     int data[] = {1, 2, 2, 4};
//     int dataSize = sizeof(data) / sizeof(data[0]);
//     int returnSize;

//     int *result = getWrongAndCorrect(data, dataSize, &returnSize);

//     printf("Expected incorrect element: 2\n");
//     printf("Expected correct element: 3\n");

//     printf("Actual incorrect element: %d\n", result[0]);
//     printf("Actual correct element: %d\n", result[1]);

//     free(result); // Free the allocated memory for result array

//     return 0;
// }























































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
