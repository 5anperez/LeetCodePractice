/*
    ~ PROBLEM: 645. Set Mismatch

    ~ PROMPT: Implement a C function that accepts three arguments. The first is an array "int *data" such that "data's" elements consist of every member in the set [1, n]. The second arg is the size "int *dataSize" and the third is "int *returnSize", so name the function "int* getWrongAndCorrect(int *data, int dataSize, int *returnSize)". This data was transmitted over the air and experienced noise. Therefore, the function must find the error and return an array with the incorrect element and the correct element. Also note that the data is guaranteed to have atleast 2 elements and no more than one thousand elts.
*/


/*  OG  */
#include <stdio.h>
#include <stdlib.h>

int* getWrongAndCorrect(int *data, int dataSize, int *returnSize) 
{
    int *result = (int*)malloc(2 * sizeof(int));

     // To store the count of each number
    int *count = (int*)calloc(dataSize + 1, sizeof(int));
    
    // Initialize return size
    *returnSize = 2;

    // Count occurrences of each number
    for (int i = 0; i < dataSize; i++) 
        count[data[i]]++;

    int incorrect = -1;
    int correct = -1;

    // Find the incorrect and correct numbers
    for (int i = 1; i <= dataSize; i++) {
        if (count[i] == 2) 
            incorrect = i; 
        else if (count[i] == 0) 
            correct = i;
    }

    result[0] = incorrect;
    result[1] = correct;

    free(count); // Free the allocated memory for count array
    return result;
}






// Test the function
int main() {
    int data[] = {1, 2, 2, 4};
    int dataSize = sizeof(data) / sizeof(data[0]);
    int returnSize;

    int *result = getWrongAndCorrect(data, dataSize, &returnSize);

    printf("Incorrect element: %d\n", result[0]);
    printf("Correct element: %d\n", result[1]);

    free(result); // Free the allocated memory for result array

    return 0;
}






















































/*
    ~ PROBLEM: 907. Sum of Subarray Minimums

    ~ PROMPT: Can you generate a C++ function that accepts an integer vector "arr"? Name it "arrSumMins" and implement it such that it returns the sum of the smallest value elements from all possible subarrays of "arr". (switch it up a bit tho)
*/

/*  OG  */
// #include <iostream>
// #include <vector>
// #include <stack>
// #include <algorithm>

// int arrSumMins(const std::vector<int>& arr) 
// {
//     int n = arr.size();
//     std::vector<int> prevSmaller(n), nextSmaller(n);
//     std::stack<int> st;

//     // Calculate previous smaller elements
//     for (int i = 0; i < n; ++i) 
//     {
//         while (!st.empty() && arr[st.top()] >= arr[i]) 
//         {
//             st.pop();
//         }

//         prevSmaller[i] = st.empty() ? -1 : st.top();
//         st.push(i);
//     }

//     // Clear the stack for next use
//     while (!st.empty()) 
//     {
//         st.pop();
//     }

//     // Calculate next smaller elements
//     for (int i = n - 1; i >= 0; --i) 
//     {
//         while (!st.empty() && arr[st.top()] > arr[i]) 
//         {
//             st.pop();
//         }

//         nextSmaller[i] = st.empty() ? n : st.top();
//         st.push(i);
//     }

//     // Calculate the sum of the smallest values in all subarrays
//     long long result = 0;
//     const int MOD = 1e9 + 7; // Use modulo to prevent overflow
//     for (int i = 0; i < n; ++i) {
//         long long left = i - prevSmaller[i];
//         long long right = nextSmaller[i] - i;
//         result = (result + arr[i] * left % MOD * right % MOD) % MOD;
//     }

//     return result;
// }




/*  MB (fixed)  */
// #include <vector>
// #include <climits>
// #include <stack>
// #include <iostream>

// // Function to calculate sum of minimums of all subarrays
// int arrSumMins(const std::vector<int>& arr) 
// {
//     int n = arr.size();

//     // Left and right arrays to store the previous less element (PLE) and next less element (NLE) indices
//     std::vector<int> left(n), right(n);
//     std::stack<int> st;

//     // Find PLE for each element
//     for (int i = 0; i < n; i++) 
//     {
//         while (!st.empty() && arr[st.top()] >= arr[i]) 
//             st.pop();
        
//         if (!st.empty())
//             left[i] = st.top();
//         else
//             left[i] = -1;

//         st.push(i);
//     }

//     // Reset the stack for next use
//     while (!st.empty()) 
//         st.pop();

//     // Find NLE for each element
//     for (int i = n - 1; i >= 0; i--) 
//     {
//         while (!st.empty() && arr[st.top()] > arr[i]) 
//             st.pop();
        
//         if (!st.empty()) 
//             right[i] = st.top();
//         else
//             right[i] = n;

//         st.push(i);
//     }

//     // Calculate sum of minimums, avoiding overflow by using long long
//     long long sum = 0;

//     // Use modulo to prevent overflow
//     const int MOD = 1e9 + 7; 

//     for (int i = 0; i < n; i++) 
//     {
//         // Explicit cast to avoid overflow
//         long long l = (i - left[i]);
//         long long r = (right[i] - i);
//         sum = (sum + arr[i] * l * r) % MOD;
//     }

//     return sum;
// }




// int main() {
//     std::vector<int> arr = {3, 1, 2, 4};
//     std::cout << "Expected: 17\nActual: " << arrSumMins(arr) << std::endl;

//     return 0;
// }























































// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <map>
// #include <fstream>
// #include <sstream>



// void decode(const std::vector<std::pair<int, std::string>>& nums)
// {
//    size_t n = nums.size();

//    // Sorting the pairs based on the keys
//    std::vector<std::pair<int, std::string>> sortedNums = nums;
//    std::sort(sortedNums.begin(), sortedNums.end());

//    // Find the number of rows in the pyramid
//    size_t rows = 0;
//    while (rows * (rows + 1) / 2 < n) {
//         rows++;
//     }

//    size_t index = 0;
//    std::vector<int> rightMostKeys;
//    for (size_t i = 1; i <= rows; ++i) {
//         // Store the numbers
//         for (size_t j = 0; j < i; ++j) {
//             if (j == i - 1) {
//                 rightMostKeys.push_back(sortedNums[index].first);
//             }
//             index++;
//         }
//    }

//    // Display the values of the right-most keys i.e., the decoded message
//    for (int key : rightMostKeys) {
//         for (const auto& pair : sortedNums) {
//             if (pair.first == key) {
//                 std::cout << pair.second << " ";
//                 break;
//             }
//         }
//    }
// }


// std::vector<std::pair<int, std::string>> readFromFile(const std::string& filename) {
//     std::ifstream file(filename);
//     std::vector<std::pair<int, std::string>> nums;

//     // Quick error check
//     if (!file.is_open()) {
//         std::cerr << "Failed to open the file: " << filename << std::endl;
//         return nums;
//     }

//     std::string line;
//     while (getline(file, line)) {
//         std::istringstream iss(line);
//         int key;
//         std::string value;

//         if (iss >> key >> value) {
//             nums.push_back({key, value});
//         }
//     }

//     file.close();
//     return nums;
// }

// int main() {
//     std::vector<std::pair<int, std::string>> nums = readFromFile("small_message_file.txt");
//     decode(nums); 
//     return 0;
// }


// #include <iostream>
// #include <stack>
// #include <vector>
// #include <string>

// using namespace std;

// int countWaysToSplit(const std::string& s) {
//     int totalOnes = 0;
//     for (char c : s) {
//         if (c == '1') {
//             totalOnes++;
//         }
//     }

//     if (totalOnes % 3 != 0) {
//         return 0; // No way to split if not divisible by 3
//     }
//     if (totalOnes == 0) {
//         return 1; // Special case when there are no '1's
//     }

//     int onesPerPart = totalOnes / 3;
//     std::vector<int> onePositions;

//     // Find positions of every '1'
//     for (int i = 0; i < s.length(); ++i) {
//         if (s[i] == '1') {
//             onePositions.push_back(i);
//         }
//     }

//     // Calculate the number of '0's between the boundary '1's
//     int waysFirst = onePositions[onesPerPart] - onePositions[onesPerPart - 1];
//     int waysSecond = onePositions[2 * onesPerPart] - onePositions[2 * onesPerPart - 1];

//     return waysFirst * waysSecond;
// }

// int main() {
//     std::string s = "10101";
//     int ways = countWaysToSplit(s);
//     std::cout << "Number of ways to split: " << ways << std::endl;

//     return 0;
// }


