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


#include <iostream>
#include <stack>
#include <vector>
#include <string>

using namespace std;




int countWaysToSplit(const std::string& s) {
    int totalOnes = 0;
    for (char c : s) {
        if (c == '1') {
            totalOnes++;
        }
    }

    if (totalOnes % 3 != 0) {
        return 0; // No way to split if not divisible by 3
    }
    if (totalOnes == 0) {
        return 1; // Special case when there are no '1's
    }

    int onesPerPart = totalOnes / 3;
    std::vector<int> onePositions;

    // Find positions of every '1'
    for (int i = 0; i < s.length(); ++i) {
        if (s[i] == '1') {
            onePositions.push_back(i);
        }
    }

    // Calculate the number of '0's between the boundary '1's
    int waysFirst = onePositions[onesPerPart] - onePositions[onesPerPart - 1];
    int waysSecond = onePositions[2 * onesPerPart] - onePositions[2 * onesPerPart - 1];

    return waysFirst * waysSecond;
}

int main() {
    std::string s = "10101";
    int ways = countWaysToSplit(s);
    std::cout << "Number of ways to split: " << ways << std::endl;

    return 0;
}


