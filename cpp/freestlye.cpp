



// MA
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

const unsigned int SCREEN_WIDTH = 800;
const unsigned int SCREEN_HEIGHT = 600;

// Define the vertices of the complex shape
struct Vertex {
  glm::vec3 position;
  glm::vec3 color;
};

const Vertex vertices[] = {
  {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)},
  {glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.0f)},
  {glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 1.0f)},
  {glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f)},
  // Add more vertices to define the complex shape
  {glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 1.0f)},
  {glm::vec3(0.5f, 0.5f, 1.0f), glm::vec3(1.0f, 0.5f, 1.0f)},
  {glm::vec3(1.0f, 0.5f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f)},
  {glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.5f, 1.0f, 1.0f)},
  {glm::vec3(0.5f, 1.0f, 1.0f), glm::vec3(0.5f, 0.5f, 1.0f)},
};

const unsigned int vertex_count = sizeof(vertices) / sizeof(vertices[0]);

int main() {
  // Initialize GLFW
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Create a window
  GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Complex Shape Rendering", nullptr, nullptr);
  if (window == nullptr) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  // Initialize GLEW
  if (glewInit() != GLEW_OK) {
    std::cout << "Failed to initialize GLEW" << std::endl;
    glfwTerminate();
    return -1;
  }

  // Create a VAO
  unsigned int VAO;
  glGenVertexArrays(1, &VAO);

  // Create a VBO
  unsigned int VBO;
  glGenBuffers(1, &VBO);

  // Allocate memory for vertex data
  unsigned int vertex_buffer_size = sizeof(Vertex) * vertex_count;
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, vertex_buffer_size, nullptr, GL_STATIC_DRAW);

  // Copy the vertex data to the GPU
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // Create an index buffer
  unsigned int index_buffer_size = sizeof(unsigned int) * 6; // For a triangle strip
  unsigned int indices[] = {
    0, 1, 2,
    0, 2, 3,
    // Add more indices to define the complex shape
    4, 5, 6,
    4, 6, 7,
  };

  unsigned int index_buffer;
  glGenBuffers(1, &index_buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_buffer_size, indices, GL_STATIC_DRAW);

  // Define the vertex shader
  const char* vertex_shader_source = "#version 330 core\n"
                                    "layout (location = 0) in vec3 aPos;\n"
                                    "layout (location = 1) in vec3 aColor;\n"
                                    "out vec3 ourColor;\n"
                                    "void main()\n"
                                    "{\n"
                                    "  gl_Position = vec4(aPos, 1.0);\n"
                                    "  ourColor = aColor;\n"
                                    "}";

  // Define the fragment shader
  const char* fragment_shader_source = "#version 330 core\n"
                                      "out vec3 FragColor;\n"
                                      "void main()\n"
                                      "{\n"
                                      "  FragColor = ourColor;\n"
                                      "}";

  // Compile the shaders
  GLuint vertex_shader, fragment_shader;
  if (glCreateShader(GL_VERTEX_SHADER, vertex_shader_source, nullptr, &vertex_shader) != GL_TRUE) {
    std::cout << "Failed to create vertex shader" << std::endl;
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glfwTerminate();
    return -1;
  }

  if (glCreateShader(GL_FRAGMENT_SHADER, fragment_shader_source, nullptr, &fragment_shader) != GL_TRUE) {
    std::cout << "Failed to create fragment shader" << std::endl;
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    glfwTerminate();
    return -1;
  }

  // Link the shader program
  GLuint shader_program;
  if (glCreateProgram() != GL_TRUE) {
    std::cout << "Failed to create shader program" << std::endl;
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glfwTerminate();
    return -1;
  }

  if (glAttachShader(shader_program, vertex_shader) != GL_TRUE) {
    std::cout << "Failed to attach vertex shader" << std::endl;
    glDeleteProgram(shader_program);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_











// MB
// #include <SDL2/SDL.h>
// #include <SDL2/SDL_image.h>
// #include <SDL2/SDL_ttf.h>

// #include <iostream>
// #include <cmath>

// int main(int argc, char* argv[]) {
//     // Initialize SDL
//     if (SDL_Init(SDL_INIT_VIDEO) < 0) {
//         std::cout << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
//         return -1;
//     }

//     // Create window
//     SDL_Window* window = SDL_CreateWindow("Complex Shape Rendering", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_SHOWN);
//     if (window == NULL) {
//         std::cout << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
//         return -1;
//     }

//     // Create renderer
//     SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
//     if (renderer == NULL) {
//         std::cout << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
//         return -1;
//     }

//     // Set texture filtering to linear
//     SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");

//     // Load image
//     SDL_Surface* imageSurface = IMG_Load("path/to/your/image.png");
//     if (imageSurface == NULL) {
//         std::cout << "Unable to load image: " << SDL_GetError() << std::endl;
//         return -1;
//     }

//     // Create texture from surface
//     SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, imageSurface);
//     SDL_FreeSurface(imageSurface);

//     // Set the position of the texture
//     SDL_Rect rect = { 100, 100, imageSurface->w, imageSurface->h };

//     // Render the texture
//     SDL_RenderCopy(renderer, texture, NULL, &rect);

//     // Clean up
//     SDL_DestroyTexture(texture);
//     SDL_DestroyRenderer(renderer);
//     SDL_DestroyWindow(window);
//     SDL_Quit();

//     return 0;
// }










































/*
    ~ PROBLEM: 1143. Longest Common Subsequence

    ~ PROMPT: Can you implement a c function that analyzes strings and finds their intersection? Name the function "int intersectInOrder(string set1, string set2)" such that the function accepts two strings, "set1" and "set2". The function should then find the longest string that is an intersection of set1 and set2 without changing the relative order of set1 and set2's elements. If none is found, then just return zero.
*/


/*  OG  */
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// int intersectInOrder(char *set1, char *set2) {
//     int len1 = strlen(set1);
//     int len2 = strlen(set2);

//     // Create a 2D array to store lengths of longest common subsequence
//     int **dp = (int **)malloc((len1 + 1) * sizeof(int *));
//     for (int i = 0; i <= len1; i++) {
//         dp[i] = (int *)malloc((len2 + 1) * sizeof(int));
//     }

//     // Fill dp array using dynamic programming
//     for (int i = 0; i <= len1; i++) {
//         for (int j = 0; j <= len2; j++) {
//             if (i == 0 || j == 0) {
//                 dp[i][j] = 0;
//             } else if (set1[i - 1] == set2[j - 1]) {
//                 dp[i][j] = dp[i - 1][j - 1] + 1;
//             } else {
//                 dp[i][j] = dp[i - 1][j] > dp[i][j - 1] ? dp[i - 1][j] : dp[i][j - 1];
//             }
//         }
//     }

//     int result = dp[len1][len2];

//     // Free allocated memory
//     for (int i = 0; i <= len1; i++) {
//         free(dp[i]);
//     }
//     free(dp);

//     return result;
// }







/*  MA  */
// #include <stdio.h>
// #include <string.h>

// int intersectInOrder(char* set1, char* set2) 
// {
//     int m = strlen(set1);
//     int n = strlen(set2);

//     // Dynamic Programming table to store longest common substring lengths
//     int dp[m + 1][n + 1]; 
//     memset(dp, 0, sizeof(dp));

//     int maxLength = 0; // Length of the longest common substring
//     int endIndex = -1;  // Index of the last character of the longest common substring in set1

//     // Fill the DP table
//     for (int i = 1; i <= m; i++) {
//         for (int j = 1; j <= n; j++) {
//             if (set1[i - 1] == set2[j - 1]) {
//                 dp[i][j] = dp[i - 1][j - 1] + 1; // Extend the common substring
//                 if (dp[i][j] > maxLength) {
//                     maxLength = dp[i][j];
//                     endIndex = i; // Update the end index
//                 }
//             } else {
//                 dp[i][j] = 0; // Reset the length if characters don't match
//             }
//         }
//     }

//     if (maxLength > 0) {
//         // Extract the longest common substring
//         char result[maxLength + 1];
//         strncpy(result, set1 + endIndex - maxLength, maxLength);
//         result[maxLength] = '\0'; 
//         printf("Longest common substring: %s\n", result);
//     }

//     return maxLength;
// }









/*  MB  */
// #include <stdio.h>
// #include <string.h>

// int intersectInOrder(char set1[], char set2[]) {
//     int m = strlen(set1);
//     int n = strlen(set2);
//     int dp[m + 1][n + 1]; 

//     // Base case: empty strings have no intersection
//     for (int i = 0; i <= m; i++) {
//         dp[i][0] = 0;
//     }
//     for (int j = 0; j <= n; j++) {
//         dp[0][j] = 0;
//     }

//     // Dynamic programming to find longest common subsequence (LCS)
//     for (int i = 1; i <= m; i++) {
//         for (int j = 1; j <= n; j++) {
//             if (set1[i - 1] == set2[j - 1]) {
//                 dp[i][j] = dp[i - 1][j - 1] + 1;
//             } else {
//                 dp[i][j] = (dp[i - 1][j] > dp[i][j - 1]) ? dp[i - 1][j] : dp[i][j - 1];
//             }
//         }
//     }

//     // Backtrack to extract the intersection string
//     int length = dp[m][n]; // Length of the intersection
//     char intersection[length + 1]; // Array to store the intersection
//     intersection[length] = '\0'; // Null-terminate the string

//     int i = m, j = n;
//     while (i > 0 && j > 0) {
//         if (set1[i - 1] == set2[j - 1]) {
//             intersection[length - 1] = set1[i - 1];
//             i--;
//             j--;
//             length--;
//         } else if (dp[i - 1][j] > dp[i][j - 1]) {
//             i--;
//         } else {
//             j--;
//         }
//     }

//     // Print (or use) the intersection string 
//     printf("Intersection: %s\n", intersection); 

//     return dp[m][n]; // Return the length of the intersection
// }












// // Test the function
// int main() {
//     char set1[] = "rabcd";
//     char set2[] = "race";
//     printf("Longest common subsequence length: %d\n", intersectInOrder(set1, set2)); // Expected output: 3

//     char set1_2[] = "abc";
//     char set2_2[] = "def";
//     printf("Longest common subsequence length: %d\n", intersectInOrder(set1_2, set2_2)); // Expected output: 0

//     return 0;
// }























































/*
    ~ PROBLEM: 1457. Pseudo-Palindromic Paths in a Binary Tree

    ~ PROMPT: Can you implement a c++ function that detects specific sequences or paths within a binary tree? Name the function "int findSymmPaths(TreeNode *root)" such that it accepts a binary tree and analyzes the tree's root-to-leaf paths to find the total number of "symmetric paths", where "symmetric paths" are paths that are made up of nodes with values that can be rearranged to form a sequence that reads the same backwards as forwards. Note that I have already implemented a TreeNode struct where each node has three member variables (int val, TreeNode *left, TreeNode *right) and three member functions that construct nodes with or without children and values. 
*/


/*  OG  */
// #include <iostream>
// #include <vector>
// #include <unordered_map>
// #include <algorithm>
// using namespace std;


// // Definition for a binary tree node.
// struct TreeNode 
// {
//     int val;
//     TreeNode *left;
//     TreeNode *right;
//     TreeNode() : val(0), left(nullptr), right(nullptr) {}
//     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
//     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
// };
 


// class Solution 
// {
// public:
//     int findSymmPaths(TreeNode* root) 
//     {
//         if (!root) return 0;
//         unordered_map<int, int> freq;
//         return dfs(root, freq);
//     }

// private:
//     int dfs(TreeNode* node, unordered_map<int, int>& freq) 
//     {
//         if (!node) return 0;

//         // Increment the frequency of the current node's value
//         freq[node->val]++;

//         int count = 0;

//         // Check if it is a leaf node
//         if (!node->left && !node->right) 
//         {
//             // Check if the current path is symmetric
//             if (isSymmetric(freq)) 
//                 count = 1;
//         } 
//         else 
//         {
//             // Continue DFS on the children
//             count += dfs(node->left, freq);
//             count += dfs(node->right, freq);
//         }

//         // Decrement the frequency of the current node's value before backtracking
//         freq[node->val]--;

//         return count;
//     }

//     bool isSymmetric(const unordered_map<int, int>& freq) 
//     {
//         int oddCount = 0;
//         for (const auto& pair : freq) 
//         {
//             if (pair.second % 2 != 0) 
//                 oddCount++;
    
//             if (oddCount > 1) 
//                 return false;
//         }

//         return true;
//     }
// };





// // Example usage
// int main() 
// {
//     // Construct a test binary tree
//     TreeNode* root = new TreeNode(3);
//     root->left = new TreeNode(4);
//     root->right = new TreeNode(2);
//     root->left->left = new TreeNode(4);
//     root->left->right = new TreeNode(2);
//     // NULL
//     root->right->right = new TreeNode(2);

//     Solution solution;
//     cout << "Number of symmetric paths: " << solution.findSymmPaths(root) << endl;

//     return 0;
// }
























































/*
    THIS WAS USED IN A MODIFICATION TASK SO YOU MIGHT BE ABLE TO REUSE IT FOR A CODE CREAION TASK, SINCE THE PROMPT WASNT USED. IT WAS USED ONLY FOR GPT.

    ~ PROBLEM: 1239. Maximum Length of a Concatenated String with Unique Characters

    ~ PROMPT: Can you please implement a c++ function that returns and maximizes the length of a newly formed string that's made up of elements from an array of strings? Name the function "int maximizeCat(vector<string>& pool)" such that it accepts the array of strings as an argument named "pool", where "pool" represents a set of strings. The function must utilize and concatenate pool's elements to create a new string called "cat" such that "cat" has no duplicate characters and is a subsequence of "pool". Finally, the function should return the max length possible for "cat".
*/


/*  OG  */
// #include <iostream>
// #include <vector>
// #include <unordered_set>

// using namespace std;

// class Solution 
// {
// public:
//     int maximizeCat(vector<string>& pool) 
//         {return backtrack(pool, "", 0);}

// private:
//     int backtrack(const vector<string>& pool, string current, int index) 
//     {
//         if (!isUnique(current))
//             return 0;

//         int maxLength = current.length();
//         for (int i = index; i < pool.size(); ++i) 
//             maxLength = max(maxLength, backtrack(pool, current + pool[i], i + 1));

//         return maxLength;
//     }

//     bool isUnique(const string& str) 
//     {
//         unordered_set<char> charSet;
//         for (char ch : str) 
//         {
//             if (charSet.count(ch) > 0) 
//                 return false;

//             charSet.insert(ch);
//         }

//         return true;
//     }
// };









// class Solution 
// {
// public:
//     int maximizeCat(vector<string>& pool) 
//     {
//         return backtrack(pool, "", 0);
//     }

// private:
//     int backtrack(const vector<string>& pool, string current, int index) 
//     {
//         unordered_set<char> charSet;
//         for (char ch : current) 
//         {
//             if (charSet.count(ch) > 0) 
//                 return 0;

//             charSet.insert(ch);
//         }

//         int maxLength = current.length();
//         for (int i = index; i < pool.size(); ++i) 
//             maxLength = max(maxLength, backtrack(pool, current + pool[i], i + 1));

//         return maxLength;
//     }
// };











// // Test the function
// int main() {
//     Solution solution;
//     vector<string> pool = {"zn", "iq", "ze"};
//     cout << "Max length of cat: " << solution.maximizeCat(pool) << endl; // Expected output: 4

//     pool = {"cho", "p", "oct", "eps"};
//     cout << "Max length of cat: " << solution.maximizeCat(pool) << endl; // Expected output: 6

//     pool = {"abcdefghijklmnopqrstuvwxyz"};
//     cout << "Max length of cat: " << solution.maximizeCat(pool) << endl; // Expected output: 26

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






// // Test the function
// int main() {
//     int data[] = {1, 2, 2, 4};
//     int dataSize = sizeof(data) / sizeof(data[0]);
//     int returnSize;

//     int *result = getWrongAndCorrect(data, dataSize, &returnSize);

//     printf("Incorrect element: %d\n", result[0]);
//     printf("Correct element: %d\n", result[1]);

//     free(result); // Free the allocated memory for result array

//     return 0;
// }






















































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
//             st.pop();

//         prevSmaller[i] = st.empty() ? -1 : st.top();
//         st.push(i);
//     }

//     // Clear the stack for next use
//     while (!st.empty()) 
//         st.pop();

//     // Calculate next smaller elements
//     for (int i = n - 1; i >= 0; --i) 
//     {
//         while (!st.empty() && arr[st.top()] > arr[i]) 
//             st.pop();

//         nextSmaller[i] = st.empty() ? n : st.top();
//         st.push(i);
//     }

//     // Calculate the sum of the smallest values in all subarrays
//     long long result = 0;
//     const int MOD = 1e9 + 7; // Use modulo to prevent overflow
//     for (int i = 0; i < n; ++i) 
//     {
//         long long left = i - prevSmaller[i];
//         long long right = nextSmaller[i] - i;
//         result = (result + arr[i] * left * right) % MOD;
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


