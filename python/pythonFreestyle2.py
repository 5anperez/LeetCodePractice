'''
    GENERATOR EXPRESSION:
    - A COOL WAY OF CHECKING FOR SET MEMBERSHIP WHILE TRAVERSING A 2D ARRAY!
'''
# def destCity(self, paths: List[List[str]]) -> str:
    
#     """
#         Finds the destination city in a line graph of paths.

#         Args:
#             paths: A list of paths, where each path is a list of two cities.
#                 [cityA_i, cityB_i] means there's a direct path from cityA_i to cityB_i.

#         Constraints:
#             1 <= paths.length <= 100, paths[i].length == 2
#             1 <= cityA_i.length, cityB_i.length <= 10
#             cityA_i != cityB_i
#             All strings consist of English letters and spaces only.

#         Returns:
#             The destination city, which is a city without any outgoing paths.
#     """

#     # Heres a cool way of checking for set membership while 
#     # traversing a 2D list. 
#     starts = set(path[0] for path in paths)
#     return next(path[1] for path in paths if path[1] not in starts)






# # Read and process the lines within the with block
# with open("message_file.txt") as file:
#     # Get the first 5 lines, removing newlines using a list comprehension
#     first_5_lines = [line.strip() for line in itertools.islice(file, 5)]

# # Now you can safely print or process the lines outside the with block
# for line in first_5_lines:
#     print(line)  # Output without newline characters


# with open("message_file.txt") as file:
#     first_5_lines = (line.strip() for line in itertools.islice(file, 5))  

#     # Print or process the lines as needed
#     for line in first_5_lines:
#         print(line)  # Output without newline characters

# with open("message_file.txt") as file:
#     for index, line in enumerate(file):
#         if index == 5:
#             break
#         print(line.strip())







# def restoreMatrix(rowSum, colSum):
#   """
#   Creates a 2D matrix from its row and column sum requirements.

#   Args:
#     rowSum: List of non-negative integers representing the row sums.
#     colSum: List of non-negative integers representing the column sums.

#   Returns:
#     A 2D list representing a matrix fulfilling the requirements.
#   """

#   # Define matrix dimensions
#   rows, cols = len(rowSum), len(colSum)

#   # Initialize the matrix with zeros
#   matrix = [[0] * cols for _ in range(rows)]

#   # Iterate through each cell
#   for i in range(rows):
#     for j in range(cols):
#       # Calculate the minimum value for the current cell
#       minimum = min(rowSum[i], colSum[j])

#       # Update the row and column sums
#       rowSum[i] -= minimum
#       colSum[j] -= minimum

#       # Update the matrix
#       matrix[i][j] = minimum

#   # Check for remaining values in row or column sums
#   for i in range(rows):
#     if rowSum[i] > 0:
#       matrix[i][-1] += rowSum[i]

#   for j in range(cols):
#     if colSum[j] > 0:
#       matrix[-1][j] += colSum[j]

#   return matrix








# def restoreMatrix(rowSum, colSum):
#     n, m = len(rowSum), len(colSum)

#     def dfs(i, j, remainingRowSum, remainingColSum):
#         if i == n and j == m:
#             return True

#         if remainingRowSum < 0 or remainingColSum < 0:
#             return False

#         if remainingRowSum == 0 and remainingColSum == 0:
#             return True

#         if remainingRowSum < colSum[j] or remainingColSum < rowSum[i]:
#             return False

#         matrix[i][j] = min(remainingRowSum, remainingColSum)
#         remainingRowSum -= matrix[i][j]
#         remainingColSum -= matrix[i][j]

#         if dfs(i, j + 1, remainingRowSum, remainingColSum):
#             return True

#         if dfs(i + 1, j, remainingRowSum, remainingColSum):
#             return True

#         matrix[i][j] = 0
#         remainingRowSum += matrix[i][j]
#         remainingColSum += matrix[i][j]

#         return False

#     matrix = [[0 for _ in range(m)] for _ in range(n)]
#     dfs(0, 0, sum(rowSum), sum(colSum))
#     return matrix





# def restoreMatrix(rowSum, colSum):
#     n, m = len(rowSum), len(colSum)
#     matrix = [[0 for _ in range(m)] for _ in range(n)]

#     def dfs(i, j, remainingRowSum, remainingColSum):
#         if i >= n or j >= m:
#             return False

#         if i == n - 1 and j == m - 1:
#             matrix[i][j] = min(remainingRowSum, remainingColSum)
#             return matrix[i][j] == remainingRowSum and matrix[i][j] == remainingColSum

#         originalVal = matrix[i][j]
#         for val in range(min(remainingRowSum, remainingColSum) + 1):
#             matrix[i][j] = val
#             if (j < m - 1 and dfs(i, j + 1, remainingRowSum - val, colSum[j + 1])) or \
#                (i < n - 1 and dfs(i + 1, j, rowSum[i + 1], remainingColSum - val)):
#                 return True

#         matrix[i][j] = originalVal
#         return False

#     if dfs(0, 0, rowSum[0], colSum[0]):
#         return matrix
#     else:
#         return "No valid matrix found"







# Expected: [[3, 0], [1, 7]]
# Expected A: [[3, 0], [1, 7]]
# Expected B: [[11, 0], [0, 0]]
# rowSum = [3, 8]
# colSum = [4, 7]

# Expected: [[5, 0, 0], [3, 4, 0], [0, 2, 8]]
# Expected A: [[5, 0, 0], [3, 4, 0], [0, 2, 8]]
# Expected B: [[22, 0, 0], [0, 0, 0], [0, 0, 0]]
# rowSum = [5,7,10]
# colSum = [8,6,8]
# print(restoreMatrix(rowSum, colSum))
        










'''
    GPT CANNOT SOLVE THIS ONE!!!! ITS A PASSWORD CHECKER AND 
    THE FOLLOWING TEST CASES ARE TRIPPING IT UP:
    password = "bbaaaaaaaaaaaaaaacccccc" and "aaa111"
'''


# class Solution:def strongPasswordChecker(password: str) -> int:
#     n = len(password)
#     has_lower = any(c.islower() for c in password)
#     has_upper = any(c.isupper() for c in password)
#     has_digit = any(c.isdigit() for c in password)

#     # Count missing types of characters
#     missing_types = 3 - (has_lower + has_upper + has_digit)

#     # Count repeating sequences
#     repeats = []
#     i = 2
#     while i < n:
#         if password[i] == password[i - 1] == password[i - 2]:
#             length = 2
#             while i < n and password[i] == password[i - 1]:
#                 length += 1
#                 i += 1
#             repeats.append(length)
#         else:
#             i += 1

#     # Handle different length cases
#     if n < 6:
#         return max(missing_types, 6 - n)
#     elif n <= 20:
#         change = 0
#         for r in repeats:
#             change += r // 3
#         return max(missing_types, change)
#     else:
#         delete = n - 20
#         for i in range(3):
#             for j in range(len(repeats)):
#                 if repeats[j] and repeats[j] % 3 == i:
#                     repeats[j] -= min(delete, i + 1)
#                     delete -= i + 1
#                     if repeats[j] < 3:
#                         change -= 1
#         change = sum(r // 3 for r in repeats)
#         return delete + max(missing_types, change)

# # Example usage
# print(strongPasswordChecker("aaa111"))  # Output should be 2








'''
    ******************* DYNAMIC PROGRAMMING PROBLEMS *******************
'''








'''
    Given a string s and a dictionary of strings wordDict, return true if s can 
    be segmented into a space-separated sequence of one or more dictionary words.

    Note that the same word in the dictionary may be reused multiple times 
    in the segmentation.

    NOTE: for this problem, try to figure out a way to swap the loops, i.e.,
    instead of traversing the 's' in the outer, traverse the Dict instead...

    NOTE: Is this true: "You would end up checking the same substrings of s 
    multiple times for different words in wordDict. This is inefficient, 
    especially if wordDict contains a large number of words."?
'''


# def wordBreak(s: str, wordDict: List[str]) -> bool:
#     wordSet = set(wordDict)  # Convert list to set for faster lookup
#     dp = [False] * (len(s) + 1)
#     dp[0] = True  # Empty string can always be segmented

#     for i in range(1, len(s) + 1):
#         for j in range(i):
#             if dp[j] and s[j:i] in wordSet:
#                 dp[i] = True
#                 break

#     return dp[len(s)]


# # DIFFERENT APPROACH HERE!!!!!!!
# def wordBreak(s: str, wordDict: List[str]) -> bool:
#     wordSet = set(wordDict)  # Convert list to set for faster lookup
#     dp = [False] * (len(s) + 1)
#     dp[0] = True  # Empty string can always be segmented

#     for word in wordSet:
#         for i in range(len(word), len(s) + 1): # I THINK THIS RANGE IS INCORRECT
#             if dp[i - len(word)] and s[i - len(word):i] == word:
#                 dp[i] = True

#     return dp[len(s)]



# # THIS MIGHT BE MORE ACCURATE!!!!!
# def wordBreak(self, s: str, wordDict: List[str]) -> bool:
#         dp = [False] * (len(s) + 1)
#         dp[0] = True

#         for word in wordDict:  # Outer loop over words in wordDict
#             for i in range(len(s) - len(word) + 1):  # Inner loop over possible start indices
#                 if s[i: i + len(word)] == word and dp[i]:
#                     dp[i + len(word)] = True

#         return dp[len(s)]














'''
    THIS ONE IS GOOD PRACTICE! DO THIS ONE BY
    HAND SO YOU CAN SEE THE 3D TABLE AND HOW
    IT IS BUILT!
'''


# def knightProbability(n: int, k: int, row: int, column: int) -> float:
#     # Directions a knight can move
#     directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

#     # Check if the knight is on the board
#     def is_on_board(x, y):
#         return 0 <= x < n and 0 <= y < n

#     # Dynamic programming table
#     dp = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(k + 1)]
#     dp[0][row][column] = 1  # Starting position

#     for step in range(1, k + 1):
#         for i in range(n):
#             for j in range(n):
#                 for dx, dy in directions:

#                     # Calculate where the knight would end up and if its
#                     # still on the board, then assign this cell a probability.
#                     x, y = i + dx, j + dy
#                     if is_on_board(x, y):

#                         # Since these are sequential steps/moves, the law of probab
#                         dp[step][i][j] += dp[step - 1][x][y] / 8

#     # Sum the probabilities of being on any cell after k moves
#     return sum(dp[k][i][j] for i in range(n) for j in range(n))

# # Example usage
# print(knightProbability(8, 3, 0, 0))  # Example: 8x8 board, 1 move, starting at (0, 0)






# def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
#     directions = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

#     def is_valid(x, y):
#         return 0 <= x < n and 0 <= y < n

#     dp = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(k + 1)]
#     dp[0][row][column] = 1.0  # Base case: Probability is 1 at the starting position

#     for step in range(1, k + 1):
#         for i in range(n):
#             for j in range(n):
#                 for dx, dy in directions:
#                     x, y = i + dx, j + dy
#                     if is_valid(x, y):
#                         dp[step][i][j] += dp[step - 1][x][y] / 8  # 1/8 chance for each move

#     return sum(dp[k][i][j] for i in range(n) for j in range(n))







'''
You are given an integer array nums. Two players are playing a game with this array: player 1 and player 2.

Player 1 and player 2 take turns, with player 1 starting first. Both players start the game with a score of 0. At each turn, the player takes one of the numbers from either end of the array (i.e., nums[0] or nums[nums.length - 1]) which reduces the size of the array by 1. The player adds the chosen number to their score. The game ends when there are no more elements in the array.

Return true if Player 1 can win the game. If the scores of both players are equal, then player 1 is still the winner, and you should also return true. You may assume that both players are playing optimally.


NOTE:
- In a zero-sum game like this, each player's goal is to maximize their own score relative to the other player. This can be modeled as maximizing the score difference between the players.
- Instead of tracking scores separately for Player 1 and Player 2, we can track the score difference from Player 1's perspective. A positive score difference means Player 1 is leading, while a negative score difference means Player 2 is leading.
- At each turn, the player whose turn it is will choose the option that maximizes their score difference.
'''



# def PredictTheWinner(nums):
#     def calculate_scores(start, end):
#         if start == end:
#             return nums[start]
#         if memo[start][end] != -1:
#             return memo[start][end]

#         # Player's choice at the start or end of the array
#         pick_start = nums[start] - calculate_scores(start + 1, end)
#         pick_end = nums[end] - calculate_scores(start, end - 1)

#         # Store the best outcome for the current player
#         memo[start][end] = max(pick_start, pick_end)
#         return memo[start][end]

#     n = len(nums)
#     memo = [[-1 for _ in range(n)] for _ in range(n)]
#     return calculate_scores(0, n - 1) >= 0

# # Example usage
# print(PredictTheWinner([1, 5, 2]))  # Output: False
# print(PredictTheWinner([1, 5, 233, 7]))  # Output: True












'''
    ******************* DYNAMIC PROGRAMMING PROBLEMS *******************
'''


# from typing import List

# def combine(n: int, k: int) -> List[List[int]]:
#     def backtrack(start, path):
#         # If the combination is of length k, add it to the result
#         if len(path) == k:
#             result.append(path.copy())
#             return
        
#         # Try adding each number to the combination and backtrack
#         for i in range(start, n + 1):
#             path.append(i)
#             backtrack(i + 1, path)
#             path.pop()  # Remove the last element to backtrack

#     result = []
#     backtrack(1, [])
#     return result

# # Example usage
# print(combine(4, 2))  # Output: [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
# print(combine(1, 1))  # Output: [[1]]
# print(combine(5, 3))  # Output: [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5], [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]




# import math
# from typing import List

# def getFactors(n: int) -> List[List[int]]:
#     # base case
#     if n == 1: return []

#     res = []

#     def dfs(path=[], rest=2, target=n):
#         if len(path)>0:
#             res.append(path+[target])
#         # i <= target // i, i.e., i <= sqrt(target)
#         for i in range(rest, int(math.sqrt(target))+1): 
#             if target % i == 0:
#                 dfs(path + [i], i, target // i)
#     dfs()
#     return res


# Example usage
# print(getFactors(16))  # Output: [[2, 8], [2, 2, 4], [2, 2, 2, 2], [4, 4]]
# print(getFactors(12)) # Output: [[2, 2, 3], [2, 6], [3, 4]]


# from itertools import product
# def getFactors(n: int) -> List[List[int]]:
#     factors = []
#     for i in range(2, n):
#         if n % i == 0:
#             factors.append(i)

#     factor_combinations = []
#     def backtrack(current_combination, remaining_factors):
#         current_product = math.prod(current_combination)
#         if current_product == n:
#             # Store copy to avoid modification
#             factor_combinations.append(current_combination.copy())  
#             return

#         for i in range(len(remaining_factors)):
#             next_factor = remaining_factors[i]
#             # Early pruning for efficiency
#             if current_product * next_factor <= n:  
#                 current_combination.append(next_factor)
#                 backtrack(current_combination, remaining_factors[i+1:])
#                 current_combination.pop()

#     backtrack([], factors)
#     return factor_combinations





# # Example usage
# print(getFactors(16))  # Output: [[2, 8], [2, 2, 4], [2, 2, 2, 2], [4, 4]]
# print(getFactors(12)) # Output: [[2, 2, 3], [2, 6], [3, 4]]




















# user_input = input('Enter your Name: ')

# print("trial 1: the long way")

# if user_input:
#     name = user_input
# else:
#     name = 'N/A'
    
# print(name)

# print("trial 2: quick & correct")
# name = user_input or "N/A"  # Assigns user_input if it has a value, otherwise assigns "N/A"
# print(name)

# print("trial 3: quick & correct")
# print(name := user_input if user_input else "N/A")


# print("trial 4: quick & correct")
# print((lambda: user_input, lambda: "N/A")[not user_input]())


# print("trial 5: quick")
# name = [user_input if user_input else 'N/A'][0]
# print(name)


# print("trial 6: quick")
# name_dict = {True: user_input, False: 'N/A'} 
# name = name_dict[bool(user_input)]
# print(name)












'''
    Two versions of finding the first AND last occurrence of an element
'''

# def findFirstAndLast(arr, target):
#     def binarySearchFirst(arr, target):
#         left, right = 0, len(arr) - 1
#         while left <= right:
#             mid = left + (right - left) // 2
#             if arr[mid] > target:
#                 right = mid - 1
#             elif arr[mid] < target:
#                 left = mid + 1
#             else:
#                 if mid == 0 or arr[mid - 1] != target:
#                     return mid
#                 right = mid - 1
#         return -1

#     def binarySearchLast(arr, target):
#         left, right = 0, len(arr) - 1
#         while left <= right:
#             mid = left + (right - left) // 2
#             if arr[mid] > target:
#                 right = mid - 1
#             elif arr[mid] < target:
#                 left = mid + 1
#             else:
#                 if mid == len(arr) - 1 or arr[mid + 1] != target:
#                     return mid
#                 left = mid + 1
#         return -1

#     return [binarySearchFirst(arr, target), binarySearchLast(arr, target)]

# # Example usage
# arr = [1, 2, 2, 2, 3, 4, 5, 2]
# target = 2
# print(findFirstAndLast(arr, target))  # Output: [1, 3] bc its not completely sorted

# arr1 = [5, 7, 7, 8, 8, 10]
# target1 = 8
# print(findFirstAndLast(arr1, target1))  # Output: [3, 4]






# def find_first_and_last(nums, target):
#     def find_first():
#         left, right = 0, len(nums) - 1
#         result = -1  # Set to -1 if target is not found

#         while left <= right:
#             mid = (left + right) // 2
#             if nums[mid] == target:
#                 result = mid  # Potential first occurrence (keep searching left)
#                 right = mid - 1
#             elif nums[mid] < target:
#                 left = mid + 1
#             else:
#                 right = mid - 1

#         print("Left:", left)
#         return result

#     def find_last():  # Similar implementation to find_first()
#         left, right = 0, len(nums) - 1
#         result = -1  # Set to -1 if target is not found

#         while left <= right:
#             mid = (left + right) // 2
#             if nums[mid] == target:
#                 result = mid 
#                 left = mid + 1
#             elif nums[mid] < target:
#                 left = mid + 1
#             else:
#                 right = mid - 1

#         print("Right:", right)
#         return result

#     return (find_first(), find_last())
    

# # Example usage
# nums = [2, 4, 5, 5, 5, 5, 7, 8]
# target = 5
# first, last = find_first_and_last(nums, target)
# print("First Occurrence:", first)
# print("Last Occurrence:", last)





# nums=range(1,1000)

# def pr(num):
#     for x in range(2,num):
#         if (num%x) == 0:
#             return False
#     return True
    



# def pr(num):
#   """ Optimized function to check for prime numbers. """
#   if num <= 1:
#     return False  # Numbers less than or equal to 1 are not prime.
#   elif num <= 3:
#     return True  # 2 and 3 are prime.
#   elif num % 2 == 0 or num % 3 == 0:
#     return False  # Divisible by 2 or 3, not prime.

#   # Check for divisibility by numbers up to the square root of the number.
#   limit = int(num**0.5) + 1
#   for i in range(5, limit, 6):  # Only check 6k ± 1, as 6k ± 2 will always be divisible by 2.
#     if num % i == 0 or num % (i+2) == 0:
#       return False
#   return True




    







# from typing import List
# from collections import Counter

# class Solution:
#     def smallSubset(self, s: str, t: str) -> str:
#         if not t or not s:
#             return ""

#         # Count characters in t
#         t_count = Counter(t)
#         required = len(t_count)

#         # Initialize window pointers and counts
#         left, right = 0, 0
#         formed = 0
#         window_counts = Counter()

#         # Variables to keep track of the minimum window
#         min_length = float("inf")
#         min_left, min_right = 0, 0

#         while right < len(s):
#             # Add one character from the right to the window
#             character = s[right]
#             window_counts[character] += 1

#             if character in t_count and window_counts[character] == t_count[character]:
#                 formed += 1

#             # Try and contract the window till the point where it ceases to be 'desirable'
#             while left <= right and formed == required:
#                 character = s[left]

#                 # Save the smallest window until now
#                 if right - left + 1 < min_length:
#                     min_length = right - left + 1
#                     min_left, min_right = left, right

#                 window_counts[character] -= 1
#                 if character in t_count and window_counts[character] < t_count[character]:
#                     formed -= 1

#                 left += 1

#             right += 1

#         return "" if min_length == float("inf") else s[min_left:min_right + 1]

# # Example usage
# solution = Solution()
# # s = "ADOBECODEBANC"
# # t = "ABC"
# # print(solution.smallSubset(s, t))  # Output: "BANC"

# s = "abcdeabfghi"
# t = "bdefghi"
# print(solution.smallSubset(s, t))  # Output: "deabfghi"




# from typing import List

# def generate_unique_subsets(nums: List[int]) -> List[List[int]]:
#     # Calculate the length of the input list
#     num_elements = len(nums)

#     # Sort the list to ensure duplicates are adjacent
#     nums.sort()

#     # Initialize a set to store unique subsets
#     unique_subsets = set()

#     # Iterate over all possible combinations using binary representation
#     for subset_indicator in range(1 << num_elements):
#         current_subset = []

#         # Check each element to see if it should be included in the current subset
#         for index in range(num_elements):
#             # Check if the bit at position 'index' is set in 'subset_indicator'
#             if (subset_indicator >> index) & 1:
#                 current_subset.append(nums[index])

#         # Add the current subset as a tuple to the set of unique subsets
#         unique_subsets.add(tuple(current_subset))

#     # Convert the set of tuples back to a list of lists and return
#     return list(map(list, unique_subsets))




# # Example usage
# print(generate_unique_subsets([5, 7, 9, 5, 1]))








'''
    *******************************************************************************************************************************************

    START OF THE PROMPT + UNIT TESTS PROJECT!

    Come back here and add comments to all these methods for good practice! Try to break the DP problems down, i.e., what are the smaller problems that overlap, which is step 1 of creating the DP solute? Can you see some pattern amongst multiple probs?

    *******************************************************************************************************************************************
'''










'''
    91. Decode Ways

    Can you write a function called "def possibleMessages(self, s: str) -> int:" that takes a string 's' and calculates the number of possible partitions, i.e., the function partitions 's' evenly into groups of integers in the range [1, 26] (inclusive), where each partition or group is made from the elements in 's', and returns the number of possible ways to partition 's'.

    Since there are only two subproblems that over lap, we can sum the ways bc its an OR situation...
'''



# def possibleMessages(msg: str) -> int:
#     if not msg or msg[0] == '0':
#         return 0

#     dp = [0] * (len(msg) + 1)
#     dp[0], dp[1] = 1, 1 

#     for i in range(2, len(msg) + 1):        
#         if msg[i - 1] != '0':
#             dp[i] += dp[i - 1]

#         two_digit = int(msg[i - 2:i])
#         if 10 <= two_digit <= 26:
#             dp[i] += dp[i - 2]

#     return dp[len(msg)]



# assert possibleMessages("06") == 0
# assert possibleMessages("17") == 2
# assert possibleMessages("22208") == 2
# assert possibleMessages("26260") == 0





# Example usage
# print(possibleMessages("22208"))  # Example input










'''
    93. Restore IP Addresses

    Write me a function that takes in a string of numbers and returns all possible IP addresses that can be formed by inserting dots into 's' without reordering or removing any digits in 's'. Name the function "def ipDistribute(self, s: str) -> List[str]:"

    It seems like the recursive strategy is best here since we know the worst case recursive call stack will be, more or less. Each segment can be a 
'''


# from typing import list

# def ipDistribute(raw: str) -> list[str]:
#     def is_valid(segment):
#         return len(segment) == 1 or (segment[0] != '0' and int(segment) <= MAX_SEGMENT_VAL)

#     def backtrack(start, path):
#         if len(path) == NUM_SEGMENTS:
#             if start == len(raw):
#                 result.append('.'.join(path))
#             return

#         for end in range(start + 1, len(raw) + 1):
#             segment = raw[start:end]
#             if is_valid(segment):
#                 backtrack(end, path + [segment])

#     NUM_SEGMENTS = 4
#     MAX_SEGMENT_VAL = 255
#     result = []
#     backtrack(0, [])
#     return result

# # Example usage
# print(ipDistribute("2550000"))
# print(ipDistribute("2561001234"))
# print(ipDistribute("1111"))



# assert ipDistribute("1111") == ["1.1.1.1"]
# assert ipDistribute("25514135255") == ["255.14.135.255", "255.141.35.255"]
# assert ipDistribute("202025") == ["2.0.20.25","2.0.202.5","20.2.0.25","20.20.2.5","202.0.2.5"]
# assert ipDistribute("2561001234") == []
# assert ipDistribute("0000") == ["0.0.0.0"]








'''
    97. Interleaving String

    Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

    An interleaving of two strings s and t is a configuration where s and t are divided into n and m 
    substrings respectively, such that:

    s = s1 + s2 + ... + sn, 
    t = t1 + t2 + ... + tm, 
    |n - m| <= 1,

    The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...

    Note: a + b is the concatenation of strings a and b.


    Application: DNA sequence analysis - I am building a bioinformatics program used by genetic engineers to understand genetic recombination, diagnose diseases, develop treatments, and study evolutionary biology. The feature I am currently working on will verify precise gene editing. Therefore, the function that implements this feature determines whether a given DNA sequence, denoted by "dnaSeq" in the function, was formed by an interleaving of two other sequences, denoted by "seqA" and "seqB", thereby helping engineers predict potential off-target effects and recombination events. The function is not case-sensitive and returns a boolean value based on the status of the DNA sequence passed in, i.e., it returns true if it was formed by an interleaving of sequence A and sequence B, and false otherwise. The function also checks whether the strings contain only the characters that represent the four nucleotides in DNA: Adenine (A), Cytosine (C), Guanine (G), and Thymine (T). If any of the strings do not, then an error is raised. The function also returns false in the case where "dnaSeq" is a concatenation of sequences, as opposed to an interleaving.
'''

# def validSequence(s1: str, s2: str, s3: str) -> bool:
#     if len(s1) + len(s2) != len(s3):
#         return False
    
#     if len(s1) == 0 or len(s2) ==0:
#         return False
    
#     def is_valid_dna_sequence(seq):
#         # Convert the sequence to uppercase for consistency
#         seq = seq.upper()

#         # Set of valid nucleotide characters
#         valid_nucleotides = {'A', 'C', 'G', 'T'}

#         # Check if each character in the sequence is a valid nucleotide
#         return all(char in valid_nucleotides for char in seq) and seq.isalpha()
    
#     if not (is_valid_dna_sequence(s1)) or not (is_valid_dna_sequence(s2)) or not (is_valid_dna_sequence(s3)):
#         raise TypeError

#     def dfs(i, j):
#         if i == len(s1) and j == len(s2):
#             return True
#         choose_s1, choose_s2 = False, False
#         if i < len(s1) and s1[i] == s3[i + j]:
#             choose_s1 = dfs(i + 1, j)
#         if j < len(s2) and s2[j] == s3[i + j]:
#             choose_s2 = dfs(i, j + 1)

#         return choose_s1 or choose_s2

#     return dfs(0, 0)




# # def validSequence(seqA: str, seqB: str, dnaSeq: str) -> bool: pass

# import unittest

# class TestValidSequence(unittest.TestCase):
#     def test_validSequence_true(self) -> None:
#         sequenceA = "tttga"
#         sequenceB = "gccac"
#         targetDNA = "ttgccatgca"

#         test_bool = validSequence(sequenceA, sequenceB, targetDNA)
#         self.assertTrue(test_bool)

#     def test_validSequence_false(self) -> None:
#         sequenceA = "tttga" 
#         sequenceB = "gccac"
#         targetDNA = "ttgcctcaga"

#         test_bool = validSequence(sequenceA, sequenceB, targetDNA)
#         self.assertFalse(test_bool)

#     def test_validSequence_notValid_int(self) -> None:
#         sequenceA = "tttga" 
#         sequenceB = "gccac"
#         targetDNA = "ttgccatgc3"

#         with self.assertRaises(TypeError):
#             validSequence(sequenceA, sequenceB, targetDNA)

#     def test_validSequence_notValid_char(self) -> None:
#         sequenceA = "tttga" 
#         sequenceB = "pccac"
#         targetDNA = "ttgccatgca"

#         with self.assertRaises(TypeError):
#             validSequence(sequenceA, sequenceB, targetDNA)

#     def test_validSequence_AB_empty(self) -> None:
#         sequenceA = "" 
#         sequenceB = ""
#         targetDNA = "t"

#         test_bool = validSequence(sequenceA, sequenceB, targetDNA)
#         self.assertFalse(test_bool)

#     def test_validSequence_B_empty(self) -> None:
#         sequenceA = "t" 
#         sequenceB = ""
#         targetDNA = "t"

#         test_bool = validSequence(sequenceA, sequenceB, targetDNA)
#         self.assertFalse(test_bool)

#     def test_validSequence_empty(self) -> None:
#         sequenceA = "t" 
#         sequenceB = "a"
#         targetDNA = ""

#         test_bool = validSequence(sequenceA, sequenceB, targetDNA)
#         self.assertFalse(test_bool)

#     def test_validSequence_cat(self) -> None:
#         sequenceA = "ttt" 
#         sequenceB = "aaa"
#         targetDNA = "tttaaa"

#         test_bool = validSequence(sequenceA, sequenceB, targetDNA)
#         self.assertFalse(test_bool)



# if __name__ == "__main__":
#     unittest.main()









'''
    115. Distinct Subsequences

    Given two strings s and t, return the number of distinct subsequences of s which equals t. The test cases are generated so that the answer fits on a 32-bit signed integer.

    App: text messages - I am building my own text messaging app and I have a helper function that helps suggest words whenever a misspelled word is detected. When a misspelling is detected, the larger program invokes this helper function; its output will be used in subsequent methods, which are not relevant here, to eventually pinpoint word suggestions with the highest probability. This helper takes two strings, "before" and "after", and returns the number of unique, possible substrings of "before" that are the same as "after". If any of the inputs are empty, then a value error should be raised. Note that the function is case-sensitive and subsequences are also valid as long as they are in the same relative order.
'''

# def numPossWords(s: str, t: str) -> int:
#     n=len(s)
#     m=len(t)

#     if n < 1 or m < 1:
#         raise ValueError
    
#     dp=[[0 for i in range(m+1)]for j in range(n+1)]
#     for i in range(n+1):
#         dp[i][0]=1
#     for j in range(1,m+1):
#         dp[0][j]=0
#     for i in range(1,n+1):
#         for j in range(1,m+1):
#             if s[i-1]==t[j-1]:
#                 dp[i][j]=dp[i-1][j-1]+dp[i-1][j]
#             else:
#                 dp[i][j]=dp[i-1][j]
#     return dp[n][m]



# # def numPossWords(before: str, after: str) -> int: pass

# import unittest

# class TestNumPossWords(unittest.TestCase):
#     def test_numPossWords_three(self) -> None:
#         before = "helllo"
#         after = "hello"
#         expected = 3

#         self.assertEqual(numPossWords(before, after), expected)

#     def test_numPossWords_five(self) -> None:
#         before = "sasdsad"
#         after = "sad"
#         expected = 5

#         self.assertEqual(numPossWords(before, after), expected)

#     def test_numPossWords_empty(self) -> None:
#         before = ""
#         after = ""
#         with self.assertRaises(ValueError):
#             numPossWords(before, after)

#     def test_numPossWords_notPoss1(self) -> None:
#         before = "sad"
#         after = "sasdsad"
#         expected = 0

#         self.assertEqual(numPossWords(before, after), expected)

#     def test_numPossWords_one(self) -> None:
#         before = "sad"
#         after = "sad"
#         expected = 1

#         self.assertEqual(numPossWords(before, after), expected)

#     def test_numPossWords_notPoss2(self) -> None:
#         before = "sad"
#         after = "car"
#         expected = 0

#         self.assertEqual(numPossWords(before, after), expected)

#     def test_numPossWords_caseSen(self) -> None:
#         before = "HEllo"
#         after = "HEllo"
#         expected = 1

#         self.assertEqual(numPossWords(before, after), expected)

#     def test_numPossWords_caseSen(self) -> None:
#         before = "hello"
#         after = "HEllo"
#         expected = 0

#         self.assertEqual(numPossWords(before, after), expected)

#     def test_numPossWords_subSeq(self) -> None:
#         before = "wordl"
#         after = "world"
#         expected = 0

#         self.assertEqual(numPossWords(before, after), expected)


# if __name__ == "__main__":
#     unittest.main()













'''
        FUTURE PROMPT&TEST THAT YOU CAN ALIGN WITH THE SPELL CHECKER STORY

        126. Word Latter 2
'''



                
                
                
                
        
















'''
    123. Best Time to Buy and Sell Stocks 3

    You are given an array prices where prices[i] is the price of a given stock on the ith day. Find the maximum profit you can achieve. You may complete at most two transactions. Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

    App - I am implementing a program that helps me decide when and what Pokemon cards to buy and sell while I am at the Collect-A-Con convention. Since it is always packed with people, I need to quickly calculate a sequence of trades that will maximize the profit I collect, so I implemented this logic within a function. I give the function a list that represents card buy and sell prices, which is sorted in the order that I must execute trades in, and it returns a pair (e.g., (totalProfit, [sequence of trades])) where the first element is the max profit and the second element is a list of indexes representing the cards to trade. This is a physical event, so I can only do one trade at a time, but I can also do two back-to-back, i.e., in sequence. The returned sequence list can come in one of three flavors. The first is when no trades exists and its all zeros, e.g., [0, 0, 0, 0]. The second is when a single trade yields the best outcome and the last two elements are zero, e.g., [i, j, 0, 0] where 'i' and 'j' are the indexes of the buy and sell, respectively. The third is when I need to perform a back-to-back trade, e.g., [i, j, p, q] where 'i' and 'j' are the indexes of the first trade and 'p' and 'q' are the indexes of the second trade, respectively.
'''



# def pokeTradeSequence_confirmMaxProfit(prices: list[int]) -> int:
#     """
#     :type prices: List[int]
#     :rtype: int
#     """
#     high = 0
#     low = float('inf')
#     p1 = 0
#     p2 = 0
#     short = 0
#     #1st, 2nd long
#     for i in range(len(prices)): 
#         price = prices[i]
#         if price > high:
#             high = price
#             hiIndex = i
#         if price < low or i == len(prices)-1:
#             tmp = high - low
#             if tmp > p1:
#                 p1 = tmp
#                 pIndex = [lowIndex,hiIndex]
#                 short =1
#             low = price
#             lowIndex = i
#             high = low
#             hiIndex = i
#     if short == 1:
#         fr = float('inf')
#         to = 0  
#         p2 = 0
#         for i in range(pIndex[0]):
#             price = prices[i]
#             if price > to:
#                 to = price
#             if price < fr or i == pIndex[0] -1:
#                 tmp = to - fr
#                 if tmp > p2:
#                     p2 = tmp
#                 fr = to = price 
#         fr = float('inf')
#         to = 0    
#         for i in range(pIndex[1]+1,len(prices)):
#             price = prices[i]
#             if price > to:
#                 to = price
#             if price < fr or i == len(prices)-1:
#                 tmp = to - fr
#                 if tmp > p2:
#                     p2 = tmp
#                 fr = to = price 
        
#     p3 = 0
#     fr = 0
#     to = float('inf')     #max short profit when you long it
#     if short == 1:
#         for i in range(pIndex[0]+1,pIndex[1]):
#             price = prices[i]
#             if price < to:
#                 to = price
#             if price > fr or i == pIndex[1]-1:
#                 tmp = fr - to
#                 if tmp > p3:
#                     p3 = tmp
#                 fr = to = price            
#     # compare 2 longs vs long + short
#     return max(p1,0)+ max(p2,0,p3)

# prices = [3, 3, 5, 0, 0, 3, 1, 4]
# print(pokeTradeSequence_confirmMaxProfit(prices))




# THIS NEEDS DEBUGGING
# def pokeTradeSequence(prices: list[int]) -> (int, list[int]):
#     def findBestTransaction(prices):
#         min_price = float('inf')
#         max_profit = 0
#         buy_day = sell_day = 0

#         for i, price in enumerate(prices):
#             if price < min_price:
#                 min_price = price
#                 potential_buy_day = i
#             elif price - min_price > max_profit:
#                 max_profit = price - min_price
#                 buy_day = potential_buy_day
#                 sell_day = i

#         return max_profit, buy_day, sell_day

#     # First transaction
#     first_profit, first_buy_day, first_sell_day = findBestTransaction(prices)

#     # Modify prices to remove the effect of the first transaction
#     modified_prices = prices[:]
#     for i in range(first_buy_day, first_sell_day + 1):
#         modified_prices[i] = -1  # Set to -1 or any non-influential value

#     # Second transaction
#     second_profit, second_buy_day, second_sell_day = findBestTransaction(modified_prices)

#     # Total profit and days
#     total_profit = first_profit + second_profit
#     transaction_days = [first_buy_day, first_sell_day, second_buy_day, second_sell_day]

#     return total_profit, transaction_days

# # Example usage
# prices = [12,12,14,9,9,12,10,13]
# max_profit, days = pokeTradeSequence(prices)
# print(f"Max Profit: {max_profit}, Days: {days}")  













# def pokeTradeSequence(prices: list[int]) -> (int, list[int]): pass

# import unittest

# class TestPokeTradeSequence(unittest.TestCase):
#     def test_pokeTradeSeq_firstLast(self) -> None:
#         prices = [1,2,3,4,5]
#         expected = (4, [0, 4, 0, 0])
#         actual = pokeTradeSequence(prices)
#         self.assertEqual(expected, actual)

#     def test_pokeTradeSeq_notPoss(self) -> None:
#         prices = [10,9,8,7,6]
#         expected = (0, [0, 0, 0, 0])
#         actual = pokeTradeSequence(prices)
#         self.assertEqual(expected, actual)

#     def test_pokeTradeSeq_many(self) -> None:
#         prices = [12,12,14,9,9,12,10,13]
#         expected = (6, [0, 2, 3, 7])
#         actual = pokeTradeSequence(prices)
#         self.assertEqual(expected, actual)

#     def test_pokeTradeSeq_many2(self) -> None:
#         prices = [12,11,15,14,9,12]
#         expected = (7, [1, 2, 4, 5])
#         actual = pokeTradeSequence(prices)
#         self.assertEqual(expected, actual)

#     def test_pokeTradeSeq_small(self) -> None:
#         prices = [9,8,9,7,8]
#         expected = (2, [1, 2, 3, 4])
#         actual = pokeTradeSequence(prices)
#         self.assertEqual(expected, actual)

#     def test_pokeTradeSeq_med(self) -> None:
#         prices = [6,36,18,12,42,24]
#         expected = (60, [0, 1, 3, 4])
#         actual = pokeTradeSequence(prices)
#         self.assertEqual(expected, actual)

#     def test_pokeTradeSeq_med2(self) -> None:
#         prices = [36,6,18,12,24,42]
#         expected = (42, [1, 2, 3, 5])
#         actual = pokeTradeSequence(prices)
#         self.assertEqual(expected, actual)

#     def test_pokeTradeSeq_large(self) -> None:
#         prices = [36,6,18,12,24,72,6,35,30,42]
#         expected = (102, [1, 5, 6, 9])
#         actual = pokeTradeSequence(prices)
#         self.assertEqual(expected, actual)

# if __name__ == "__main__":
#     unittest.main()














'''
    134. Gas Station

    There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i]. You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations. Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique

    I am implementing a function that simulates an electric delivery robot making deliveries via a route that maximizes its battery's charge. Every delivery location also has a charge port, but not all charge ports are the same. The function takes two lists. The first, 'power', represents the amount of power you will get from charging the robot at the ith charging port, and 'deplete' represents the route where 'deplete[i]' is the amount of power depleted when going to the next charging port. Since the robot starts off with a fully depleted battery, it must choose a starting point such that it can traverse the entire route once. The function should then return the traversable route or an empty route if it is not feasible.
    
'''





# def isTraversable(power: list[int], deplete: list[int]) -> int:
#     total_power, total_deplete, start, tank = 0, 0, 0, 0

#     for i in range(len(power)):
#         total_power += power[i]
#         total_deplete += deplete[i]
#         tank += power[i] - deplete[i]

#         # If tank is negative, reset the start position and tank
#         if tank < 0:
#             start = i + 1
#             tank = 0

#     # Check if the total power is enough to cover the total depletion
#     if total_power < total_deplete:
#         return []  # Not possible to traverse the entire route

#     # Create the route starting from the identified start position
#     route = list(range(start, len(power))) + list(range(start))
#     return route


# # def isTraversable(power: list[int], deplete: list[int]) -> list[int]: pass

# import unittest

# class TestIsTraversable(unittest.TestCase):
#     def test_isTraversable_fourth(self) -> None:
#         power = [36, 45, 27, 9, 54]
#         deplete = [45, 36, 18, 36, 27]
#         expected = [4, 0, 1, 2, 3]
#         actual = isTraversable(power, deplete)
#         self.assertEqual(expected, actual)

#     def test_isTraversable_notPossA(self) -> None:
#         power = [12, 18, 24]
#         deplete = [18, 24, 18]
#         expected = []
#         actual = isTraversable(power, deplete)
#         self.assertEqual(expected, actual)

#     def test_isTraversable_notPossB(self) -> None:
#         power = [5, 7, 8, 9, 10]
#         deplete = [8, 9, 10, 6, 7]
#         expected = []
#         actual = isTraversable(power, deplete)
#         self.assertEqual(expected, actual)

#     def test_isTraversable_third(self) -> None:
#         power = [6, 7, 8, 9, 10]
#         deplete = [8, 9, 10, 6, 7]
#         expected = [3, 4, 0, 1, 2]
#         actual = isTraversable(power, deplete)
#         self.assertEqual(expected, actual)

#     def test_isTraversable_single(self) -> None:
#         power = [6]
#         deplete = [7]
#         expected = []
#         actual = isTraversable(power, deplete)
#         self.assertEqual(expected, actual)

#     def test_isTraversable_any(self) -> None:
#         power = [7, 6, 7, 6]
#         deplete = [6, 7, 6, 7]
#         expected = [0, 1, 2, 3]
#         actual = isTraversable(power, deplete)
#         self.assertEqual(expected, actual)


# if __name__ == "__main__":
#     unittest.main()










'''
    135. Candy

    There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings. You are giving candies to these children subjected to the following requirements: 1. Each child must have at least one candy. and 2. Children with a higher rating get more candies than their neighbors. Return the minimum number of candies you need to have to distribute the candies to the children.

    App - I maintain a weekly list of employees, and everyone gets ranked by their time served in the company and their performance, which is just an integer rank. I will be giving out Lions tickets as a small weekly bonus. Therefore, I am using this function to help me calculate and minimize the number of Lions tickets I will need such that everyone on my list has at least one ticket, but better ranking employees get more tickets than lower ranking adjacent employees.
'''



# def weeklyTix(rankings: list[int]) -> int:
#     n = len(rankings)
#     if n == 0:
#         return 0

#     # Initialize tickets array
#     tickets = [1] * n

#     # Forward pass: Give more tickets if the current employee has a higher rank
#     for i in range(1, n):
#         if rankings[i] > rankings[i - 1]:
#             tickets[i] = tickets[i - 1] + 1

#     # Backward pass: Adjust tickets based on the next employee's rank and tickets
#     for i in range(n - 2, -1, -1):
#         if rankings[i] > rankings[i + 1]:
#             tickets[i] = max(tickets[i], tickets[i + 1] + 1)

#     return sum(tickets)

# # Test the function
# rankings = [0, 50, 100, 50, 0]
# print(weeklyTix(rankings))  




# # def weeklyTix(rankings: list[int]) -> int: pass

# import unittest

# class TestWeeklyTix(unittest.TestCase):
#     def test_weeklyTix1(self) -> None:
#         rankings = [8,7,9]
#         expected = 5
#         actual = weeklyTix(rankings)
#         self.assertEqual(expected, actual)

#     def test_weeklyTix2(self) -> None:
#         rankings = [19,41,77,77,62,2]
#         expected = 12
#         actual = weeklyTix(rankings)
#         self.assertEqual(expected, actual)

#     def test_weeklyTix3(self) -> None:
#         rankings = [1,1,1]
#         expected = 3
#         actual = weeklyTix(rankings)
#         self.assertEqual(expected, actual)

#     def test_weeklyTix4(self) -> None:
#         rankings = [3,6,6]
#         expected = 4
#         actual = weeklyTix(rankings)
#         self.assertEqual(expected, actual)

#     def test_weeklyTix5(self) -> None:
#         rankings = [3,5,4,4,3]
#         expected = 7
#         actual = weeklyTix(rankings)
#         self.assertEqual(expected, actual)

#     def test_weeklyTix6(self) -> None:
#         rankings = [30]
#         expected = 1
#         actual = weeklyTix(rankings)
#         self.assertEqual(expected, actual)

#     def test_weeklyTix7(self) -> None:
#         rankings = [10,9,8,7,6,5,4,3,2,1]
#         expected = 55
#         actual = weeklyTix(rankings)
#         self.assertEqual(expected, actual)

#     def test_weeklyTix8(self) -> None:
#         rankings = [0, 50, 100, 50, 0]
#         expected = 9
#         actual = weeklyTix(rankings)
#         self.assertEqual(expected, actual)


# if __name__ == "__main__":
#     unittest.main()






















'''
    137. Single Number 2

    Given an integer array nums where every element appears three times except for one, which appears exactly once. Find the single element and return it. You must implement a solution with a linear runtime complexity and use only constant extra space.

    App - I am using the following function on an embedded device that receives a continuous stream of data wirelessly. The embedded device is very low on resources and since it runs in real-time, it needs to be very efficient with respect to memory and time, so the function has to run in linear time with linear space complexity in the worst case. However, it has been reported that the transmission is picking up noise, which is altering the data stream, i.e., the device expects the stream to contain only a single, 32-bit integer, but it receives multiple. The team has consistently noticed repeated digits, specifically triplets, which do not correspond to any data on our end, so after rigorous trial and error, we deduce that the triplets represent the noise; the following function cleans up the noise and returns the valid data. If the stream passed in is empty, it simply returns None.
'''




# def cleaned(stream: list[int]) -> int:
#     if not len(stream):
#         return None
#     once = twice = 0
#     for num in stream:
#         # Appear once
#         once = ~twice & (once ^ num)
#         # Appear twice
#         twice = ~once & (twice ^ num)

#     return once 

# # Test the function
# nums = []
# print(cleaned(nums))  # Expected output: 3



# # def cleaned(stream: list[int]) -> int: pass

# import unittest

# class TestCleaned(unittest.TestCase):
#     def test_cleaned1(self) -> None:
#         stream = [1, 1, 3, 1]
#         expected = 3
#         actual = cleaned(stream)
#         self.assertEqual(expected, actual)

#     def test_cleaned2(self) -> None:
#         stream = [2, 2, 3, 2, 3, 1, 3]
#         expected = 1
#         actual = cleaned(stream)
#         self.assertEqual(expected, actual)

#     def test_cleaned3(self) -> None:
#         stream = [0,1,0,2,100,2,0,2,1,1]
#         expected = 100
#         actual = cleaned(stream)
#         self.assertEqual(expected, actual)

#     def test_cleaned4(self) -> None:
#         stream = [100,0,1,0,2,2,0,2,1,1]
#         expected = 100
#         actual = cleaned(stream)
#         self.assertEqual(expected, actual)

#     def test_cleaned5(self) -> None:
#         stream = [0]
#         expected = 0
#         actual = cleaned(stream)
#         self.assertEqual(expected, actual)

#     def test_cleaned6(self) -> None:
#         stream = [990,8010,3456,8010,25000,3456,990,8010,990,3456]
#         expected = 25000
#         actual = cleaned(stream)
#         self.assertEqual(expected, actual)

#     def test_cleaned7(self) -> None:
#         stream = [-990,8010,-3456,8010,-25000,-3456,-990,8010,-990,-3456]
#         expected = -25000
#         actual = cleaned(stream)
#         self.assertEqual(expected, actual)

#     def test_cleaned8(self) -> None:
#         stream = []
#         expected = None
#         actual = cleaned(stream)
#         self.assertEqual(expected, actual)


# if __name__ == "__main__":
#     unittest.main()
        


















'''
    140. Word Break 2

    Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order. Note that the same word in the dictionary may be reused multiple times in the segmentation.

    App - I am using the following function in a program I am building that helps people with their grammar. This function helps make suggestions when a misspelling or a grammar mistake has been detected, i.e., it suggests sentences that the user might have meant. The larger program detects the misspelling or grammar mistake and then finds the words that are calculated to have the highest probability, which are then sent to this function. The function takes the invalid word or sentence and a dictionary of words that are what the user probably meant. It then creates and returns a list of suggestion sentences that should have cleaned up the grammar mistake. Note that the order of elements in the returned list is not guaranteed, nor does it matter.
'''



# def buildSentence(invalid: str, words: list[str]) -> list[str]:
#     # Return an empty list if 'invalid' is empty or contains only whitespace
#     if not invalid.strip():
#         return []
    
#     def backtrack(start, path):
#         if start == len(invalid):
#             sentences.append(" ".join(path))
#             return
        
#         for end in range(start + 1, len(invalid) + 1):
#             word = invalid[start:end]
#             if word in word_set:
#                 backtrack(end, path + [word])

#     word_set = set(words)
#     sentences = []
#     backtrack(0, [])
#     return sentences

# # Example usage
# # invalid = "catsanddog"
# # words = ["cat", "cats", "and", "sand", "dog"]
# invalid = "batsanddung"
# words = ["bat", "bats", "sand", "and", "dung"]
# print(buildSentence(invalid, words))





# # def buildSentence(invalid: str, words: list[str]) -> list[str]: pass


# import unittest

# class TestBuildSentence(unittest.TestCase):
#     def test_buildSentence1(self) -> None:
#         invalid = "themangonowhere"
#         words = ["the", "them", "man", "mango", "go", "no", "where", "nowhere"]
#         expected = ['the man go no where', 'the man go nowhere', 'the mango no where', 'the mango nowhere']
#         actual = buildSentence(invalid, words)
#         self.assertCountEqual(expected, actual)

#     def test_buildSentence2(self) -> None:
#         invalid = "batslamblast"
#         words = ["bat", "bats", "slam", "lamb", "blast", "last"]
#         expected = ['bat slam blast', 'bats lamb last']
#         actual = buildSentence(invalid, words)
#         self.assertCountEqual(expected, actual)

#     def test_buildSentence3(self) -> None:
#         invalid = "helloworld"
#         words = ["world", "hello"]
#         expected = ['hello world']
#         actual = buildSentence(invalid, words)
#         self.assertCountEqual(expected, actual)

#     def test_buildSentence4(self) -> None:
#         invalid = ""
#         words = ["world", "hello"]
#         expected = []
#         actual = buildSentence(invalid, words)
#         self.assertCountEqual(expected, actual)

#     def test_buildSentence5(self) -> None:
#         invalid = " "
#         words = ["world", "hello"]
#         expected = []
#         actual = buildSentence(invalid, words)
#         self.assertCountEqual(expected, actual)

#     def test_buildSentence6(self) -> None:
#         invalid = "butterflyingestimate"
#         words = ["but", "butter", "fly", "butterfly", "flying", "estimate", "ingest", "im", "ate"]
#         expected = ['butter fly ingest im ate', 'butter flying estimate', 'butterfly ingest im ate']
#         actual = buildSentence(invalid, words)
#         self.assertCountEqual(expected, actual)

#     def test_buildSentence7(self) -> None:
#         invalid = "batsanddung"
#         words = ["bat", "bats", "sand", "and", "dung"]
#         expected = ['bat sand dung', 'bats and dung']
#         actual = buildSentence(invalid, words)
#         self.assertCountEqual(expected, actual)

#     def test_buildSentence8(self) -> None:
#         invalid = "batsandung"
#         words = ["bat", "bats", "sand", "and", "dung"]
#         expected = []
#         actual = buildSentence(invalid, words)
#         self.assertCountEqual(expected, actual)

# if __name__ == "__main__":
#     unittest.main()


























'''
    2312. Selling pieces of wood

    You are given two integers m and n that represent the height and width of a rectangular piece of wood. You are also given a 2D integer array prices, where prices[i] = [h_i, w_i, price_i] indicates you can sell a rectangular piece of wood of height h_i and width w_i for price_i dollars. To cut a piece of wood, you must make a vertical or horizontal cut across the entire height or width of the piece to split it into two smaller pieces. After cutting a piece of wood into some number of smaller pieces, you can sell pieces according to prices. You may sell multiple pieces of the same shape, and you do not have to sell all the shapes. The grain of the wood makes a difference, so you cannot rotate a piece to swap its height and width. Return the maximum money you can earn after cutting an m x n piece of wood. Note that you can cut the piece of wood as many times as you want.

    App - I am selling large plots of land on behalf of my customers and I use the following function to calculate the highest sale price after cross-referencing the current market prices. I can sell the land quicker and possibly for more if I break the land up because the market is willing to pay more for a specific size plot of land, so I give the function these market values and let it tell me what's the most I can sell the entire plot for. However, when we partition the land, we must do it such that its split into two constituent pieces that either have the same length or the same height. This way, no matter how the original plot was partitioned, we are guaranteed to always have pieces with four sides. The market prices are formatted in the following way: a list of lists where each sub-list is a market valuation triple, e.g., the land's dimensions followed by the sale price, so if the market is willing to pay 100k for a m x n plot, then one element in the list 'marketValues' will be [m, n, 100]. The function also needs to know how much land I am currently selling, so it also takes the dimensions of my current plot denoted myPlot_m and myPlot_n. If the function gets passed any market values with non-positive integers or where the dimensions are larger than whats being sold, then a value error should be raised.
'''


# def salePrice(myPlot_m: int, myPlot_n: int, marketValues: list[list[int]]) -> int:

#     if not marketValues or not marketValues[0]:
#         return 0

#     # Create a table for all sizes of land
#     dp = [[0] * (myPlot_n + 1) for _ in range(myPlot_m + 1)]

#     # Seed with market prices
#     for length, width, price in marketValues:
#         if (0 < length <= myPlot_m) and (0 < width <= myPlot_n) and (price > 0):
#             dp[length][width] = price
#         else:
#             raise ValueError(f"The [{length}, {width}, {price}] market valuation is not compatible")
    
#     # Traverse all cells
#     for length in range(myPlot_m + 1):
#         for width in range(myPlot_n + 1):
            
#             # Attempt to slice about the x
#             for partition in range(length // 2 + 1):
#                 dp[length][width] = max(dp[length][width], dp[partition][width] + dp[length - partition][width])

#             # Attempt to slice about the y
#             for partition in range(width // 2 + 1):
#                 dp[length][width] = max(dp[length][width], dp[length][partition] + dp[length][width - partition])

#     # The best sale price
#     return dp[myPlot_m][myPlot_n]


# m = 20
# n = 10
# marketValues = [[]]
# print(salePrice(m, n, marketValues))





# # def salePrice(myPlot_m: int, myPlot_n: int, marketValues: list[list[int]]) -> int: pass

# import unittest

# class TestSalePrice(unittest.TestCase):
#     def test_salePrice1(self) -> None:
#         m = 9
#         n = 15
#         marketValues = [[3,12,6],[6,6,21],[6,3,9]]
#         expected = 57
#         actual = salePrice(m,n,marketValues)
#         self.assertEqual(expected, actual)

#     def test_salePrice2(self) -> None:
#         m = 20
#         n = 30
#         marketValues = [[3,12,67],[6,16,215],[6,13,90],[20,20,3226]]
#         expected = 3226
#         actual = salePrice(m,n,marketValues)
#         self.assertEqual(expected, actual)

#     def test_salePrice3(self) -> None:
#         m = 20
#         n = 10
#         marketValues = [[3,12,67],[6,6,215],[6,3,90]]
#         with self.assertRaises(ValueError):
#             salePrice(m,n,marketValues)

#     def test_salePrice4(self) -> None:
#         m = 20
#         n = 10
#         marketValues = [[6,6,215],[6,3,90],[17,8,916]]
#         expected = 916
#         actual = salePrice(m,n,marketValues)
#         self.assertEqual(expected, actual)

#     def test_salePrice5(self) -> None:
#         m = 20
#         n = 10
#         marketValues = [[3,12,67],[6,6,215],[6,3,-90]]
#         with self.assertRaises(ValueError):
#             salePrice(m,n,marketValues)

#     def test_salePrice6(self) -> None:
#         m = 20
#         n = 10
#         marketValues = [[3,12,67],[6,6,215],[-6,3,90]]
#         with self.assertRaises(ValueError):
#             salePrice(m,n,marketValues)

#     def test_salePrice7(self) -> None:
#         m = 20
#         n = 10
#         marketValues = [[1,1,215],[6,3,90],[20,10,43001],[19,10,43002]]
#         expected = 45152
#         actual = salePrice(m,n,marketValues)
#         self.assertEqual(expected, actual)

#     def test_salePrice8(self) -> None:
#         m = 20
#         n = 10
#         marketValues = [[]]
#         expected = 0
#         actual = salePrice(m,n,marketValues)
#         self.assertEqual(expected, actual)


# if __name__ == "__main__":
#     unittest.main()

















'''
    2316. Count unreachable pairs of nodes in an undirected graph

    You are given an integer n. There is an undirected graph with n nodes, numbered from 0 to n - 1. You are given a 2D integer array edges where edges[i] = [ai, bi] denotes that there exists an undirected edge connecting nodes ai and bi. Return the number of pairs of different nodes that are unreachable from each other.

    App - I help maintain the communal solar farm in my neighborhood and our solar grid has grown to provide energy accross multiple blocks. Every house on the grid is connected so that we can share batteries, which means that new comers do not have to spend a ton right away, all they need is some panels installed to get connected. I am using the following function whenever the grid experiences outage to find out which houses are still connected and which houses are not. The function takes a 2D list where every inner list is two integers in length that represent a pair of connected houses. It also takes the total number of houses on our grid. With that information, this function gives me all disconnected houses, so that I can begin investigating and repairing connections. If the list contains any integers that are out of range, a value error is raised.
'''




def disconnected(numHomes: int, connections: list[list[int]]) -> list[list[int]]:
    # Validate the list
    invalidList = any(element > numHomes or element < 0 for pair in connections for element in pair)
    if invalidList:
        raise ValueError("There are invalid connections!")

    # Create a set of all possible connections
    all_connections = {(min(i, j), max(i, j)) for i in range(numHomes) for j in range(i + 1, numHomes)}

    # Remove the existing connections
    existing_connections = {tuple(sorted(connection)) for connection in connections}
    disconnected_pairs = all_connections - existing_connections

    # Convert the set of tuples back to a list of lists
    disconnected = [list(pair) for pair in disconnected_pairs]

    return disconnected

# Example usage
# numHomes = 4
# connections = [[1,0],[6,2],[0,3]]
# print(disconnected(numHomes, connections))  # Output will be the pairs of homes that are disconnected




# def disconnected(numHomes: int, connections: list[list[int]]) -> list[list[int]]: pass

import unittest

class TestDisconnected(unittest.TestCase):
    def test_disconnected1(self) -> None:
        numHomes = 4
        connections = [[1,2],[2,3]]
        expected = [[0,1],[0,2],[0,3],[1,3]]
        actual = disconnected(numHomes, connections)
        self.assertCountEqual(expected, actual)

    def test_disconnected2(self) -> None:
        numHomes = 7
        connections = [[0,2],[0,5],[2,4],[1,6],[5,4]]
        expected = [[0,1],[0,3],[0,6],[1,2],[1,3],[1,4],[1,5],[2,3],[2,6],[3,4],[3,5],[3,6],[4,6],[5,6],[0,4],[2,5]]
        actual = disconnected(numHomes, connections)
        self.assertCountEqual(expected, actual)

    def test_disconnected3(self) -> None:
        numHomes = 4
        connections = []
        expected = [[0,1],[2,3],[0,2],[1,2],[0,3],[1,3]]
        actual = disconnected(numHomes, connections)
        self.assertCountEqual(expected, actual)

    def test_disconnected4(self) -> None:
        numHomes = 4
        connections = [[0,1],[2,3],[0,2],[1,2],[0,3],[1,3]]
        expected = []
        actual = disconnected(numHomes, connections)
        self.assertCountEqual(expected, actual)

    def test_disconnected5(self) -> None:
        numHomes = 4
        connections = [[0,1],[2,-3],[1,2],[3,0]]
        with self.assertRaises(ValueError):
            disconnected(numHomes, connections)

    def test_disconnected6(self) -> None:
        numHomes = 4
        connections = [[0,1],[1,0],[0,1]]
        expected = [[2,3],[0,2],[1,2],[0,3],[1,3]]
        actual = disconnected(numHomes, connections)
        self.assertCountEqual(expected, actual)

    def test_disconnected7(self) -> None:
        numHomes = 4
        connections = [[1,0],[6,2],[0,3]]
        with self.assertRaises(ValueError):
            disconnected(numHomes, connections)

    def test_disconnected8(self) -> None:
        numHomes = 3
        connections = [[2,1],[2,0],[1,0]]
        expected = []
        actual = disconnected(numHomes, connections)
        self.assertCountEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

























