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




import itertools

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
    91. Decode Ways

    Can you write a function called "def possibileGroups(self, s: str) -> int:" that takes a string 's' and calculates the number of possible partitions, i.e., the function partitions 's' evenly into groups of integers in the range [1, 26] (inclusive), where each partition or group is made from the elements in 's', and returns the number of possible ways to partition 's'.
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

    Write me a function that
'''


































