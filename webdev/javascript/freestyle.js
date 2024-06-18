

// var checkPalindromeFormation = function(a, b) 
// {
//     // Function to check if a given string is a palindrome
//     const isPalindrome = (str, i, j) => 
//     {
//         while (i < j) 
//         {
//             if (str[i] !== str[j]) return false;
//             i++; j--;
//         }
//         return true;
//     };

//     // Function to check if valid palindrome can be formed by concatenating prefixes and suffixes
//     const check = (a, b) => 
//     {
//         let i = 0, j = a.length - 1;
//         while (i < j && a[i] === b[j]) 
//         {
//             i++; j--;
//         }

//         // Check if the remaining part of a or b is a palindrome
//         return isPalindrome(a, i, j) || isPalindrome(b, i, j);
//     };

//     // Check both combinations: a_prefix + b_suffix and b_prefix + a_suffix
//     return check(a, b) || check(b, a);
// };


// var longestValidParentheses = function(s) {
//     let maxLen = 0;
//     let stack = [-1];  // Initialize stack with -1 to handle edge cases

//     for (let i = 0; i < s.length; i++) {
//         if (s[i] === '(') {
//             // Push the index of '(' onto the stack
//             stack.push(i);
//         } else {
//             // Pop the top element from the stack
//             stack.pop();
//             if (stack.length === 0) {
//                 // If stack is empty, push the current index onto the stack
//                 stack.push(i);
//             } else {
//                 // Calculate the length of the current valid substring
//                 maxLen = Math.max(maxLen, i - stack[stack.length - 1]);
//             }
//         }
//     }

//     return maxLen;
// };




// gpt



















// A






















// B










