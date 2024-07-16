

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










class Dictionary 
{
    constructor(dictionaryString) 
    {

        this.items = {};

        // Trim whitespace from the input string
        const cleanString = dictionaryString.trim();

        // Check if the string is empty or only contains "{}"
        if (cleanString === '' || cleanString === '{}' || cleanString === '{ }') 
        {
            // If the string is empty or contains only "{}", initialize with an empty dictionary
            return; // Exit the constructor early
        }

        // Remove leading and trailing curly braces if present
        const content = cleanString.startsWith('{') && cleanString.endsWith('}')
            ? cleanString.slice(1, -1)
            : cleanString;

        // Split the cleaned string by commas to separate key-value pairs
        const pairs = content.split(',');

        // Process each key-value pair
        pairs.forEach(pair => 
        {
            // Split each pair into key and value
            const [rawKey, rawValue] = pair.trim().split(':').map(part => part.trim());

            // Remove quotes from key
            const cleanedKey = rawKey.slice(1, -1);
  
            // Determine the cleaned value
            let cleanedValue;
            if (rawValue.startsWith("'") && rawValue.endsWith("'")) 
            {
                // If value is a string in single quotes
                cleanedValue = rawValue.slice(1, -1);
            } 
            else 
            {
                // If value is a number or boolean
                cleanedValue = Number(rawValue) || (rawValue === "true" ? true : false);
            }

            // Set the key-value pair in the dictionary
            this.items[cleanedKey] = cleanedValue;
        });
    }
  
    // Method to check if key exists in the dictionary
    has(key) {
      return key in this.items;
    }
  
    // Method to add or update a key-value pair in the dictionary
    set(key, value) {
      this.items[key] = value;
    }
  
    // Method to get the value associated with a key
    get(key) {
      return this.has(key) ? this.items[key] : undefined;
    }
  
    // Method to remove a key and its associated value from the dictionary
    remove(key) {
      console.log("Removed caleed key: " + key + " " + JSON.stringify(this.items));
      if (this.has(key)) {
        delete this.items[key];
        console.log("Removed caleed key: " + key + " " + JSON.stringify(this.items));
        return true;
      }
      return false;
    }
  
    // Method to get all keys in the dictionary
    keys() {
      return Object.keys(this.items);
    }
  
    // Method to get all values in the dictionary
    values() {
      return Object.values(this.items);
    }
  
    // Method to get all key-value pairs in the dictionary
    entries() {
      return Object.entries(this.items);
    }
  
    // Method to get the size (number of key-value pairs) of the dictionary
    size() {
      return Object.keys(this.items).length;
    }
  
    // Method to clear all key-value pairs from the dictionary
    clear() {
      this.items = {};
    }
  
    // Method to get the current dictionary as a JSON object
    getDict() {
      return this.items;
    }
  
    // Method to remove items that start with a matched string
    removeMatching(startsWith) {
      const keysToRemove = [];
      for (const key in this.items) {
        if (key.startsWith(startsWith)) {
          keysToRemove.push(key);
        }
      }
      keysToRemove.forEach(key => {
        requestedVMs.remove(key); // should be this.remove(key)
      });
    }
  
    // Method to return keys that start with a similar string
    getKeysStartingWith(startsWith) {
      const matchingKeys = [];
      for (const key in this.items) {
        if (key.startsWith(startsWith)) {
          matchingKeys.push(key);
        }
      }
      return matchingKeys;
    }
  
    // Method to remove items by their values
    removeByValue(targetValue) {
      const keysToRemove = [];
  
      // Iterate through the dictionary items
      for (const key in this.items) {
        if (this.items[key] === targetValue) {
          keysToRemove.push(key);
        }
      }
  
      // Remove keys found with the target value
      keysToRemove.forEach(key => {
        delete this.items[key];
      });
    }

    removeStaleRequests() {
      const tenMinutesAgo = Date.now() - (10 * 60 * 1000); // Calculate 10 minutes ago timestamp
      const staleItems = []; // Array to hold stale items
  
      // Iterate over keys to check and collect stale requests
      for (const key in this.items) {
        if (this.items[key] && JSON.parse(this.items[key]).requestTime < tenMinutesAgo) {
          // Push the stale item into the array
          let item = {}
          item[key] = JSON.parse(requestedVMs.get(key));
          staleItems.push(JSON.stringify(item));
          // Remove the stale item from the dictionary
          requestedVMs.remove(key)
        }
      }
      writeSystemVar("requestedVMs", JSON.stringify(requestedVMs.getDict()));
  
      if (staleItems.length > 0) {
        let errorMessage = `The following requests were not processed within the required time:\n`;
        errorMessage += JSON.stringify(staleItems);
        errorMessage += `\n Current queue size: ${queue.size()}\n`;
        postToTelegram(errorMessage);
        console.error(errorMessage); // Output error message to console (or handle it as needed)
      }
    }
  }










































//   class Dictionary 
//   {
//     constructor(dictionaryString) 
//     {

//       this.items = {};
  
//       // Trim whitespace from the input string
//       const cleanString = dictionaryString.trim();
  
//       // Check if the string is empty or only contains "{}"
//       if (cleanString === '' || cleanString === '{}' || cleanString === '{ }') 
//         {
//         // If the string is empty or contains only "{}", initialize with an empty dictionary
//         return; // Exit the constructor early
//       }
  
//       // Remove leading and trailing curly braces if present
//       const content = cleanString.startsWith('{') && cleanString.endsWith('}')
//         ? cleanString.slice(1, -1)
//         : cleanString;
  
//       // Split the cleaned string by commas to separate key-value pairs
//       const pairs = content.split(',');
  
//       // Process each key-value pair
//       pairs.forEach(pair => 
//         {
//         // Split each pair into key and value
//         const [rawKey, rawValue] = pair.trim().split(':').map(part => part.trim());
  
//         // Remove quotes from key
//         const cleanedKey = rawKey.slice(1, -1);
  
//         // Determine the cleaned value
//         let cleanedValue;
//         if (rawValue.startsWith("'") && rawValue.endsWith("'")) {
//           // If value is a string in single quotes
//           cleanedValue = rawValue.slice(1, -1);
//         } else {
//           // If value is a number or boolean
//           cleanedValue = Number(rawValue) || (rawValue === "true" ? true : false);
//         }
  
//         // Set the key-value pair in the dictionary
//         this.items[cleanedKey] = cleanedValue;
//       });
//     }
  
//     // Method to check if key exists in the dictionary
//     has(key) {
//       return key in this.items;
//     }
  
//     // Method to add or update a key-value pair in the dictionary
//     set(key, value) {
//       this.items[key] = value;
//     }
  
//     // Method to get the value associated with a key
//     get(key) {
//       return this.has(key) ? this.items[key] : undefined;
//     }
  
//     // Method to remove a key and its associated value from the dictionary
//     remove(key) {
//       if (this.has(key)) {
//         delete this.items[key];
//         return true;
//       }
//       return false;
//     }
  
//     // Method to get all keys in the dictionary
//     keys() {
//       return Object.keys(this.items);
//     }
  
//     // Method to get all values in the dictionary
//     values() {
//       return Object.values(this.items);
//     }
  
//     // Method to get all key-value pairs in the dictionary
//     entries() {
//       return Object.entries(this.items);
//     }
  
//     // Method to get the size (number of key-value pairs) of the dictionary
//     size() {
//       return Object.keys(this.items).length;
//     }
  
//     // Method to clear all key-value pairs from the dictionary
//     clear() {
//       this.items = {};
//     }
  
//     // Method to get the current dictionary as a JSON object
//     getDict() {
//       return this.items;
//     }
  
//     // Method to remove items that start with a matched string
//     removeMatching(startsWith) {
//       const keysToRemove = [];
//       for (const key in this.items) {
//         if (key.startsWith(startsWith)) {
//           keysToRemove.push(key);
//         }
//       }
//       keysToRemove.forEach(key => {
//         this.remove(key);
//       });
//     }
  
//     // Method to return keys that start with a similar string
//     getKeysStartingWith(startsWith) {
//       const matchingKeys = [];
//       for (const key in this.items) {
//         if (key.startsWith(startsWith)) {
//           matchingKeys.push(key);
//         }
//       }
//       return matchingKeys;
//     }
  
//     // Method to remove items by their values
//     removeByValue(targetValue) {
//       const keysToRemove = [];
  
//       // Iterate through the dictionary items
//       for (const key in this.items) {
//         if (this.items[key] === targetValue) {
//           keysToRemove.push(key);
//         }
//       }
  
//       // Remove keys found with the target value
//       keysToRemove.forEach(key => {
//         delete this.items[key];
//       });
//     }
  
//     // Method to remove stale requests
//     removeStaleRequests() {
//       const tenMinutesAgo = Date.now() - (10 * 60 * 1000); // Calculate 10 minutes ago timestamp
//       const staleItems = []; // Array to hold stale items
  
//       // Iterate over keys to check and collect stale requests
//       for (const key in this.items) {
//         try {
//           const item = JSON.parse(this.get(key)); // Parse item safely
//           if (item.requestTime < tenMinutesAgo) {
//             staleItems.push({[key]: item});
//             this.remove(key); // Remove the stale item from the dictionary
//           }
//         } catch (e) {
//           console.error(`Failed to parse JSON for key ${key}: `, e);
//         }
//       }
  
//       writeSystemVar("requestedVMs", JSON.stringify(this.getDict()));
  
//       if (staleItems.length > 0) {
//         let errorMessage = `The following requests were not processed within the required time:\n`;
//         errorMessage += JSON.stringify(staleItems);
//         errorMessage += `\n Current queue size: ${queue.size()}\n`;
//         postToTelegram(errorMessage);
//         console.error(errorMessage); // Output error message to console (or handle it as needed)
//       }
//     }
//   }
































































//   class Dictionary {
//     constructor(dictionaryString) {
//       this.items = {};
  
//       // ...
  
//       this.validateInput(dictionaryString);
//     }
  
//     validateInput(dictionaryString) {
//       if (typeof dictionaryString !== 'string') {
//         throw new Error('Input must be a string');
//       }
//     }
  
//     // ...
  
//     remove(key) {
//       try {
//         delete this.items[key];
//         return true;
//       } catch (error) {
//         console.error(`Error removing key ${key}: ${error}`);
//         return false;
//       }
//     }
  
//     // ...
  
//     removeStaleRequests() {
//       const staleItems = [];
//       const tenMinutesAgo = Date.now() - Dictionary.STALE_REQUEST_THRESHOLD;
  
//       Object.keys(this.items).forEach((key) => {
//         const item = this.items[key];
//         if (item.requestTime < tenMinutesAgo) {
//           staleItems.push(item);
//           delete this.items[key];
//         }
//       });
  
//       if (staleItems.length > 0) {
//         // ...
//       }
//     }
//   }
  
//   Dictionary.STALE_REQUEST_THRESHOLD = 10 * 60 * 1000; // 10 minutes












































