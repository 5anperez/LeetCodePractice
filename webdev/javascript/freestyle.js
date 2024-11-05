







const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

window.addEventListener('resize', function () {
  resetCanvas();
});

function resetCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  drawHexagonGrid();
}   
    
const radius = 50;
const numberOfSides = 6;
const angle = (2 * Math.PI) / numberOfSides;
const columnWidth = radius + (radius * Math.cos(angle));
const rowHeight = 2 * radius * Math.sin(angle);

function drawHexagon(x, y) {
  ctx.beginPath();
  ctx.moveTo(x + radius * Math.cos(0), y + radius * Math.sin(0));
  for (let i = 1; i <= numberOfSides; i++) {
    ctx.lineTo(x + radius * Math.cos(angle * i), y + radius * Math.sin(angle * i));
  }
  ctx.closePath();
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawRedDot(x, y) {  
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, 2 * Math.PI, true);
  ctx.fillStyle = 'red';
  ctx.fill();
}


// OG
// function drawHexagonGrid() {
//   ctx.clearRect(0, 0, canvas.width, canvas.height);
//   const precision = 4;
//   const drawnVertices = new Set();
//   const horizontalHexagons = Math.ceil(canvas.width / columnWidth) + 1;
//   const verticalHexagons = Math.ceil(canvas.height / rowHeight);  

//   for (let i = 0; i < horizontalHexagons; i++) 
//   {
//     const x = i * columnWidth;
//     const yOffset = i % 2 === 0 ? 0 : rowHeight / 2;    
  
//     for (let j = 0; j < verticalHexagons; j++) 
//     {
//       const y = yOffset + j * rowHeight;
//       drawHexagon(x, y); 
    
//       for (let k = 0; k < numberOfSides; k++) 
//       {
//         const dotX = roundToDecimalPlace(x + radius * Math.cos(angle * k), precision);
//         const dotY = roundToDecimalPlace(y + radius * Math.sin(angle * k), precision);
      
//         const vertexKey = `${dotX},${dotY}`;
//         if (!drawnVertices.has(vertexKey)) {
//           drawnVertices.add(vertexKey);          
//           drawRedDot(dotX, dotY);
//         }
//       }
//     }
//   }
// }




// ma
function drawHexagonGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const precision = 4;
  const drawnVertices = new Set();
  const verticesToDraw = []; // New array to store vertices coordinates
  const horizontalHexagons = Math.ceil(canvas.width / columnWidth) + 1;
  const verticalHexagons = Math.ceil(canvas.height / rowHeight);  

  for (let i = 0; i < horizontalHexagons; i++) 
  {
    const x = i * columnWidth;
    const yOffset = i % 2 === 0 ? 0 : rowHeight / 2;    
  
    for (let j = 0; j < verticalHexagons; j++) 
    {
      const y = yOffset + j * rowHeight;
      drawHexagon(x, y); 
    
      for (let k = 0; k < numberOfSides; k++) 
      {
        const dotX = roundToDecimalPlace(x + radius * Math.cos(angle * k), precision);
        const dotY = roundToDecimalPlace(y + radius * Math.sin(angle * k), precision);
      
        const vertexKey = `${dotX},${dotY}`;
        if (!drawnVertices.has(vertexKey)) {
          drawnVertices.add(vertexKey);          
          verticesToDraw.push({x: dotX, y: dotY}); // Store vertex instead of drawing
        }
      }
    }
  }

  // Draw vertices after all hexagons are drawn
  verticesToDraw.forEach(vertex => {
    drawRedDot(vertex.x, vertex.y);
  });
}





// mb
function drawHexagonGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const precision = 4;
  const drawnVertices = new Set();
  const horizontalHexagons = Math.ceil(canvas.width / columnWidth) + 1;
  const verticalHexagons = Math.ceil(canvas.height / rowHeight);  
  
  // First: Draw all hexagons and collect vertices
  for (let i = 0; i < horizontalHexagons; i++) 
  {
    const x = i * columnWidth;
    const yOffset = i % 2 === 0 ? 0 : rowHeight / 2;    
  
    for (let j = 0; j < verticalHexagons; j++) 
    {
      const y = yOffset + j * rowHeight;
      drawHexagon(x, y); 
    
      for (let k = 0; k < numberOfSides; k++) 
      {
        const dotX = roundToDecimalPlace(x + radius * Math.cos(angle * k), precision);
        const dotY = roundToDecimalPlace(y + radius * Math.sin(angle * k), precision);
      
        const vertexKey = `${dotX},${dotY}`;
        // Collect vertices without drawing yet
        if (!drawnVertices.has(vertexKey)) {
          drawnVertices.add(vertexKey);          
        }
      }
    }
  }

  // Second: Draw all collected vertices (red dots) on top of hexagons
  drawnVertices.forEach(vertexKey => {
    const [dotX, dotY] = vertexKey.split(',').map(Number);
    drawRedDot(dotX, dotY);
  });
}



function roundToDecimalPlace(number, decimalPlaces) {
    const factor = Math.pow(10, decimalPlaces);
    return Math.round(number * factor) / factor;
}
    
resetCanvas();

























































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










// class Dictionary 
// {
//     constructor(dictionaryString) 
//     {

//         this.items = {};

//         // Trim whitespace from the input string
//         const cleanString = dictionaryString.trim();

//         // Check if the string is empty or only contains "{}"
//         if (cleanString === '' || cleanString === '{}' || cleanString === '{ }') 
//         {
//             // If the string is empty or contains only "{}", initialize with an empty dictionary
//             return; // Exit the constructor early
//         }

//         // Remove leading and trailing curly braces if present
//         const content = cleanString.startsWith('{') && cleanString.endsWith('}')
//             ? cleanString.slice(1, -1)
//             : cleanString;

//         // Split the cleaned string by commas to separate key-value pairs
//         const pairs = content.split(',');

//         // Process each key-value pair
//         pairs.forEach(pair => 
//         {
//             // Split each pair into key and value
//             const [rawKey, rawValue] = pair.trim().split(':').map(part => part.trim());

//             // Remove quotes from key
//             const cleanedKey = rawKey.slice(1, -1);
  
//             // Determine the cleaned value
//             let cleanedValue;
//             if (rawValue.startsWith("'") && rawValue.endsWith("'")) 
//             {
//                 // If value is a string in single quotes
//                 cleanedValue = rawValue.slice(1, -1);
//             } 
//             else 
//             {
//                 // If value is a number or boolean
//                 cleanedValue = Number(rawValue) || (rawValue === "true" ? true : false);
//             }

//             // Set the key-value pair in the dictionary
//             this.items[cleanedKey] = cleanedValue;
//         });
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
//       console.log("Removed caleed key: " + key + " " + JSON.stringify(this.items));
//       if (this.has(key)) {
//         delete this.items[key];
//         console.log("Removed caleed key: " + key + " " + JSON.stringify(this.items));
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
//         requestedVMs.remove(key); // should be this.remove(key)
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

//     removeStaleRequests() {
//       const tenMinutesAgo = Date.now() - (10 * 60 * 1000); // Calculate 10 minutes ago timestamp
//       const staleItems = []; // Array to hold stale items
  
//       // Iterate over keys to check and collect stale requests
//       for (const key in this.items) {
//         if (this.items[key] && JSON.parse(this.items[key]).requestTime < tenMinutesAgo) {
//           // Push the stale item into the array
//           let item = {}
//           item[key] = JSON.parse(requestedVMs.get(key));
//           staleItems.push(JSON.stringify(item));
//           // Remove the stale item from the dictionary
//           requestedVMs.remove(key)
//         }
//       }
//       writeSystemVar("requestedVMs", JSON.stringify(requestedVMs.getDict()));
  
//       if (staleItems.length > 0) {
//         let errorMessage = `The following requests were not processed within the required time:\n`;
//         errorMessage += JSON.stringify(staleItems);
//         errorMessage += `\n Current queue size: ${queue.size()}\n`;
//         postToTelegram(errorMessage);
//         console.error(errorMessage); // Output error message to console (or handle it as needed)
//       }
//     }
//   }










































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












































