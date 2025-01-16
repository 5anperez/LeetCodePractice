

/*

Prompt:

Create a simple JavaScript example that shows how to use arrow functions

*/


// gpt


// Example 1: Basic Arrow Function
const greet = (name) => {
  return `Hello, ${name}!`;
};
console.log(greet("Alice")); // Output: Hello, Alice!



// Example 2: Arrow Function with Implicit Return
const square = (num) => num * num; // No need for 'return' or curly braces
console.log(square(4)); // Output: 16

// Example 3: Arrow Function with No Parameters
const sayHello = () => "Hello, World!";
console.log(sayHello()); // Output: Hello, World!



// Example 4: Arrow Function with One Parameter (Parentheses Optional)
const double = x => x * 2; // Parentheses around 'x' are optional
console.log(double(5)); // Output: 10



// Example 5: Arrow Function in Array Mapping
const numbers = [1, 2, 3, 4, 5];
const doubledNumbers = numbers.map(num => num * 2);
console.log(doubledNumbers); // Output: [2, 4, 6, 8, 10]




// Example 6: Arrow Function as a Callback
const isEven = num => num % 2 === 0;
const evens = numbers.filter(isEven); // Filter even numbers
console.log(evens); // Output: [2, 4]

// Example 7: Arrow Functions and 'this' Binding
class Counter {
  constructor() {
      this.count = 0;
  }

  increment() {
      setTimeout(() => {
          this.count++;
          console.log(this.count); // Arrow function uses 'this' from Counter
      }, 1000);
  }
}

const counter = new Counter();
counter.increment(); // Output: 1 (after 1 second)


























// ma
// Define an arrow function to greet someone
// const greet = (name) => `Hello, ${name}!`;

// // Call the arrow function
// console.log(greet('Alice'));  // Output: Hello, Alice!



// // Define an arrow function to add two numbers
// const add = (a, b) => a + b;

// // Call the arrow function
// console.log(add(5, 3));  // Output: 8



// // Define an arrow function that returns a message
// const sayHello = () => 'Hello!';

// // Call the arrow function
// console.log(sayHello());  // Output: Hello!



// // Define an object with an arrow function as a property
// const person = {
//   name: 'Bob',
//   greet: () => `Hello, my name is ${this.name}!`
// };

// // Call the arrow function on the object
// console.log(person.greet());  // Output: Hello, my name is Bob!

// // Define an arrow function that returns the square of a number
// const square = num => num * num;

// // Call the arrow function
// console.log(square(4));  // Output: 16




















// mb 
// Traditional function
function greet(name) {
  return `Hello, ${name}!`;
}

// Arrow function
const greetArrow = name => `Hello, ${name}!`;

// Using the traditional function
console.log(greet('Alice'));  // Output: Hello, Alice!

// Using the arrow function
console.log(greetArrow('Bob'));  // Output: Hello, Bob!

// Arrow functions with a single argument can omit the parentheses
const addOne = x => x + 1;
console.log(addOne(5));  // Output: 6

// Arrow functions with multiple arguments require parentheses
const multiply = (x, y) => x * y;
console.log(multiply(3, 4));  // Output: 12

// Arrow functions can be used as callback functions
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2);
console.log(doubled);  // Output: [2, 4, 6, 8, 10]
























































