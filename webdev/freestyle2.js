var checkPalindromeFormation = function(a, b) {
    if (a.length !== b.length) {
        return false;
    }
    
    for (let i = 0; i < a.length; i++) {
        const a_prefix = a.slice(0, i);
        const a_suffix = a.slice(i);
        const b_prefix = b.slice(0, i);
        const b_suffix = b.slice(i);

        // console.log("This is what I am comparing: ")
        // console.log(a_prefix + b_suffix)
        // console.log("and ")
        // console.log(b_suffix.split('').reverse().join('') + a_prefix)

        let possPal1 = a_prefix + b_suffix;
        
        if (possPal1 === possPal1.split('').reverse().join(''))
            return true;


        // console.log("Then, I compare: ")
        // console.log(b_prefix + a_suffix)
        // console.log("and ")
        // console.log(a_suffix.split('').reverse().join('') + b_prefix)

        let possPal2 = b_prefix + a_suffix;
        
        if (possPal2 === possPal2.split('').reverse().join(''))
            return true;

        // if (a_prefix + b_suffix === b_suffix.split('').reverse().join('') + a_prefix ||
        //     b_prefix + a_suffix === a_suffix.split('').reverse().join('') + b_prefix) {
        //     return true;
        // }
    }

    return false;
};


var checkPalindromeFormation = function(a, b) {
    if (a.length !== b.length) {
      return false; // Invalid input, different lengths
    }
  
    for (let splitIndex = 0; splitIndex <= a.length; splitIndex++) {
      const aPrefix = a.slice(0, splitIndex);
      const aSuffix = a.slice(splitIndex);
      const bPrefix = b.slice(0, splitIndex);
      const bSuffix = b.slice(splitIndex);
  
      // Check both possible combinations
      if (isPalindrome(aPrefix + bSuffix) || isPalindrome(bPrefix + aSuffix)) {
        return true;
      }
    }
  
    return false; // No combination worked
  };
  
  // Improved version of isPalindrome for efficiency
  function isPalindrome(str) {
    for (let i = 0, j = str.length - 1; i < j; i++, j--) {
      if (str[i] !== str[j]) {
        return false;
      }
    }
    return true;
  }



