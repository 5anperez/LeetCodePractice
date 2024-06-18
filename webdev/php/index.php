<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>


<body>

    <?php
    function isPy($c) {
        // Check if the input is a non-negative integer
        if (!is_int($c) || $c < 0) {
            return false;
        }

        // Iterate through possible values of a
        for ($a = 0; $a * $a <= $c; $a++) {
            $b_squared = $c - $a * $a;
            $b = (int)sqrt($b_squared);
            if ($b * $b == $b_squared) {
                return true;
            }
        }

        return false;
    }

    echo isPy(5) ? "true" : "false";  // true (1^2 + 2^2 = 5)
    echo isPy(3) ? "true" : "false";  // false
    ?>
</body>
</html>