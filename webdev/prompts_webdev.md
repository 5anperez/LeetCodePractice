P1:

  I need a webpage where the main focus is an auto running slider with a slider transition button and the ability to click on thumbnail images using HTML, CSS, and JS. Split it in three sections: 1. Header, 2. Slider, and 3. Thumbnails. The web page should cycle through images every 5 seconds, and each image will be accompanied by lines of text content. The images correspond to the thumbnails in the slider and should be the size of the entire veiw port but should be black on the bottom and transition via gradient. Regarding the buttons, there should be a next and a previous button to manually cycle through the thumbnail images that upon selected, stand out more than the other non-selected images. A user can also just simply click on any thumbnail to get the corresponding image. When the automatic cycle gets to the last image, it wraps around to the first, such that users can continually press the next button and never run out of images. Users can also click directly on a thumbnail to select the corresponding image and have its lines of text display if they do not want to click through the slider via the buttons and/or if they dont want to wait throught the automatic cycle. It is also a responsive design that is compatible with all screens.



A: 

  In the index.html file (UPDATE THIS WHEN YOU MOVE IT INTO THE PORTFOLIO, IF YOU DELETE THE ORIGINAL)



































P2:
Explain the main.css and the main.scss files and break them down into logical components. (within the html5-paradigm)


critique:
The model is a bit vague, e.g., when describing some components it only described what was already known and did not offer any insight like the fact that the icon components are reusable and that icon is a helper class that allows other elements to display font awesome icons; also, mentioning what modal functionality executes with the gallery component would be helpful, yet it only mentions the functionality is available. 




























P3:

I created an html css javascript website to showcase my projects and have created an auto running slider that cycles through thumbnail images, where the images represent the projects. However, the thumbnails are small, so I would like to add the functionality needed to make the thumbnails grow into their own pop-up window that displays the entire image along with a description of the project. Can you help me do this? Here is the html, css, and JS code that needs to be modified: 1. html: "<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="../css/style.css"
</head>


<body>

    <!--Header-->
    <header>
        <div class="logo">SP</div>
        <ul class="menu">
            <li>Home</li>
            <li>Blog</li>
            <li>Info</li>
        </ul>
    </header>

    <!--Slider: An auto running slider using HTML, CSS, and JS. It uses images to cycle through every 5 seconds. Each image will be accomp, by lines of text content. When the cycle gets to the last image, it wraps around to the first, such that users can continually press the next b utton and never run out of images. Also, there are buttons to manually cycle through thumbnail images that upon selected, stand out more than the other non-selected images. A user can also just simply click on any thumbnail to get the corresponding image. It is also a responsive design that is compatible with all screens. -->
    <div class="slider">

        <!--List-->
        <div class="list">
            <div class="item active">
                <img src="../images/euchreDalli.jpg">
                <div class="content">
                    <p>Project:</p> 
                    <h2>Euchre: </h2>
                    <p>The Card Game</p>
                </div>
            </div>
            <div class="item">
                <img src="../images/treasureHuntDalli.jpg">
                <div class="content">
                    <p>Project:</p> 
                    <h2>Treasure Hunt:</h2>
                    <p>Captain & First-Mate</p>
                </div>
            </div>
            <div class="item">
                <img src="../images/myLogo.png">
                <div class="content">
                    <p>Project:</p> 
                    <h2>My Logo:</h2>
                    <p>By Dalle</p>
                </div>
            </div>
            <div class="item">
                <img src="../images/finPlanWebsite.png">
                <div class="content">
                    <p>Project:</p> 
                    <h2>Finance:</h2>
                    <p>Planning</p>
                </div>
            </div>
            <div class="item">
                <img src="../images/me.jpg">
                <div class="content">
                    <p>Project:</p> 
                    <h2>Faceshot:</h2>
                    <p>Professh</p>
                </div>
            </div>
        </div>
        <!--List-->
        <!--Arrow Buttons-->
        <div class="arrows">
            <button id="prev"><</button>
            <button id="next">></button>
        </div>
        <!--Arrow Buttons-->
        <!--Thumbnails-->
        <div class="thumbnail">
            <div class="item active">
                <img src="../images/euchreDalli.jpg">
                <div class="content">
                    Euchre!
                </div>
            </div>
            <div class="item">
                <img src="../images/treasureHuntDalli.jpg">
                <div class="content">
                    Treasure Hunt!
                </div>
            </div>
            <div class="item">
                <img src="../images/myLogo.png">
                <div class="content">
                    My Logo 
                </div>
            </div>
            <div class="item">
                <img src="../images/finPlanWebsite.png">
                <div class="content">
                    Financial Planning
                </div>
            </div>
            <div class="item">
                <img src="../images/me.jpg">
                <div class="content">
                    Ya BOY
                </div>
            </div>
        </div>
        <!--Thumbnails-->
    </div>


    <script src="../javascript/freestyle.js"></script>
    
</body>


</html>", 2. css: "/* PUT A COOL FONT HERE! */

body{
  font-family: Arial, Helvetica, sans-serif;
  margin: 0;
  background-color: blueviolet;
  color: chartreuse;
}


header{
  width: 1200px;
  max-width: 90%;
  margin: auto;
  display: grid;
  grid-template-columns: 50px 1fr 50px;
  grid-template-rows: 50px;
  justify-content: center;
  align-items: center;
  position: relative;
  z-index: 100;
}

header .logo{
  font-weight: bold;
}
header .menu{
  color: blueviolet;
  font-weight: bolder;
  padding: 0;
  margin: 0;
  list-style: none;
  display: flex;
  justify-content: center;
  gap: 20px;
}

/* CSS Slider */
.slider{
  height: 100vh;
  margin-top: -50px;
  position: relative;
}
.slider .list .item{
  position: absolute;
  inset: 0 0 0 0;
  overflow: hidden;
  opacity: 0;
  transition: .5s;
}
.slider .list .item img{
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.slider .list .item::after{
  content: '';
  width: 100%;
  height: 100%;
  position: absolute;
  left: 0;
  bottom: 0;
  background-image: linear-gradient(
    to top, #000 20%, transparent
  );
}
.slider .list .item .content{
  position: absolute;
  left: 10%;
  top: 20%;
  width: 500px;
  max-width: 80%;
  z-index: 1;
}
.slider .list .item .content p:nth-child(1){
  text-transform: uppercase;
  letter-spacing: 10px;
}
.slider .list .item .content h2{
  font-size: 100px;
  margin: 0;
}
.slider .list .item.active{
  opacity: 1;
  z-index: 10;
}
@keyframes showContent {
  to{
    transform: translateY(0);
    filter: blur(0);
    opacity: 1;
  }
}
.slider .list .item.active p:nth-child(1),
.slider .list .item.active h2,
.slider .list .item.active p:nth-child(3){
  transform: translateY(30px);
  filter: blur(20px);
  opacity: 0;
  animation: showContent .5s .7s ease-in-out 1 forwards;
}
.slider .list .item.active h2{
  animation-delay: 1s;
}
.slider .list .item.active p:nth-child(3){
  animation-duration: 1.5s;
}

/* Arrow Buttons */
.arrows{
  position: absolute;
  top: 30%;
  right: 50px;
  z-index: 100;
}
.arrows button{
  background-color: blue;
  border: none;
  font-family: monospace;
  width: 40px;
  height: 40px;
  border-radius: 5px;
  font-size: x-large;
  color: aqua;
  transition: .5s;
}
.arrows button:hover{
  background-color: chartreuse;
  color: #000;
}

/* Thumbnails */
.thumbnail{
  position: absolute;
  bottom: 70px;
  z-index: 11;
  display: flex;
  gap: 10px;
  width: 100%;
  height: 250px;
  padding: 0 50px;
  box-sizing: border-box;
  overflow: auto;
  justify-content: center;
}
.thumbnail::-webkit-scrollbar{
  width: 0;
}
.thumbnail .item{
  position: relative;
  width: 150px;
  height: 220px;
  filter: brightness(.5);
  transition: .5s;
  flex-shrink: 0;
}
.thumbnail .item img{
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 10px;
}
.thumbnail .item.active{
  filter: brightness(1.5);
}
.thumbnail .item .content{
  position: absolute;
  inset: auto 10px 10px 10px;
  justify-content: center;
  color: aqua;
  font-size: 14px;
  text-align: center;
}
@media screen and (max-width: 678px) {
  .thumbnail{
    justify-content: start;
  }
  .slider .list .item .content h2{
    font-size: 60px;
  }
  .arrows{
    top: 10%;
  }
}
", 3. javascript: "
// Get the HTML items we are working with: list items, thumbnails, buttons
// Must be specific because there are two different classes w/ "item"
let items = document.querySelectorAll('.slider .list .item');   // slider class
let thumbnails = document.querySelectorAll('.thumbnail .item'); // thumnail cls
let next = document.getElementById('next');
let prev = document.getElementById('prev');


// Number of items (thumbnails) in the slider and the active element index.
let countItem = items.length;
let itemActive = 0;


// Method to traverse the slider active thumbnails via next button
next.onclick = function(){
    itemActive = itemActive + 1;
    if (itemActive >= countItem){
        itemActive = 0;
    }
    showSlider();
}


// Method to traverse the slider active thumbnails via prev button
prev.onclick = function(){
    itemActive = itemActive - 1;
    if (itemActive < 0){
        itemActive = countItem - 1;
    }
    showSlider();
}


// Automatically click the next button to simulate
// the cycling behavior.
let refreshInterval = setInterval(() => {
    next.click();
}, 5000)


function showSlider(){
    // Get the current active elements
    let itemActiveOld = document.querySelector('.slider .list .item.active');
    let thumbnailActiveOld = document.querySelector('.thumbnail .item.active');
    
    // Remove them to activate the next one.
    itemActiveOld.classList.remove('active');
    thumbnailActiveOld.classList.remove('active');

    // Add the 'active' class onto the next element in line.
    items[itemActive].classList.add('active');
    thumbnails[itemActive].classList.add('active');

    clearInterval(refreshInterval);
    refreshInterval = setInterval(() => {
        next.click();
    }, 5000)
}


// Method to let the user click a thumbnail directly.
thumbnails.forEach((thumbnail, index) => {
    thumbnail.addEventListener('click', () => {
        itemActive = index;
        showSlider();
    })
})
"
















P3.2:

Translate it to utilize jQuery




P3.3

Add buttons to take the user to the prj site and the git repo 


P3.4:

So the site has prev and next if the user doesnt want to wait for the auto cycling. Lets add those as well 

P3.5

didnt work, debug



P3.6

Add an inverted border or how to unlink the current "content" in the modal and add a unique one?


































































