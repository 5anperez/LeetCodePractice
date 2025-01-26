
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




