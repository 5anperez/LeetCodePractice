
// // Get the HTML items we are working with: list items, thumbnails, buttons
// // Must be specific because there are two different classes w/ "item"
// let items = document.querySelectorAll('.slider .list .item');   // slider class
// let thumbnails = document.querySelectorAll('.thumbnail .item'); // thumnail cls
// let next = document.getElementById('next');
// let prev = document.getElementById('prev');


// // Number of items (thumbnails) in the slider and the active element index.
// let countItem = items.length;
// let itemActive = 0;


// // Method to traverse the slider active thumbnails via next button
// next.onclick = function(){
//     itemActive = itemActive + 1;
//     if (itemActive >= countItem){
//         itemActive = 0;
//     }
//     showSlider();
// }


// // Method to traverse the slider active thumbnails via prev button
// prev.onclick = function(){
//     itemActive = itemActive - 1;
//     if (itemActive < 0){
//         itemActive = countItem - 1;
//     }
//     showSlider();
// }


// // Automatically click the next button to simulate
// // the cycling behavior.
// let refreshInterval = setInterval(() => {
//     next.click();
// }, 5000)


// function showSlider(){
//     // Get the current active elements
//     let itemActiveOld = document.querySelector('.slider .list .item.active');
//     let thumbnailActiveOld = document.querySelector('.thumbnail .item.active');
    
//     // Remove them to activate the next one.
//     itemActiveOld.classList.remove('active');
//     thumbnailActiveOld.classList.remove('active');

//     // Add the 'active' class onto the next element in line.
//     items[itemActive].classList.add('active');
//     thumbnails[itemActive].classList.add('active');

//     clearInterval(refreshInterval);
//     refreshInterval = setInterval(() => {
//         next.click();
//     }, 5000)
// }


// // Method to let the user click a thumbnail directly.
// thumbnails.forEach((thumbnail, index) => {
//     thumbnail.addEventListener('click', () => {
//         itemActive = index;
//         showSlider();
//     })
// })




// // Modal elements
// let modal = document.getElementById('projectModal');
// let modalImage = document.getElementById('modalImage');
// let modalTitle = document.getElementById('modalTitle');
// let modalDescription = document.getElementById('modalDescription');
// let closeModal = document.querySelector('.close');

// // Thumbnail click event to open modal
// thumbnails.forEach((thumbnail, index) => {
//   thumbnail.addEventListener('click', () => {
//     itemActive = index;
//     showSlider();

//     // Get the corresponding full image and description
//     let fullItem = items[index];
//     let fullImageSrc = fullItem.querySelector('img').src;
//     let fullTitle = fullItem.querySelector('h2').innerText;
//     let fullDescription = fullItem.querySelector('p:nth-child(3)').innerText;

//     // Populate modal with data
//     modalImage.src = fullImageSrc;
//     modalTitle.innerText = fullTitle;
//     modalDescription.innerText = fullDescription;

//     // Show modal
//     modal.style.display = 'flex';
//   });
// });

// // Close modal when "×" is clicked
// closeModal.onclick = function () {
//   modal.style.display = 'none';
// };

// // Close modal when clicking outside the modal content
// window.onclick = function (event) {
//   if (event.target === modal) {
//     modal.style.display = 'none';
//   }
// };











// // Slider elements
// let items = $('.slider .list .item');
// let thumbnails = $('.thumbnail .item');
// let countItem = items.length;
// let itemActive = 0;

// // Next button
// $('#next').on('click', function () {
//   itemActive = (itemActive + 1) % countItem;
//   showSlider();
// });

// // Previous button
// $('#prev').on('click', function () {
//   itemActive = (itemActive - 1 + countItem) % countItem;
//   showSlider();
// });

// // Auto-cycle
// let refreshInterval = setInterval(() => $('#next').click(), 5000);

// // Thumbnail click to open modal
// thumbnails.on('click', function () {
//   itemActive = thumbnails.index(this);
//   showSlider();

//   let fullItem = items.eq(itemActive);
//   let fullImageSrc = fullItem.find('img').attr('src');
//   let fullTitle = fullItem.find('h2').text();
//   let fullDescription = fullItem.find('p:nth-child(3)').text();

//   $('#modalImage').attr('src', fullImageSrc);
//   $('#modalTitle').text(fullTitle);
//   $('#modalDescription').text(fullDescription);

//   $('#projectModal').css('display', 'flex');
// });

// // Close modal
// $('.close').on('click', function () {
//   $('#projectModal').hide();
// });

// // Close modal when clicking outside
// $(window).on('click', function (event) {
//   if (event.target.id === 'projectModal') {
//     $('#projectModal').hide();
//   }
// });

// // Update slider
// function showSlider() {
//   $('.slider .list .item.active, .thumbnail .item.active').removeClass('active');
//   items.eq(itemActive).addClass('active');
//   thumbnails.eq(itemActive).addClass('active');

//   clearInterval(refreshInterval);
//   refreshInterval = setInterval(() => $('#next').click(), 5000);
// }











// Project data
const projects = [
  {
    github: "https://github.com/5anperez/",
    site: "https://google.com"
  },
  {
    github: "https://github.com/yourusername/treasure-hunt",
    site: "https://your-treasure-hunt-site.com"
  },
  {
    github: "https://github.com/yourusername/my-logo",
    site: "https://your-logo-site.com"
  },
  {
    github: "https://github.com/yourusername/finance-planning",
    site: "https://your-finance-site.com"
  },
  {
    github: "https://github.com/yourusername/faceshot",
    site: "https://your-faceshot-site.com"
  }
];

// Slider elements
let items = $('.slider .list .item');
let thumbnails = $('.thumbnail .item');
let countItem = items.length;
let itemActive = 0;

// Next button
$('#next').on('click', function () {
  itemActive = (itemActive + 1) % countItem;
  showSlider();
});

// Previous button
$('#prev').on('click', function () {
  itemActive = (itemActive - 1 + countItem) % countItem;
  showSlider();
});

// Auto-cycle
let refreshInterval = setInterval(() => $('#next').click(), 5000);

// Thumbnail click to open modal
thumbnails.on('click', function () {
  itemActive = thumbnails.index(this);
  showSlider();
  openModal();
});

// Close modal
$('.close').on('click', function () {
  $('#projectModal').hide();
});

// Close modal when clicking outside
$(window).on('click', function (event) {
  if (event.target.id === 'projectModal') {
    $('#projectModal').hide();
  }
});

// Modal Previous button
$('#modalPrev').on('click', function () {
  itemActive = (itemActive - 1 + countItem) % countItem;
  showSlider();
  openModal();
});

// Modal Next button
$('#modalNext').on('click', function () {
  itemActive = (itemActive + 1) % countItem;
  showSlider();
  openModal();
});

// Update slider
function showSlider() {
  $('.slider .list .item.active, .thumbnail .item.active').removeClass('active');
  items.eq(itemActive).addClass('active');
  thumbnails.eq(itemActive).addClass('active');

  clearInterval(refreshInterval);
  refreshInterval = setInterval(() => $('#next').click(), 5000);
}

// Open modal and populate content
function openModal() {
  let fullItem = items.eq(itemActive);
  let fullImageSrc = fullItem.find('img').attr('src');
  let fullTitle = fullItem.find('h2').text();
  let modalDescription = fullItem.find('.m-content p').text();

  $('#modalImage').attr('src', fullImageSrc);
  $('#modalTitle').text(fullTitle);
  $('#modalDescription').text(modalDescription);

  $('#githubButton').attr('href', projects[itemActive].github);
  $('#siteButton').attr('href', projects[itemActive].site);

  $('#projectModal').css('display', 'flex');
}























