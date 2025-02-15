// Refactored to utilize jQuery, control the modals, and the modal's buttons:
$(document).ready(function() {
    // Get the slider items and thumbnail items as jQuery objects.
    // Must be specific because there are two different classes named "item"
    let $items = $('.slider .list .item');         // Slider items
    let $thumbnails = $('.thumbnail .item');       // Thumbnail items
    let $next = $('#next');                        // Next button
    let $prev = $('#prev');                        // Prev button
  
    // Number of items, and active item index.
    let countItem = $items.length;
    let itemActive = 0;
  


    // ---- Home Page Navigation Buttons ----
    // Method to traverse the slider active thumbnails via next button
    $next.on('click', function() {
        itemActive = (itemActive + 1) % countItem;
        showSlider();
    });
  
    // Method to traverse the slider active thumbnails via prev button
    $prev.on('click', function() {
        itemActive = (itemActive - 1 + countItem) % countItem;
        showSlider();
    });
  


    // Automatically click the next button to simulate cycling.
    let refreshInterval = setInterval(() => {
        $next.trigger('click');
    }, 5000);
  

    // Enables and disables list items and thumbnail items.
    function showSlider() {
        // Get the currently active items and then 
        // remove them to activate the next one.
        $('.slider .list .item.active').removeClass('active');
        $('.thumbnail .item.active').removeClass('active');
    
        // Add the 'active' class to the next slider item and thumbnail.
        $items.eq(itemActive).addClass('active');
        $thumbnails.eq(itemActive).addClass('active');


        // Restart the auto-cycle iff the modal is closed.
        if (!$("#project-modal").is(":visible")){
            // Reset the refresh interval.
            clearInterval(refreshInterval);
            refreshInterval = setInterval(function(){
                $next.trigger('click');
            }, 5000);
        }
    } // showSlider()
  

    // Bind a click event on each thumbnail to let 
    // the user click a thumbnail directly.
    $thumbnails.each(function(index) {
        $(this).on('click', function() {
            itemActive = index;
            showSlider();
            // Open the modal with the full view.
            openModal(index);
        });
    });


    /**
     * Use jQuery’s event binding to attach a click handler to the modal’s close button and also to the modal background (so clicking outside the content closes the modal).
     * 
     * In the thumbnail click event (which already updates the slider), we call an openModal(index) function that gathers the full‑sized image source and description from the corresponding slider item and then displays the modal.
    */
    

    // Function to open the modal with full image and description for a given index.
    function openModal(index) {
        // Pause auto-cycling.
        clearInterval(refreshInterval);

        // Get the corresponding slider item.
        let $sliderItem = $items.eq(index);
        let imgSrc = $sliderItem.find('img').attr('src');

        // Use the modal-specific description from .m-content.
        let description = $sliderItem.find('.m-content').html();
        
        // Get website and GitHub URLs from the data attributes.
        let websiteURL = $sliderItem.data('website');
        let githubURL = $sliderItem.data('github');
    
        // Set the modal's image source and description.
        $("#modal-image").attr("src", imgSrc);
        $("#modal-description").html(description);
        $("#project-website").attr("href", websiteURL);
        $("#github-repo").attr("href", githubURL);
    
        // Show the modal with a fade-in effect.
        $("#project-modal").fadeIn();
    } // openModal()



    // Function to close the modal.
    function closeModal() {
        $("#project-modal").fadeOut(function() {
            // Resume auto-cycling after the modal is closed.
            refreshInterval = setInterval(() => {
                $next.trigger('click');
            }, 5000);
        });
    } // closeModal()



    // Attach modal event handlers.
    $("#project-modal .close").on("click", closeModal);

    // Also close the modal if the user clicks on the background (outside the modal-content).
    $("#project-modal").on("click", function(e) {
    if ($(e.target).is("#project-modal")) {
        closeModal();
    }
    });



    // ---- Modal Navigation Buttons ----
    function updateActiveItem(newIdx) {
        itemActive = newIdx;
        showSlider();
        openModal(itemActive);
    } // updateActiveItem()

    $("#modal-prev").on("click", function(e) {
        e.stopPropagation(); // Prevent event bubbling so modal doesn't close.
        updateActiveItem((itemActive - 1 + countItem) % countItem);
    });

    $("#modal-next").on("click", function(e) {
        e.stopPropagation();
        updateActiveItem((itemActive + 1) % countItem);
    });
  }); // ready()
  














