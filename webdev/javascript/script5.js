

// Using the event object (recommended if you bind this function as an event handler)
function opentab(tabname, e) {
    // Remove the "active-link" class from all elements with the "tab-links" class
    $(".tab-links").removeClass("active-link");
    // Remove the "active-tab" class from all elements with the "tab-contents" class
    $(".tab-contents").removeClass("active-tab");
    
    // Add the "active-link" class to the element that triggered the event.
    // (You can also use $(this) if you bind the handler directly.)
    $(e.currentTarget).addClass("active-link");
    
    // Select the element with the given id and add the "active-tab" class.
    $("#" + tabname).addClass("active-tab");
}



// Style functions for landing page transitions
const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        console.log(entry);
        if (entry.isIntersecting) {
            entry.target.classList.add('show');
        }
    });
});


const hiddenElts = document.querySelectorAll('.container');
hiddenElts.forEach((elt) => observer.observe(elt));
const hiddenElts2 = document.querySelectorAll('.nav-item');
hiddenElts2.forEach((elt) => observer.observe(elt));
const hiddenElts3 = document.querySelectorAll('.work-card');
hiddenElts2.forEach((elt) => observer.observe(elt));








 // Refactored to utilize jQuery, control the modals, and the modal's buttons:
$(document).ready(function() {
    // Get the slider items and thumbnail items as jQuery objects.
    // Must be specific because there are two different classes named "item"
    let $items = $('#projects .container .work-card');        

    // Number of items, and active item index.
    let countItem = $items.length;
    let itemActive = 0;


    // Enables and disables list items and thumbnail items.
    function showSlider() {
        // Get the currently active items and then 
        // remove them to activate the next one.
        $('#projects .container .work-card.active').removeClass('active');
    
        // Add the 'active' class to the next slider item.
        $items.eq(itemActive).addClass('active');
    } // showSlider()
  

    // Bind a click event on each thumbnail to let 
    // the user click a thumbnail directly.
    $items.each(function(index) {
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
        $("#project-modal").fadeOut();
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
  



































