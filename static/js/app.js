// Custom JavaScript for HepatoTox Predictor

// Global utility functions
window.HepatoTox = {
    // Show toast notification
    showToast: function(message, type = 'info') {
        const bgColor = {
            'success': 'bg-success',
            'error': 'bg-danger',
            'warning': 'bg-warning',
            'info': 'bg-info'
        }[type];

        const toast = `
            <div class="toast align-items-center text-white ${bgColor} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        // Append toast and show
        const container = $('#toastContainer');
        if (container.length === 0) {
            $('body').append('<div id="toastContainer" class="position-fixed bottom-0 end-0 p-3" style="z-index: 11"></div>');
        }
        $('#toastContainer').append(toast);
        $('.toast').last().toast({ delay: 3000 }).toast('show');
    },

    // Format number with commas
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },

    // Copy to clipboard
    copyToClipboard: function(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showToast('Copied to clipboard!', 'success');
        });
    }
};

// Initialize tooltips
$(document).ready(function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Smooth scroll for anchor links
$('a[href^="#"]').on('click', function(e) {
    e.preventDefault();
    const target = $(this.getAttribute('href'));
    if (target.length) {
        $('html, body').stop().animate({
            scrollTop: target.offset().top - 70
        }, 1000);
    }
});

// Add fade-in animation to elements
$(window).on('scroll', function() {
    $('.fade-in-on-scroll').each(function() {
        const elementTop = $(this).offset().top;
        const viewportBottom = $(window).scrollTop() + $(window).height();
        if (viewportBottom > elementTop) {
            $(this).addClass('fade-in');
        }
    });
});
