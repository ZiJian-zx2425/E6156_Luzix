// static/js/login_status.js
document.addEventListener('DOMContentLoaded', function() {
    fetch('/api/is_logged_in')
        .then(response => response.json())
        .then(data => {
            const patientButton = document.getElementById('patientButton');
            const dropdownContent = patientButton.nextElementSibling;
            // Check if logged in and set button text based on the role
            if (data.logged_in && data.role) {
                patientButton.textContent = data.role.charAt(0).toUpperCase() + data.role.slice(1);is_logged_in
            } else {
                patientButton.textContent = 'Not Logged In';
                // Remove the logout link if it exists
                const logoutLink = dropdownContent.querySelector('.logout-link');
                if (logoutLink) {
                    dropdownContent.removeChild(logoutLink);
                }
            }
        })
        .catch(error => console.error('Error:', error));
});
