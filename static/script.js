
// Additional utility functions for enhanced functionality
function showView(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
        view.classList.add('d-none');
    });

    // Show selected view
    const targetView = document.getElementById(viewName);
    if (targetView) {
        targetView.classList.add('active');
        targetView.classList.remove('d-none');
    }
}

// Image compression utility (optional, for large files)
function compressImage(file, maxWidth = 1920, quality = 0.8) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = function() {
            const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
            canvas.width = img.width * ratio;
            canvas.height = img.height * ratio;

            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(resolve, 'image/jpeg', quality);
        };

        img.src = URL.createObjectURL(file);
    });
}

document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('id_image_modal');
  const previewBlock = document.getElementById('file-preview-modal');
  const previewImg = document.getElementById('preview-image-modal');
  const removeBtn = document.getElementById('remove-file-modal');

  if (fileInput) {
    fileInput.addEventListener('change', function () {
      const file = this.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          previewImg.src = e.target.result;
          previewBlock.classList.remove('d-none');
        };
        reader.readAsDataURL(file);
      }
    });
  }

  if (removeBtn) {
    removeBtn.addEventListener('click', function () {
      previewImg.src = '#';
      previewBlock.classList.add('d-none');
      fileInput.value = '';
    });
  }
});


document.getElementById('receipt-upload-form-modal')
.addEventListener('submit', () => {
  console.log("Submitting receipt form");
});

const closeButton = document.querySelector('.alert .close');

// Hide the close button after 5 seconds
setTimeout(() => {
  closeButton.style.display = 'none';
}, 5000);

// Helper: Simulate logged-in user (replace with real auth)
let user = JSON.parse(localStorage.getItem('auth_user')) || null;

// Auto show profile if already logged in
document.getElementById('loginForm').addEventListener('submit', async function (e) {
  e.preventDefault();
  const email = document.getElementById('loginEmail').value;
  const password = document.getElementById('loginPassword').value;

  const response = await fetch('{% url "login_view" %}', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
    credentials: 'include'  // important for sessions
  });

  const data = await response.json();

  if (data.success) {
    location.href = "{% url 'profile' %}";  // redirect to profile
  } else {
    document.getElementById('loginError').textContent = data.error;
    document.getElementById('loginError').classList.remove('d-none');
  }
});

document.getElementById('signupForm').addEventListener('submit', async function (e) {
  e.preventDefault();
  const name = document.getElementById('signupName').value;
  const email = document.getElementById('signupEmail').value;
  const password = document.getElementById('signupPassword').value;

  const response = await fetch('{% url "signup_view" %}', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      username: name,
      email: email,
      password1: password,
      password2: password
    }),
    credentials: 'include'
  });

  const data = await response.json();

  if (data.success) {
    location.href = "{% url 'profile' %}";
  } else {
    document.getElementById('signupError').textContent =
      data.errors?.username?.[0]?.message || 'Signup failed.';
    document.getElementById('signupError').classList.remove('d-none');
  }
});

document.getElementById('logoutBtn').addEventListener('click', async function () {
  const response = await fetch('{% url "logout_view" %}', {
    method: 'POST',
    headers: { 'X-CSRFToken': '{{ csrf_token }}' },
    credentials: 'include'
  });

  const data = await response.json();
  if (data.success) {
    location.href = "/";
  } else {
    alert("Logout failed.");
  }
});

document.addEventListener("DOMContentLoaded", function () {
  const chartDataScript = document.getElementById("chart-data");
  const canvas = document.getElementById("consumptionChart");

  if (!chartDataScript || !canvas) {
    console.error("Required elements not found.");
    return;
  }

  try {
    const chartData = JSON.parse(chartDataScript.textContent);

    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: chartData.labels,
        datasets: [{
          label: 'Consumption',
          data: chartData.values,
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  } catch (err) {
    console.error("Chart render failed:", err);
  }
});