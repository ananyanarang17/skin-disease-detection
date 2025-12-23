// 🌗 Apply saved theme instantly to avoid white flash (runs before DOM loads)
(function() {
  const savedTheme = localStorage.getItem("theme") || "light";
  document.body.classList.toggle("dark-mode", savedTheme === "dark");
  document.body.classList.toggle("light-mode", savedTheme === "light");
})();

/* ========================================================================== */
/* 🚀 MAIN APPLICATION LOGIC - Everything runs after DOM is ready */
/* ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  console.log("✅ Application initialized");

  /* ------------------------------------------------------------------------ */
  /* 🌗 THEME TOGGLE */
  /* ------------------------------------------------------------------------ */
  
  const modeText = document.getElementById("mode-text");
  const chatPopup = document.getElementById("chat-popup");

  function applyTheme(theme) {
    document.body.classList.toggle("dark-mode", theme === "dark");
    document.body.classList.toggle("light-mode", theme === "light");

    const toggle = document.getElementById("theme-toggle");
    if (toggle) toggle.checked = theme === "dark";
    if (modeText) modeText.textContent = theme === "dark" ? "🌙 Dark Mode" : "🌞 Light Mode";
    if (chatPopup) chatPopup.classList.toggle("dark-mode", theme === "dark");
  }

  function bindToggle() {
    const toggle = document.getElementById("theme-toggle");
    if (!toggle) return false;
    
    toggle.addEventListener("change", () => {
      const newTheme = toggle.checked ? "dark" : "light";
      localStorage.setItem("theme", newTheme);
      applyTheme(newTheme);
    });
    return true;
  }

  // Retry binding if toggle not found (handles Jinja templates)
  if (!bindToggle()) {
    const retry = setInterval(() => {
      if (bindToggle()) clearInterval(retry);
    }, 200);
  }

  // Apply saved theme on load
  applyTheme(localStorage.getItem("theme") || "light");

  /* ------------------------------------------------------------------------ */
  /* 💡 TIP OF THE DAY */
  /* ------------------------------------------------------------------------ */
  
  const tips = [
    "Wash your pillowcases twice a week to prevent acne.",
    "Always use SPF, even indoors.",
    "Drink plenty of water to keep your skin hydrated.",
    "Avoid touching your face frequently.",
    "Use a mild cleanser suitable for your skin type."
  ];

  const dailyTip = document.getElementById("daily-tip");
  if (dailyTip) {
    const todayIndex = new Date().getDate() % tips.length;
    dailyTip.textContent = tips[todayIndex];
  }

  /* ------------------------------------------------------------------------ */
  /* 🌦️ WEATHER-BASED SKINCARE ADVICE */
  /* ------------------------------------------------------------------------ */
  
  const weatherInfo = document.getElementById("weather-info");
  if (weatherInfo && navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(async (pos) => {
      try {
        const { latitude, longitude } = pos.coords;
        const res = await fetch(
          `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true`
        );
        const data = await res.json();
        const temp = data.current_weather?.temperature;
        
        let advice;
        if (temp < 15) advice = "It's cold! Use a thicker moisturizer ❄️";
        else if (temp < 28) advice = "Mild weather — go with a balanced skincare routine 🌤️";
        else advice = "Hot weather — use lightweight gel-based products ☀️";
        
        weatherInfo.textContent = `Current temp: ${temp}°C — ${advice}`;
      } catch (error) {
        console.error("Weather fetch error:", error);
        weatherInfo.textContent = "Unable to fetch weather info.";
      }
    }, () => {
      weatherInfo.textContent = "Location access denied.";
    });
  }

  /* ------------------------------------------------------------------------ */
  /* 🧬 SKIN TYPE QUIZ */
  /* ------------------------------------------------------------------------ */
  
  const quizSubmit = document.getElementById("quiz-submit");
  if (quizSubmit) {
    quizSubmit.addEventListener("click", () => {
      const q1 = document.getElementById("q1")?.value;
      if (!q1) return;

      let result;
      if (q1.includes("Dry")) result = "Your skin type is: Dry 🧴";
      else if (q1.includes("Oily")) result = "Your skin type is: Oily 🌞";
      else if (q1.includes("Combination")) result = "Your skin type is: Combination 🌗";
      else result = "Your skin type is: Normal 😊";

      const resBox = document.getElementById("quiz-result");
      if (resBox) {
        resBox.textContent = result;
        resBox.classList.remove("hidden");
      }
    });
  }

  /* ------------------------------------------------------------------------ */
  /* 🧴 PERSONALIZED ROUTINE GENERATOR */
  /* ------------------------------------------------------------------------ */
  
  const generateBtn = document.getElementById("generate-routine");
  if (generateBtn) {
    generateBtn.addEventListener("click", () => {
      const type = document.getElementById("routine-skin-type")?.value;
      const output = document.getElementById("routine-output");
      if (!type || !output) return;

      let routine = "";
      
      if (type === "dry") {
        routine = `
          <b>Morning:</b> Gentle hydrating cleanser → Hyaluronic serum → Moisturizer → SPF 50<br>
          <b>Night:</b> Rich cream → Vitamin E serum → Sleep mask 💤
        `;
      } else if (type === "oily") {
        routine = `
          <b>Morning:</b> Foam cleanser → Niacinamide serum → Oil-free moisturizer → SPF 30<br>
          <b>Night:</b> Salicylic acid toner → Gel moisturizer
        `;
      } else if (type === "combination") {
        routine = `
          <b>Morning:</b> Balanced cleanser → Vitamin C serum → Moisturizer → SPF 50<br>
          <b>Night:</b> Lightweight night cream → Exfoliate twice weekly
        `;
      } else {
        routine = `
          <b>Morning:</b> Fragrance-free cleanser → Soothing serum → Moisturizer → SPF 30<br>
          <b>Night:</b> Aloe vera gel → Barrier-repair cream
        `;
      }
      
      output.innerHTML = routine;
    });
  }

  /* ------------------------------------------------------------------------ */
  /* 💬 CHATBOT - FIXED AND WORKING */
  /* ------------------------------------------------------------------------ */
  
  const chatIcon = document.getElementById("chat-icon");
  const chatPopupEl = document.getElementById("chat-popup");
  const closeChat = document.getElementById("close-chat");
  const sendBtn = document.getElementById("send-btn");
  const userInput = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");

  if (chatIcon && chatPopupEl) {
    console.log("✅ Chatbot elements found");

    // Open chat
    chatIcon.addEventListener("click", () => {
      console.log("💬 Chat opened");
      chatPopupEl.classList.add("active");
      chatIcon.style.display = "none";
    });

    // Close chat
    if (closeChat) {
      closeChat.addEventListener("click", () => {
        console.log("❌ Chat closed");
        chatPopupEl.classList.remove("active");
        chatIcon.style.display = "flex";
      });
    }

    // Send message function with bubble UI
    const sendMessage = async () => {
      const message = userInput?.value.trim();
      if (!message || !chatBox) return;

      // Add user message bubble
      chatBox.innerHTML += `
        <p class="user-message">
          <span class="bubble">${message}</span>
        </p>
      `;
      userInput.value = "";
      chatBox.scrollTop = chatBox.scrollHeight;

      // Show typing indicator
      const typingDiv = document.createElement('p');
      typingDiv.className = 'bot-message';
      typingDiv.innerHTML = `
        <span class="typing-indicator">
          <span></span><span></span><span></span>
        </span>
      `;
      chatBox.appendChild(typingDiv);
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            message,
            context: "skincare" 
          }),
        });

        const data = await response.json();
        
        // Remove typing indicator
        typingDiv.remove();
        
        // Add bot response bubble
        chatBox.innerHTML += `
          <p class="bot-message">
            <span class="bubble">${data.reply}</span>
          </p>
        `;
        chatBox.scrollTop = chatBox.scrollHeight;
      } catch (error) {
        console.error("Chat error:", error);
        typingDiv.remove();
        chatBox.innerHTML += `
          <p class="bot-message">
            <span class="bubble" style="background:#f44336;color:white;">Connection error ❌</span>
          </p>
        `;
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    };

    // Send button click
    if (sendBtn) {
      sendBtn.addEventListener("click", sendMessage);
    }

    // Send on Enter key
    if (userInput) {
      userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
      });
    }
  } else {
    console.error("❌ Chatbot elements not found");
  }

  /* ------------------------------------------------------------------------ */
  /* 👤 AUTH MODAL & AUTHENTICATION */
  /* ------------------------------------------------------------------------ */
  
  const authModal = document.getElementById("auth-modal");
  const authTrigger = document.getElementById("auth-trigger");
  const closeModal = document.getElementById("close-modal");
  const signupFormEl = document.getElementById("signup-form");
  const loginFormEl = document.getElementById("login-form");
  const showLogin = document.getElementById("show-login");
  const showSignup = document.getElementById("show-signup");

  // Check auth status on page load
  checkAuthStatus();

  if (authTrigger && authModal) {
    authTrigger.addEventListener("click", () => {
      authModal.style.display = "flex";
    });

    if (closeModal) {
      closeModal.addEventListener("click", () => {
        authModal.style.display = "none";
      });
    }

    // Close on outside click
    window.addEventListener("click", (e) => {
      if (e.target === authModal) {
        authModal.style.display = "none";
      }
    });

    // Toggle between signup and login
    if (showLogin && showSignup && signupFormEl && loginFormEl) {
      showLogin.addEventListener("click", (e) => {
        e.preventDefault();
        signupFormEl.style.display = "none";
        loginFormEl.style.display = "block";
      });

      showSignup.addEventListener("click", (e) => {
        e.preventDefault();
        loginFormEl.style.display = "none";
        signupFormEl.style.display = "block";
      });
    }
  }

  /* -------------------------------------------------------------------- */
  /* SIGNUP FORM HANDLER */
  /* -------------------------------------------------------------------- */
  
  if (signupFormEl) {
    signupFormEl.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      const username = document.getElementById('signup-username')?.value.trim();
      const email = document.getElementById('signup-email')?.value.trim();
      const password = document.getElementById('signup-password')?.value;
      const submitBtn = signupFormEl.querySelector('button[type="submit"]');

      if (!username || !email || !password) {
        alert('⚠️ Please fill in all fields!');
        return;
      }

      const originalText = submitBtn.textContent;
      submitBtn.textContent = '⏳ Creating account...';
      submitBtn.disabled = true;

      try {
        const response = await fetch('/signup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, email, password })
        });

        const data = await response.json();

        if (data.success) {
          alert(data.message);
          updateUIForLoggedInUser(data.user);
          authModal.style.display = 'none';
          signupFormEl.reset();
        } else {
          alert(data.message);
        }
      } catch (error) {
        console.error('Signup error:', error);
        alert('❌ Connection error. Please try again.');
      } finally {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
      }
    });
  }

  /* -------------------------------------------------------------------- */
  /* LOGIN FORM HANDLER */
  /* -------------------------------------------------------------------- */
  
  if (loginFormEl) {
    loginFormEl.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      const email = document.getElementById('login-email')?.value.trim();
      const password = document.getElementById('login-password')?.value;
      const submitBtn = loginFormEl.querySelector('button[type="submit"]');

      if (!email || !password) {
        alert('⚠️ Please fill in all fields!');
        return;
      }

      const originalText = submitBtn.textContent;
      submitBtn.textContent = '⏳ Logging in...';
      submitBtn.disabled = true;

      try {
        const response = await fetch('/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });

        const data = await response.json();

        if (data.success) {
          alert(data.message);
          updateUIForLoggedInUser(data.user);
          authModal.style.display = 'none';
          loginFormEl.reset();
        } else {
          alert(data.message);
        }
      } catch (error) {
        console.error('Login error:', error);
        alert('❌ Connection error. Please try again.');
      } finally {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
      }
    });
  }

  /* -------------------------------------------------------------------- */
  /* HELPER FUNCTIONS */
  /* -------------------------------------------------------------------- */

  async function checkAuthStatus() {
    try {
      const response = await fetch('/check-auth');
      const data = await response.json();
      
      if (data.authenticated) {
        updateUIForLoggedInUser(data.user);
      } else {
        updateUIForLoggedOutUser();
      }
    } catch (error) {
      console.error('Auth check error:', error);
    }
  }

  function updateUIForLoggedInUser(user) {
    // Hide login/signup button
    if (authTrigger) {
      authTrigger.style.display = 'none';
    }

    // Show user section
    let userSection = document.getElementById('user-section');
    if (!userSection) {
      userSection = document.createElement('div');
      userSection.id = 'user-section';
      userSection.innerHTML = `
        <span style="color: #0288d1; font-weight: 600; margin-right: 15px;">
          👤 ${user.username}
        </span>
        <button id="logout-btn" class="auth-btn">Logout</button>
      `;
      
      const authSection = document.querySelector('.auth-section');
      if (authSection) {
        authSection.appendChild(userSection);
      }
      
      // Attach logout handler
      document.getElementById('logout-btn').addEventListener('click', async () => {
        const response = await fetch('/logout', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
          alert(data.message);
          updateUIForLoggedOutUser();
        }
      });
    } else {
      userSection.style.display = 'block';
      userSection.querySelector('span').textContent = `👤 ${user.username}`;
    }

    console.log('✅ User logged in:', user.username);
  }

  function updateUIForLoggedOutUser() {
    if (authTrigger) {
      authTrigger.style.display = 'inline-block';
    }

    const userSection = document.getElementById('user-section');
    if (userSection) {
      userSection.style.display = 'none';
    }

    console.log('✅ User logged out');
  }

  /* ------------------------------------------------------------------------ */
  /* 📊 REPORTS PAGE */
  /* ------------------------------------------------------------------------ */
  
  const generateForm = document.getElementById("generate-report-form");
  const fileInput = document.getElementById("report-upload");

  if (generateForm && fileInput) {
    generateForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const file = fileInput.files[0];
      if (!file) {
        alert("Please upload a file before generating a report!");
        return;
      }

      const button = generateForm.querySelector("button");
      if (button) {
        button.textContent = "⏳ Generating...";
        button.disabled = true;

        // Simulate processing (replace with actual backend call)
        setTimeout(() => {
          alert("✅ Report generated successfully! You can now download it from 'Recent Reports'.");
          button.textContent = "📄 Generate Report";
          button.disabled = false;
        }, 2000);
      }
    });
  }

  // Download report buttons
  const downloadButtons = document.querySelectorAll(".download-btn");
  downloadButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      alert("📥 Report downloaded successfully (mock).");
    });
  });

  /* ------------------------------------------------------------------------ */
  /* 📨 FEEDBACK FORM - FIXED */
  /* ------------------------------------------------------------------------ */
  
  const feedbackForm = document.getElementById("feedback-form");
  console.log("🔍 Feedback form element:", feedbackForm);

  if (feedbackForm) {
    console.log("✅ Feedback form found! Attaching submit listener...");
    
    feedbackForm.addEventListener("submit", function(e) {
      console.log("🎯 Form submit event triggered!");
      e.preventDefault();
      e.stopPropagation();

      // Get all form fields
      const name = feedbackForm.querySelector("input[name='name']")?.value.trim();
      const email = feedbackForm.querySelector("input[name='email']")?.value.trim();
      const message = feedbackForm.querySelector("textarea[name='message']")?.value.trim();
      const feedbackType = feedbackForm.querySelector("select[name='feedback_type']")?.value;
      const deviceType = feedbackForm.querySelector("select[name='device_type']")?.value;
      const rating = feedbackForm.querySelector("input[name='rating']")?.value;
      const page = feedbackForm.querySelector("input[name='page']")?.value;
      const agree = feedbackForm.querySelector("input[name='agree']")?.checked;

      console.log("📝 Form data:", { name, email, message, feedbackType, deviceType, rating, page, agree });

      // Validate required fields
      if (!name || !email || !message) {
        alert("⚠️ Please fill out all required fields before submitting.");
        return;
      }

      if (!agree) {
        alert("⚠️ Please agree to the consent before submitting.");
        return;
      }

      const submitBtn = feedbackForm.querySelector("button[type='submit']");
      const originalText = submitBtn?.textContent || "Submit Feedback";
      if (submitBtn) {
        submitBtn.textContent = "⏳ Sending...";
        submitBtn.disabled = true;
      }

      (async () => {
        try {
          console.log("📤 Sending feedback to /feedback endpoint...");
          const response = await fetch("/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name,
              email,
              message,
              feedback_type: feedbackType,
              device_type: deviceType,
              rating: rating ? parseInt(rating) : null,
              page: page || "unknown",
              agree: agree
            }),
          });

          console.log("📬 Response status:", response.status);
          const data = await response.json();
          console.log("📬 Response data:", data);
          
          if (data.success) {
            alert("✅ Thank you for your feedback!");
            feedbackForm.reset();
          }else {
            alert(data.reply || "❌ Failed to submit feedback. Please try again.");
          }
        } catch (error) {
          console.error("❌ Feedback submission failed:", error);
          alert("❌ Connection error. Please try again later.");
        } finally {
          if (submitBtn) {
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
          }
        }
      })();
    });
  } else {
    console.error("❌ Feedback form NOT FOUND! Check your HTML");
  }

  console.log("✅ All features initialized successfully");
});

/* ========================================================================== */
/* ⭐ STAR RATING LOGIC */
/* ========================================================================== */

document.querySelectorAll(".star-rating .star").forEach((star, index, stars) => {
  star.addEventListener("click", () => {
    const value = parseInt(star.getAttribute("data-value"));
    document.getElementById("satisfaction").value = value;

    stars.forEach((s, i) => {
      s.classList.toggle("filled", i < value);
    });
  });

  star.addEventListener("mouseover", () => {
    const value = parseInt(star.getAttribute("data-value"));
    stars.forEach((s, i) => {
      s.classList.toggle("hovered", i < value);
    });
  });

  star.addEventListener("mouseout", () => {
    stars.forEach((s) => s.classList.remove("hovered"));
  });
});