// 🌗 Theme toggle functionality
const toggle = document.getElementById('theme-toggle');
const modeText = document.getElementById('mode-text');

// Load saved preference
if (localStorage.getItem('theme') === 'dark') {
  document.body.classList.add('dark-mode');
  toggle.checked = true;
  modeText.textContent = "🌙 Dark Mode";
}

toggle.addEventListener('change', () => {
  if (toggle.checked) {
    document.body.classList.add('dark-mode');
    document.body.classList.remove('light-mode');
    modeText.textContent = "🌙 Dark Mode";
    localStorage.setItem('theme', 'dark');
  } else {
    document.body.classList.remove('dark-mode');
    document.body.classList.add('light-mode');
    modeText.textContent = "🌞 Light Mode";
    localStorage.setItem('theme', 'light');
  }
});