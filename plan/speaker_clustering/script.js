// Speaker Clustering Planning - Interactive JavaScript

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    // Remove loader after page loads
    setTimeout(() => {
        const loader = document.getElementById('loader');
        loader.classList.add('fade-out');
        setTimeout(() => loader.style.display = 'none', 500);
    }, 1000);

    // Smooth scrolling for navigation links
    initSmoothScrolling();
    
    // Theme toggle functionality
    initThemeToggle();
    
    // Tab functionality
    initTabs();
    
    // Copy code functionality
    initCopyButtons();
    
    // Intersection Observer for animations
    initScrollAnimations();
    
    // Active navigation highlighting
    initActiveNavigation();
});

// Smooth scrolling
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const offset = 100; // Account for fixed nav
                const targetPosition = target.offsetTop - offset;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Theme toggle
function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const root = document.documentElement;
    let isDark = true;

    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        toggleTheme();
    }

    themeToggle.addEventListener('click', toggleTheme);

    function toggleTheme() {
        isDark = !isDark;
        
        if (isDark) {
            root.style.setProperty('--bg-dark', '#0f172a');
            root.style.setProperty('--bg-light', '#1e293b');
            root.style.setProperty('--text-primary', '#f1f5f9');
            root.style.setProperty('--text-secondary', '#94a3b8');
            root.style.setProperty('--border-color', '#334155');
            themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
            localStorage.setItem('theme', 'dark');
        } else {
            root.style.setProperty('--bg-dark', '#f8fafc');
            root.style.setProperty('--bg-light', '#ffffff');
            root.style.setProperty('--text-primary', '#0f172a');
            root.style.setProperty('--text-secondary', '#475569');
            root.style.setProperty('--border-color', '#e2e8f0');
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
            localStorage.setItem('theme', 'light');
        }
    }
}

// Tab functionality
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Remove active class from all buttons and panels
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));
            
            // Add active class to clicked button and corresponding panel
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
}

// Copy code functionality
function initCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const targetId = button.getAttribute('data-target');
            const codeElement = document.getElementById(targetId);
            const textToCopy = codeElement.textContent;
            
            try {
                await navigator.clipboard.writeText(textToCopy);
                
                // Show success state
                const originalHTML = button.innerHTML;
                button.innerHTML = '<i class="fas fa-check"></i> Copied!';
                button.classList.add('copied');
                
                // Reset after 2 seconds
                setTimeout(() => {
                    button.innerHTML = originalHTML;
                    button.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
            }
        });
    });
}

// Scroll animations
function initScrollAnimations() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all feature cards and sections
    const animatedElements = document.querySelectorAll('.feature-card, .req-card, .flow-step, .tech-item');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Active navigation highlighting
function initActiveNavigation() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    function updateActiveNav() {
        const scrollPosition = window.scrollY + 150; // Offset for fixed nav
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionBottom = sectionTop + section.offsetHeight;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                const currentId = section.getAttribute('id');
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${currentId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }
    
    window.addEventListener('scroll', updateActiveNav);
    updateActiveNav(); // Initial call
}

// Add floating animation to gradient orbs
function animateGradientOrbs() {
    const orbs = document.querySelectorAll('.gradient-orb');
    
    orbs.forEach((orb, index) => {
        // Add random movement
        setInterval(() => {
            const randomX = Math.random() * 100 - 50;
            const randomY = Math.random() * 100 - 50;
            orb.style.transform = `translate(${randomX}px, ${randomY}px)`;
        }, 5000 + index * 1000);
    });
}

// Initialize gradient animations
animateGradientOrbs();

// Add hover effects to pipeline steps
document.querySelectorAll('.pipeline-step').forEach(step => {
    step.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px) scale(1.05)';
    });
    
    step.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// Progress indicator for implementation status
function updateProgressIndicators() {
    const progressSteps = [
        { name: 'Design', complete: true },
        { name: 'Schema', complete: false },
        { name: 'Module', complete: false },
        { name: 'Integration', complete: false },
        { name: 'Testing', complete: false }
    ];
    
    // This could be expanded to show actual progress
    console.log('Implementation Progress:', progressSteps);
}

// Initialize progress tracking
updateProgressIndicators();

// Add keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // Close any open modals or reset states
        const activeElements = document.querySelectorAll('.active');
        activeElements.forEach(el => {
            if (el.classList.contains('modal')) {
                el.classList.remove('active');
            }
        });
    }
    
    // Tab navigation between sections
    if (e.key === 'Tab' && !e.shiftKey) {
        const focusableElements = document.querySelectorAll(
            'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        // Handle tab navigation
    }
});

// Performance optimization - Debounce scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Apply debouncing to scroll handlers
window.addEventListener('scroll', debounce(() => {
    // Update navigation or other scroll-based features
}, 100));

// Initialize syntax highlighting if Prism.js is loaded
if (typeof Prism !== 'undefined') {
    Prism.highlightAll();
}

// Export functions for potential use in other scripts
window.speakerClustering = {
    initSmoothScrolling,
    initThemeToggle,
    initTabs,
    initCopyButtons,
    updateProgressIndicators
};