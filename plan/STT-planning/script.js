// Enhanced JavaScript for Modern UI

// Theme Management
class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        this.setTheme(this.theme);
        this.bindEvents();
    }

    setTheme(theme) {
        this.theme = theme;
        document.body.className = `${theme}-mode`;
        localStorage.setItem('theme', theme);
        
        // Update theme toggle icon
        const toggle = document.getElementById('theme-toggle');
        if (toggle) {
            toggle.setAttribute('aria-label', `Switch to ${theme === 'light' ? 'dark' : 'light'} mode`);
        }
    }

    toggleTheme() {
        this.setTheme(this.theme === 'light' ? 'dark' : 'light');
    }

    bindEvents() {
        const toggle = document.getElementById('theme-toggle');
        if (toggle) {
            toggle.addEventListener('click', () => this.toggleTheme());
        }
    }
}

// Navigation Management
class NavigationManager {
    constructor() {
        this.currentSection = 'overview';
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateActiveNav();
        this.setupSmoothScrolling();
        this.setupScrollSpy();
    }

    bindEvents() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.getAttribute('data-section');
                this.navigateToSection(section);
            });
        });
    }

    navigateToSection(section) {
        this.currentSection = section;
        this.updateActiveNav();
        
        const element = document.getElementById(section);
        if (element) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }

    updateActiveNav() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            const section = link.getAttribute('data-section');
            link.classList.toggle('active', section === this.currentSection);
        });
    }

    setupSmoothScrolling() {
        // Enhance smooth scrolling for all internal links
        document.querySelectorAll('a[href^="#"]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    setupScrollSpy() {
        const sections = document.querySelectorAll('.section');
        const options = {
            threshold: 0.3,
            rootMargin: '-100px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.currentSection = entry.target.id;
                    this.updateActiveNav();
                }
            });
        }, options);

        sections.forEach(section => observer.observe(section));
    }
}

// Tab Management
class TabManager {
    constructor() {
        this.init();
    }

    init() {
        this.bindEvents();
    }

    bindEvents() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.getAttribute('data-tab');
                this.showTab(tabName, button);
            });
        });
    }

    showTab(tabName, button) {
        // Hide all tab panels
        const panels = document.querySelectorAll('.tab-panel');
        panels.forEach(panel => panel.classList.remove('active'));
        
        // Remove active class from all buttons
        const buttons = document.querySelectorAll('.tab-btn');
        buttons.forEach(btn => btn.classList.remove('active'));
        
        // Show selected tab panel
        const targetPanel = document.getElementById(`${tabName}-tab`);
        if (targetPanel) {
            targetPanel.classList.add('active');
        }
        
        // Mark button as active
        if (button) {
            button.classList.add('active');
        }
    }
}

// Copy to Clipboard
class ClipboardManager {
    constructor() {
        this.init();
    }

    init() {
        this.bindEvents();
        this.addCopyHints();
    }

    bindEvents() {
        const copyButtons = document.querySelectorAll('.copy-btn');
        copyButtons.forEach(button => {
            button.addEventListener('click', () => {
                const codeId = button.getAttribute('data-copy');
                this.copyCode(codeId, button);
            });
        });

        // Add click-to-copy for code blocks
        const codeBlocks = document.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            block.addEventListener('click', () => {
                this.copyText(block.textContent, block);
            });
        });
    }

    copyCode(codeId, button) {
        const codeElement = document.getElementById(codeId);
        if (codeElement) {
            this.copyText(codeElement.textContent, button);
        }
    }

    async copyText(text, element) {
        try {
            await navigator.clipboard.writeText(text);
            this.showCopySuccess(element);
        } catch (err) {
            console.error('Failed to copy text:', err);
            this.showCopyError(element);
        }
    }

    showCopySuccess(element) {
        const tooltip = this.createTooltip('Copied!', 'success');
        this.showTooltip(tooltip, element);
    }

    showCopyError(element) {
        const tooltip = this.createTooltip('Copy failed', 'error');
        this.showTooltip(tooltip, element);
    }

    createTooltip(text, type) {
        const tooltip = document.createElement('div');
        tooltip.textContent = text;
        tooltip.className = `copy-tooltip ${type}`;
        tooltip.style.cssText = `
            position: absolute;
            background: ${type === 'success' ? '#10b981' : '#ef4444'};
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            z-index: 1000;
            transform: translateX(-50%);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            animation: fadeInOut 2s ease-in-out forwards;
        `;
        return tooltip;
    }

    showTooltip(tooltip, element) {
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + rect.width / 2 + 'px';
        tooltip.style.top = rect.bottom + 10 + 'px';
        
        document.body.appendChild(tooltip);
        
        setTimeout(() => {
            if (tooltip.parentNode) {
                tooltip.parentNode.removeChild(tooltip);
            }
        }, 2000);
    }

    addCopyHints() {
        const codeBlocks = document.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            block.style.cursor = 'pointer';
            block.title = 'Click to copy code';
        });
    }
}

// Animation Manager
class AnimationManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupScrollAnimations();
        this.setupHoverEffects();
        this.setupProgressAnimations();
    }

    setupScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        // Observe elements for animation
        const animateElements = document.querySelectorAll(`
            .feature-card, .opt-card, .strategy-card, .model-card,
            .feature-confirmed, .flow-step, .check-item
        `);
        
        animateElements.forEach(element => {
            element.style.opacity = '0';
            element.style.transform = 'translateY(30px)';
            element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(element);
        });
    }

    setupHoverEffects() {
        // Add ripple effect to buttons
        const buttons = document.querySelectorAll('button, .nav-link');
        buttons.forEach(button => {
            button.addEventListener('click', (e) => {
                this.createRipple(e, button);
            });
        });

        // Add floating animation to cards
        const floatingCards = document.querySelectorAll('.floating-card');
        floatingCards.forEach(card => {
            this.addFloatingAnimation(card);
        });
    }

    createRipple(event, element) {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
        `;
        
        element.style.position = 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);
        
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 600);
    }

    addFloatingAnimation(element) {
        let isHovered = false;
        
        element.addEventListener('mouseenter', () => {
            isHovered = true;
            this.startFloating(element);
        });
        
        element.addEventListener('mouseleave', () => {
            isHovered = false;
        });
    }

    startFloating(element) {
        let frame = 0;
        const animate = () => {
            if (!element.matches(':hover')) return;
            
            const offset = Math.sin(frame * 0.01) * 2;
            element.style.transform = `translateY(${offset}px)`;
            frame++;
            
            requestAnimationFrame(animate);
        };
        animate();
    }

    setupProgressAnimations() {
        // Animate progress indicators
        const progressElements = document.querySelectorAll('.indicator-ring, .status-dot');
        progressElements.forEach(element => {
            this.addSpinAnimation(element);
        });
    }

    addSpinAnimation(element) {
        if (element.classList.contains('indicator-ring')) {
            element.style.animation = 'spin 2s linear infinite';
        } else if (element.classList.contains('status-dot')) {
            element.style.animation = 'pulse 2s ease-in-out infinite';
        }
    }
}

// Performance Monitor
class PerformanceMonitor {
    constructor() {
        this.init();
    }

    init() {
        this.monitorScrollPerformance();
        this.optimizeImages();
    }

    monitorScrollPerformance() {
        let ticking = false;
        
        const updateScrollElements = () => {
            // Update scroll-dependent elements
            const scrollY = window.scrollY;
            const navbar = document.querySelector('.navbar');
            
            if (navbar) {
                if (scrollY > 100) {
                    navbar.style.backdropFilter = 'blur(20px)';
                    navbar.style.background = 'var(--glass-bg)';
                } else {
                    navbar.style.backdropFilter = 'blur(10px)';
                    navbar.style.background = 'rgba(255, 255, 255, 0.1)';
                }
            }
            
            ticking = false;
        };
        
        const onScroll = () => {
            if (!ticking) {
                requestAnimationFrame(updateScrollElements);
                ticking = true;
            }
        };
        
        window.addEventListener('scroll', onScroll, { passive: true });
    }

    optimizeImages() {
        // Lazy load images if any
        const images = document.querySelectorAll('img[data-src]');
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });
        
        images.forEach(img => imageObserver.observe(img));
    }
}

// Accessibility Manager
class AccessibilityManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupKeyboardNavigation();
        this.setupFocusManagement();
        this.setupAriaLabels();
    }

    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-nav');
            }
            
            if (e.key === 'Escape') {
                // Close any open modals or dropdowns
                this.closeAllOverlays();
            }
        });
        
        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-nav');
        });
    }

    setupFocusManagement() {
        // Ensure proper focus indicators
        const focusableElements = document.querySelectorAll(`
            button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])
        `);
        
        focusableElements.forEach(element => {
            element.addEventListener('focus', () => {
                element.style.outline = '2px solid var(--primary)';
                element.style.outlineOffset = '2px';
            });
            
            element.addEventListener('blur', () => {
                element.style.outline = 'none';
            });
        });
    }

    setupAriaLabels() {
        // Add missing aria labels
        const buttons = document.querySelectorAll('button:not([aria-label])');
        buttons.forEach(button => {
            const text = button.textContent.trim() || button.getAttribute('title') || 'Button';
            button.setAttribute('aria-label', text);
        });
    }

    closeAllOverlays() {
        // Close any open overlays or modals
        const overlays = document.querySelectorAll('.overlay, .modal');
        overlays.forEach(overlay => {
            overlay.classList.remove('active');
        });
    }
}

// Enhanced Error Handling
class ErrorHandler {
    constructor() {
        this.init();
    }

    init() {
        window.addEventListener('error', (e) => {
            console.error('JavaScript error:', e.error);
            this.showErrorMessage('An unexpected error occurred. Please refresh the page.');
        });
        
        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e.reason);
            this.showErrorMessage('Failed to load some content. Please try again.');
        });
    }

    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-toast';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ef4444;
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                if (errorDiv.parentNode) {
                    errorDiv.parentNode.removeChild(errorDiv);
                }
            }, 300);
        }, 5000);
    }
}

// Initialize all managers when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize all managers
    const themeManager = new ThemeManager();
    const navigationManager = new NavigationManager();
    const tabManager = new TabManager();
    const clipboardManager = new ClipboardManager();
    const animationManager = new AnimationManager();
    const performanceMonitor = new PerformanceMonitor();
    const accessibilityManager = new AccessibilityManager();
    const errorHandler = new ErrorHandler();
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
        
        @keyframes fadeInOut {
            0% { opacity: 0; transform: translateY(-10px); }
            20%, 80% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(-10px); }
        }
        
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        
        .keyboard-nav *:focus {
            outline: 2px solid var(--primary) !important;
            outline-offset: 2px !important;
        }
    `;
    document.head.appendChild(style);
    
    // Add loading animation
    const loader = document.querySelector('.loader');
    if (loader) {
        setTimeout(() => {
            loader.style.opacity = '0';
            setTimeout(() => {
                if (loader.parentNode) {
                    loader.parentNode.removeChild(loader);
                }
            }, 300);
        }, 1000);
    }
    
    console.log('🚀 Thai STT Implementation Plan - UI Loaded Successfully');
});