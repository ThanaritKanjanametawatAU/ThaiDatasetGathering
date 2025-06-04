// Tab functionality
function showTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tabs
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked tab
    event.target.classList.add('active');
}

// Copy code functionality
function copyCode(button) {
    const codeBlock = button.parentElement.querySelector('code');
    const text = codeBlock.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        button.classList.add('copied');
        button.innerHTML = '<i class="fas fa-check"></i> Copied!';
        
        setTimeout(() => {
            button.classList.remove('copied');
            button.innerHTML = '<i class="fas fa-copy"></i> Copy';
        }, 2000);
    });
}

// Smooth scrolling for navigation links
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        const targetSection = document.querySelector(targetId);
        
        if (targetSection) {
            targetSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Theme toggle functionality
const themeToggle = document.querySelector('.theme-toggle');
let isDarkMode = true;

themeToggle.addEventListener('click', () => {
    isDarkMode = !isDarkMode;
    
    if (isDarkMode) {
        document.body.style.background = 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)';
        themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    } else {
        document.body.style.background = 'linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 50%, #e0e7ff 100%)';
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.animationDelay = '0s';
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all sections
document.querySelectorAll('.section').forEach((section, index) => {
    section.style.animationDelay = `${index * 0.1}s`;
    observer.observe(section);
});

// Add tooltips to diagram nodes
document.querySelectorAll('.diagram-node').forEach(node => {
    const tooltip = node.getAttribute('data-tooltip');
    if (tooltip) {
        node.addEventListener('mouseenter', (e) => {
            const tooltipEl = document.createElement('div');
            tooltipEl.className = 'tooltip';
            tooltipEl.textContent = tooltip;
            tooltipEl.style.cssText = `
                position: absolute;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 0.875rem;
                white-space: nowrap;
                z-index: 1000;
                pointer-events: none;
                transform: translateX(-50%) translateY(-100%);
                left: ${e.target.offsetLeft + e.target.offsetWidth / 2}px;
                top: ${e.target.offsetTop - 10}px;
            `;
            document.body.appendChild(tooltipEl);
            
            node.addEventListener('mouseleave', () => {
                tooltipEl.remove();
            });
        });
    }
});

// Progress bar animations on scroll
const progressBars = document.querySelectorAll('.progress-fill');
const progressObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const width = entry.target.style.width;
            entry.target.style.width = '0%';
            setTimeout(() => {
                entry.target.style.width = width;
            }, 100);
            progressObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

progressBars.forEach(bar => {
    progressObserver.observe(bar);
});

// Add keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // Close any open modals or reset view
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});

// Add loading animation removal
window.addEventListener('load', () => {
    document.body.style.opacity = '1';
});

// Interactive timeline
document.querySelectorAll('.timeline-content').forEach((item, index) => {
    item.addEventListener('click', () => {
        item.style.transform = 'scale(1.05)';
        item.style.boxShadow = '0 10px 30px rgba(99, 102, 241, 0.3)';
        
        setTimeout(() => {
            item.style.transform = 'scale(1.02)';
            item.style.boxShadow = 'none';
        }, 300);
    });
});

// Status card counter animation
function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16); // 60 FPS
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
            current = end;
            clearInterval(timer);
        }
        
        if (element.textContent.includes('%')) {
            element.textContent = Math.floor(current) + '%+';
        } else if (element.textContent.includes('.')) {
            element.textContent = current.toFixed(2);
        } else if (element.textContent.includes('dB')) {
            element.textContent = '>' + Math.floor(current) + ' dB';
        }
    }, 16);
}

// Animate status values on page load
window.addEventListener('load', () => {
    const statusValues = document.querySelectorAll('.status-value');
    statusValues.forEach(value => {
        const text = value.textContent;
        if (text.includes('%')) {
            animateValue(value, 0, 95, 1000);
        } else if (text === '0.95') {
            animateValue(value, 0, 0.95, 1000);
        } else if (text.includes('dB')) {
            animateValue(value, 0, 10, 1000);
        }
    });
});

// Add particle interaction
document.addEventListener('mousemove', (e) => {
    const particles = document.querySelectorAll('.particle');
    const mouseX = e.clientX;
    const mouseY = e.clientY;
    
    particles.forEach((particle, index) => {
        const rect = particle.getBoundingClientRect();
        const particleX = rect.left + rect.width / 2;
        const particleY = rect.top + rect.height / 2;
        
        const distance = Math.sqrt(
            Math.pow(mouseX - particleX, 2) + 
            Math.pow(mouseY - particleY, 2)
        );
        
        if (distance < 100) {
            const angle = Math.atan2(particleY - mouseY, particleX - mouseX);
            const push = (100 - distance) / 10;
            
            particle.style.transform = `
                translate(${Math.cos(angle) * push}px, ${Math.sin(angle) * push}px)
                rotate(${angle * 180 / Math.PI}deg)
            `;
        } else {
            particle.style.transform = '';
        }
    });
});