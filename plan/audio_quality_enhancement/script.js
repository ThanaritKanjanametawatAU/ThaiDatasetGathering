// Waveform Animation
function animateWaveform() {
    const canvas = document.getElementById('waveform');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    let time = 0;
    
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;
        
        ctx.beginPath();
        for (let x = 0; x < canvas.width; x += 5) {
            const y = canvas.height / 2 + Math.sin((x + time) * 0.02) * 30 * Math.sin(time * 0.01);
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        time += 2;
        requestAnimationFrame(draw);
    }
    
    draw();
}

// Audio Visualizer Animation
function animateAudioVisualizer(canvasId, isNoisy) {
    const container = document.getElementById(canvasId);
    if (!container) return;
    
    const canvas = container.querySelector('canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
    
    const bars = 50;
    const barWidth = canvas.width / bars;
    
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        for (let i = 0; i < bars; i++) {
            const height = Math.random() * canvas.height * (isNoisy ? 0.8 : 0.4);
            const x = i * barWidth;
            const y = canvas.height - height;
            
            const gradient = ctx.createLinearGradient(0, y, 0, canvas.height);
            if (isNoisy) {
                gradient.addColorStop(0, 'rgba(239, 68, 68, 0.8)');
                gradient.addColorStop(1, 'rgba(239, 68, 68, 0.2)');
            } else {
                gradient.addColorStop(0, 'rgba(16, 185, 129, 0.8)');
                gradient.addColorStop(1, 'rgba(16, 185, 129, 0.2)');
            }
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidth - 2, height);
        }
        
        setTimeout(() => requestAnimationFrame(draw), 100);
    }
    
    draw();
}

// Chart Drawing
function drawChart(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    const padding = 10;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;
    
    // Draw background grid
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 4; i++) {
        const y = padding + (height / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();
    }
    
    // Draw data line
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    ctx.beginPath();
    data.forEach((value, index) => {
        const x = padding + (width / (data.length - 1)) * index;
        const y = padding + height - (value * height);
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
    
    // Draw data points
    ctx.fillStyle = color;
    data.forEach((value, index) => {
        const x = padding + (width / (data.length - 1)) * index;
        const y = padding + height - (value * height);
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
    });
}

// Copy Code Function
function copyCode(button) {
    const codeBlock = button.closest('.code-block');
    const code = codeBlock.querySelector('code').textContent;
    
    navigator.clipboard.writeText(code).then(() => {
        const icon = button.querySelector('i');
        icon.className = 'fas fa-check';
        button.style.color = '#10b981';
        
        setTimeout(() => {
            icon.className = 'fas fa-copy';
            button.style.color = '';
        }, 2000);
    });
}

// Smooth Scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Intersection Observer for Animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
            
            // Initialize charts when they come into view
            if (entry.target.classList.contains('metric-card')) {
                const chartId = entry.target.querySelector('canvas')?.id;
                if (chartId) {
                    initializeChart(chartId);
                }
            }
        }
    });
}, observerOptions);

// Observe elements
document.querySelectorAll('.overview-card, .feature-item, .timeline-item, .metric-card').forEach(el => {
    observer.observe(el);
});

// Initialize Charts
function initializeChart(chartId) {
    const chartData = {
        'snr-chart': {
            data: [0.3, 0.4, 0.5, 0.7, 0.85, 0.9, 0.95],
            color: '#2563eb'
        },
        'pesq-chart': {
            data: [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88],
            color: '#7c3aed'
        },
        'stoi-chart': {
            data: [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92],
            color: '#10b981'
        },
        'dnsmos-chart': {
            data: [0.6, 0.65, 0.75, 0.8, 0.85, 0.88, 0.9],
            color: '#f59e0b'
        }
    };
    
    const chart = chartData[chartId];
    if (chart) {
        drawChart(chartId, chart.data, chart.color);
    }
}

// Progressive Enhancement Animation
function animateProgressiveEnhancement() {
    const stages = document.querySelectorAll('.enhancement-stages .stage');
    let currentStage = 0;
    
    setInterval(() => {
        stages.forEach((stage, index) => {
            if (index <= currentStage) {
                stage.classList.add('active');
            } else {
                stage.classList.remove('active');
            }
        });
        
        currentStage = (currentStage + 1) % stages.length;
    }, 2000);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .enhancement-stages .stage {
        opacity: 0.5;
        transition: all 0.3s ease;
    }
    
    .enhancement-stages .stage.active {
        opacity: 1;
        transform: scale(1.05);
    }
`;
document.head.appendChild(style);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    animateWaveform();
    animateAudioVisualizer('before-viz', true);
    animateAudioVisualizer('after-viz', false);
    animateProgressiveEnhancement();
});

// Responsive Menu Toggle
const menuToggle = document.createElement('button');
menuToggle.className = 'menu-toggle';
menuToggle.innerHTML = '<i class="fas fa-bars"></i>';
menuToggle.style.cssText = `
    display: none;
    background: transparent;
    border: none;
    font-size: 1.5rem;
    color: var(--primary-color);
    cursor: pointer;
    padding: 0.5rem;
`;

const navbar = document.querySelector('.navbar');
navbar.appendChild(menuToggle);

menuToggle.addEventListener('click', () => {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('mobile-active');
});

// Mobile menu styles
const mobileStyle = document.createElement('style');
mobileStyle.textContent = `
    @media (max-width: 768px) {
        .menu-toggle {
            display: block !important;
        }
        
        .nav-links {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            flex-direction: column;
            padding: 1rem;
            box-shadow: var(--shadow-lg);
            transform: translateY(-100%);
            opacity: 0;
            pointer-events: none;
            transition: all 0.3s ease;
        }
        
        .nav-links.mobile-active {
            transform: translateY(0);
            opacity: 1;
            pointer-events: auto;
        }
    }
`;
document.head.appendChild(mobileStyle);