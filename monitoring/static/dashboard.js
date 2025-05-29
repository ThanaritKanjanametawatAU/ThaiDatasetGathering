// Audio Enhancement Dashboard JavaScript

// Global variables
let socket = null;
let metricsChart = null;
let speedChart = null;
let metricsHistory = [];
let speedHistory = [];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    initializeWebSocket();
    setupEventHandlers();
    
    // Start periodic updates if WebSocket is not available
    if (!window.io) {
        console.log('WebSocket not available, using polling');
        setInterval(fetchMetrics, 5000);
    }
});

// Initialize Charts
function initializeCharts() {
    // Metrics Chart
    const metricsCtx = document.getElementById('metrics-chart').getContext('2d');
    metricsChart = new Chart(metricsCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'SNR Improvement (dB)',
                data: [],
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                tension: 0.4,
                yAxisID: 'y1'
            }, {
                label: 'PESQ Score',
                data: [],
                borderColor: '#2196F3',
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                tension: 0.4,
                yAxisID: 'y2'
            }, {
                label: 'STOI Score',
                data: [],
                borderColor: '#FF9800',
                backgroundColor: 'rgba(255, 152, 0, 0.1)',
                tension: 0.4,
                yAxisID: 'y2'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'SNR Improvement (dB)'
                    }
                },
                y2: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Quality Score'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
    
    // Speed Chart
    const speedCtx = document.getElementById('speed-chart').getContext('2d');
    speedChart = new Chart(speedCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Files/Minute',
                data: [],
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Processing Speed'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Initialize WebSocket connection
function initializeWebSocket() {
    if (!window.io) return;
    
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to dashboard server');
        updateConnectionStatus(true);
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from dashboard server');
        updateConnectionStatus(false);
    });
    
    socket.on('update', function(data) {
        updateDashboard(data);
    });
}

// Update connection status indicator
function updateConnectionStatus(connected) {
    const statusDot = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');
    
    if (connected) {
        statusDot.classList.add('connected');
        statusText.textContent = 'Connected';
    } else {
        statusDot.classList.remove('connected');
        statusText.textContent = 'Disconnected';
    }
}

// Update dashboard with new data
function updateDashboard(data) {
    // Update progress
    updateProgress(data.processed, data.total, data.progress);
    
    // Update metrics
    updateMetrics(data.metrics);
    
    // Update processing rate
    updateProcessingRate(data.rate);
    
    // Update charts
    updateCharts(data);
    
    // Update GPU status if available
    if (data.gpu) {
        updateGPUStatus(data.gpu);
    }
}

// Update progress bar and stats
function updateProgress(processed, total, percent) {
    document.getElementById('processed-count').textContent = processed.toLocaleString();
    document.getElementById('total-count').textContent = total.toLocaleString();
    document.getElementById('progress-percent').textContent = percent.toFixed(1);
    document.getElementById('progress-fill').style.width = percent + '%';
}

// Update metric cards
function updateMetrics(metrics) {
    if (metrics.avg_snr_improvement !== undefined) {
        document.getElementById('snr-improvement').textContent = 
            '+' + metrics.avg_snr_improvement.toFixed(1);
    }
    
    if (metrics.avg_pesq !== undefined) {
        document.getElementById('pesq-score').textContent = 
            metrics.avg_pesq.toFixed(2);
    }
    
    if (metrics.avg_stoi !== undefined) {
        document.getElementById('stoi-score').textContent = 
            metrics.avg_stoi.toFixed(3);
    }
    
    if (metrics.failed_count !== undefined) {
        document.getElementById('failed-count').textContent = 
            metrics.failed_count.toLocaleString();
    }
    
    if (metrics.low_quality_count !== undefined) {
        document.getElementById('low-quality-count').textContent = 
            metrics.low_quality_count.toLocaleString();
    }
}

// Update processing rate
function updateProcessingRate(rate) {
    document.getElementById('processing-rate').textContent = 
        Math.round(rate).toLocaleString();
}

// Update charts with new data
function updateCharts(data) {
    const timestamp = new Date().toLocaleTimeString();
    
    // Add to history
    metricsHistory.push({
        time: timestamp,
        snr: data.metrics.avg_snr_improvement || 0,
        pesq: data.metrics.avg_pesq || 0,
        stoi: data.metrics.avg_stoi || 0
    });
    
    speedHistory.push({
        time: timestamp,
        rate: data.rate || 0
    });
    
    // Keep only last 50 points
    if (metricsHistory.length > 50) {
        metricsHistory.shift();
        speedHistory.shift();
    }
    
    // Update metrics chart
    metricsChart.data.labels = metricsHistory.map(m => m.time);
    metricsChart.data.datasets[0].data = metricsHistory.map(m => m.snr);
    metricsChart.data.datasets[1].data = metricsHistory.map(m => m.pesq);
    metricsChart.data.datasets[2].data = metricsHistory.map(m => m.stoi);
    metricsChart.update('none');
    
    // Update speed chart
    speedChart.data.labels = speedHistory.map(s => s.time);
    speedChart.data.datasets[0].data = speedHistory.map(s => s.rate);
    speedChart.update('none');
}

// Update GPU status
function updateGPUStatus(gpu) {
    if (gpu.memory_percent !== undefined) {
        document.getElementById('gpu-memory').textContent = 
            gpu.memory_percent.toFixed(1) + '%';
        document.getElementById('gpu-memory-fill').style.width = 
            gpu.memory_percent + '%';
    }
    
    if (gpu.utilization !== undefined) {
        document.getElementById('gpu-util').textContent = 
            gpu.utilization + '%';
        document.getElementById('gpu-util-fill').style.width = 
            gpu.utilization + '%';
    }
    
    if (gpu.temperature !== undefined) {
        document.getElementById('gpu-temp').textContent = 
            gpu.temperature + '°C';
    }
}

// Setup event handlers
function setupEventHandlers() {
    // Configuration toggle
    const configToggle = document.getElementById('config-toggle');
    const configPanel = document.getElementById('config-panel');
    
    configToggle.addEventListener('click', function() {
        if (configPanel.style.display === 'none') {
            configPanel.style.display = 'block';
            configToggle.textContent = 'Hide Configuration';
        } else {
            configPanel.style.display = 'none';
            configToggle.textContent = 'Show Configuration';
        }
    });
    
    // Configuration form
    const configForm = document.getElementById('config-form');
    configForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(configForm);
        const config = {};
        
        for (let [key, value] of formData.entries()) {
            config[key] = value;
        }
        
        // Send configuration update
        if (socket && socket.connected) {
            socket.emit('update_config', config);
        } else {
            // Use fetch API
            fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            }).then(response => {
                if (response.ok) {
                    alert('Configuration updated successfully');
                } else {
                    alert('Failed to update configuration');
                }
            });
        }
    });
}

// Fetch metrics via HTTP (fallback when WebSocket is not available)
function fetchMetrics() {
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            if (data.current) {
                updateDashboard({
                    processed: data.current.processed_count || 0,
                    total: data.current.total_count || 0,
                    progress: data.current.progress_percent || 0,
                    rate: data.current.processing_rate || 0,
                    metrics: data.current
                });
            }
        })
        .catch(error => {
            console.error('Failed to fetch metrics:', error);
        });
}

// Utility functions
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Export functions for external use
window.dashboardAPI = {
    updateDashboard: updateDashboard,
    updateMetrics: updateMetrics,
    updateProgress: updateProgress,
    updateGPUStatus: updateGPUStatus
};