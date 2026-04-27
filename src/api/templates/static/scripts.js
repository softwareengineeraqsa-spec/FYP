const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const fileWrapper = document.getElementById('file-upload-wrapper');
const fileName = document.getElementById('file-name');
const resultsDiv = document.getElementById('results');
const loading = document.getElementById('loading');
const submitBtn = document.getElementById('submit-btn');

let barChart, pieChart;

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileName.textContent = `📄 ${e.target.files[0].name}`;
        fileName.style.display = 'block';
    }
});

fileWrapper.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileWrapper.classList.add('dragover');
});

fileWrapper.addEventListener('dragleave', () => {
    fileWrapper.classList.remove('dragover');
});

fileWrapper.addEventListener('drop', (e) => {
    e.preventDefault();
    fileWrapper.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileName.textContent = `📄 ${e.dataTransfer.files[0].name}`;
        fileName.style.display = 'block';
    }
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!fileInput.files.length) {
        alert('Please select a file to analyze');
        return;
    }

    const formData = new FormData(form);

    submitBtn.disabled = true;
    loading.style.display = 'block';
    resultsDiv.style.display = 'none';

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        loading.style.display = 'none';
        resultsDiv.style.display = 'block';

        const totalPredictions = data.decoded_predictions.length;
        const distribution = data.distribution_decoded || {};
        const normalCount = distribution["Benign"] || 0;
        const threatCount = distribution["DDoS"] || 0;
        const threatDetected = threatCount > 0;

        const threatPercentage = ((threatCount / totalPredictions) * 100).toFixed(1);

        const threatLabel = threatDetected ? "Threats Detected!" : "System Secure";
        const threatIcon = threatDetected ? "⚠️" : "✅";
        const resultClass = threatDetected ? "threat" : "safe";

        resultsDiv.innerHTML = `
                    <div class="result-hero ${resultClass}">
                        <div class="result-icon-large ${resultClass}">${threatIcon}</div>
                        <div class="result-title-large">${threatLabel}</div>
                        <p class="result-subtitle-large">
                            ${threatDetected
                ? `Found ${threatCount.toLocaleString()} potential threat${threatCount !== 1 ? 's' : ''} in your log file`
                : 'No malicious activity detected in your log file'}
                        </p>
                    </div>

                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-icon">📊</div>
                            <div class="metric-value">${totalPredictions.toLocaleString()}</div>
                            <div class="metric-label">Total Records</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">⚡</div>
                            <div class="metric-value">${data.prediction_time.toFixed(2)}s</div>
                            <div class="metric-label">Processing Time</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">🛡️</div>
                            <div class="metric-value safe">${normalCount.toLocaleString()}</div>
                            <div class="metric-label">Safe Records</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">🚨</div>
                            <div class="metric-value threat">${threatCount.toLocaleString()}</div>
                            <div class="metric-label">Threat Records</div>
                        </div>

                        <div class="metric-card full-width">
                            <div class="metric-icon">📈</div>
                            <div class="metric-value">${threatPercentage}%</div>
                            <div class="metric-label">Threat Percentage</div>
                        </div>
                        
                    </div>

                    <div class="charts-container">
                        <div class="charts-title">📈 Detection Analysis Visualization</div>
                        <div class="charts-grid">
                            <div class="chart-wrapper">
                                <canvas id="barChart"></canvas>
                            </div>
                            <div class="chart-wrapper">
                                <canvas id="pieChart"></canvas>
                            </div>
                        </div>
                    </div>
                `;

        if (barChart) barChart.destroy();
        if (pieChart) pieChart.destroy();

        const labels = ['Benign', 'DDoS'];
        const values = [normalCount, threatCount];
        const colors = ['rgba(34, 197, 94, 0.8)', 'rgba(239, 68, 68, 0.8)'];
        const borderColors = ['rgba(34, 197, 94, 1)', 'rgba(239, 68, 68, 1)'];

        const barCtx = document.getElementById('barChart').getContext('2d');
        barChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Record Count',
                    data: values,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Detection Distribution',
                        color: '#cbd5e1',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(148, 163, 184, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { display: false }
                    }
                }
            }
        });

        const pieCtx = document.getElementById('pieChart').getContext('2d');
        pieChart = new Chart(pieCtx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#cbd5e1',
                            padding: 15,
                            font: { size: 12 }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Security Status Overview',
                        color: '#cbd5e1',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                }
            }
        });

    } catch (err) {
        console.error(err);
        loading.style.display = 'none';
        resultsDiv.style.display = 'block';
        resultsDiv.innerHTML = `
                    <div class="result-hero threat">
                        <div class="result-icon-large threat">❌</div>
                        <div class="result-title-large">Connection Error</div>
                        <p class="result-subtitle-large">
                            Unable to connect to the analysis server. Please ensure the backend is running on 
                            <strong>http://127.0.0.1:8000</strong>
                        </p>
                    </div>
                `;
    } finally {
        submitBtn.disabled = false;
    }
});