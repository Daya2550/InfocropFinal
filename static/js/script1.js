// Crop Recommendation System - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Current prediction storage for downloading PDF
    let lastPredictionData = null;
    let lastUserInputs = null;

    // DOM Elements
    const form = document.getElementById('cropForm');
    const submitBtn = document.getElementById('submitBtn');
    const resetBtn = document.getElementById('resetBtn');
    const resultCard = document.getElementById('resultCard');
    const errorCard = document.getElementById('errorCard');
    const cropName = document.getElementById('cropName');
    const confidenceValue = document.getElementById('confidenceValue');
    const errorMessage = document.getElementById('errorMessage');
    const newPredictionBtn = document.getElementById('newPredictionBtn');
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    const closeErrorBtn = document.getElementById('closeErrorBtn');

    // Form Submit Handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results/errors
        hideResults();
        
        // Show loading state
        submitBtn.classList.add('btn-loading');
        submitBtn.disabled = true;
        
        // Collect form data
        const formData = {
            nitrogen: parseFloat(document.getElementById('nitrogen').value),
            phosphorus: parseFloat(document.getElementById('phosphorus').value),
            potassium: parseFloat(document.getElementById('potassium').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            ph: parseFloat(document.getElementById('ph').value),
            rainfall: parseFloat(document.getElementById('rainfall').value)
        };
        
        try {
            // Make API request
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Show success result with ensemble details
                showResult(data);
            } else {
                // Show error
                showError(data.error);
            }
        } catch (error) {
            // Check if it's a JSON parsing error (often means server returned 500 HTML)
            if (error instanceof SyntaxError) {
                showError('Server error: Received invalid response. Please try again later.');
            } else {
                showError('An unexpected error occurred. Please check your connection and try again.');
            }
            console.error('Detailed Prediction Error:', error);
        } finally {
            // Remove loading state
            submitBtn.classList.remove('btn-loading');
            submitBtn.disabled = false;
        }
    });

    // Reset Button Handler
    resetBtn.addEventListener('click', function() {
        form.reset();
        hideResults();
        
        // Add reset animation
        form.style.animation = 'none';
        setTimeout(() => {
            form.style.animation = 'fadeInUp 0.5s ease-out';
        }, 10);
    });

    // New Prediction Button Handler
    newPredictionBtn.addEventListener('click', function() {
        hideResults();
        form.reset();
        
        // Scroll to form
        form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // Close Error Button Handler
    closeErrorBtn.addEventListener('click', function() {
        hideResults();
    });

    // Show Result Function
    function showResult(data) {
        // Store data for PDF download
        lastPredictionData = data;
        lastUserInputs = {
            nitrogen: parseFloat(document.getElementById('nitrogen').value),
            phosphorus: parseFloat(document.getElementById('phosphorus').value),
            potassium: parseFloat(document.getElementById('potassium').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            ph: parseFloat(document.getElementById('ph').value),
            rainfall: parseFloat(document.getElementById('rainfall').value)
        };

        // Helper to safely set text content
        const setText = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };

        // Helper to safely set display
        const setDisplay = (id, display) => {
            const el = document.getElementById(id);
            if (el) el.style.display = display;
        };

        resultCard.style.display = 'block';
        errorCard.style.display = 'none';
        
        // Main recommendation
        setText('cropName', data.crop.charAt(0).toUpperCase() + data.crop.slice(1));
        setText('confidenceValue', data.confidence + '%');
        
        // Agreement information
        setText('agreementValue', data.agreement + '%');
        setText('voteCount', '(' + data.vote_count + ')');
        
        // Update icon based on unanimity
        const iconEl = document.getElementById('resultIcon');
        if (data.is_tie_breaker) {
            if (iconEl) iconEl.textContent = '⚖️';
            setDisplay('tieBreakerBadge', 'inline-block');
            setDisplay('agreementBadge', 'none');
        } else {
            setDisplay('tieBreakerBadge', 'none');
            setDisplay('agreementBadge', 'inline-block');
            if (iconEl) {
                if (data.unanimous) {
                    iconEl.textContent = '🏆'; // Trophy for unanimous decision
                } else {
                    iconEl.textContent = '✨'; // Sparkle for majority decision
                }
            }
        }
        
        // Display individual model predictions
        const modelsGrid = document.getElementById('modelsGrid');
        modelsGrid.innerHTML = '';
        
        if (data.individual_predictions) {
            for (const [modelName, prediction] of Object.entries(data.individual_predictions)) {
                const modelCard = document.createElement('div');
                modelCard.className = 'model-card';
                
                // Mark if this model agrees with the final decision
                if (data.unanimous || prediction.crop === data.crop) {
                    modelCard.classList.add('unanimous');
                }
                
                modelCard.innerHTML = `
                    <div class="model-name">${modelName.replace(/_/g, ' ')}</div>
                    <div class="model-prediction">${prediction.crop}</div>
                    <div class="model-confidence">${prediction.confidence}% confident</div>
                `;
                
                modelsGrid.appendChild(modelCard);
            }
        }
        
        // Display warnings if any
        const warningsContainer = document.getElementById('warningsContainer');
        const warningsList = document.getElementById('warningsList');
        
        if (data.warnings && data.warnings.length > 0) {
            if (warningsContainer) warningsContainer.style.display = 'block';
            if (warningsList) {
                warningsList.innerHTML = '';
                data.warnings.forEach(warning => {
                    const li = document.createElement('li');
                    li.textContent = warning;
                    warningsList.appendChild(li);
                });
            }
        } else {
            if (warningsContainer) warningsContainer.style.display = 'none';
        }
        
        // Display reasoning/explanation
        const reasoningList = document.getElementById('reasoningList');
        reasoningList.innerHTML = '';
        
        if (data.reasoning && data.reasoning.length > 0) {
            data.reasoning.forEach(reason => {
                const reasonDiv = document.createElement('div');
                reasonDiv.className = 'reasoning-item';
                reasonDiv.textContent = reason;
                reasoningList.appendChild(reasonDiv);
            });
        }
        
        // Display top influencing factors
        const factorsGrid = document.getElementById('factorsGrid');
        factorsGrid.innerHTML = '';
        
        if (data.top_influencing_factors && data.top_influencing_factors.length > 0) {
            data.top_influencing_factors.forEach(factor => {
                const factorCard = document.createElement('div');
                factorCard.className = 'factor-card';
                factorCard.innerHTML = `
                    <div class="factor-name">${factor.factor}</div>
                    <div class="factor-importance">${factor.importance}%</div>
                    <span class="importance-label">Importance</span>
                `;
                factorsGrid.appendChild(factorCard);
            });
        }
        
        // Display crop requirements
        if (data.crop_info) {
            const cropInfo = data.crop_info;
            
            // Description
            if (cropInfo.description) {
                document.getElementById('cropDescription').textContent = cropInfo.description;
            }
            
            // Growing conditions
            const conditionsList = document.getElementById('conditionsList');
            conditionsList.innerHTML = '';
            
            if (cropInfo.growing_conditions) {
                cropInfo.growing_conditions.forEach(condition => {
                    const li = document.createElement('li');
                    li.textContent = condition;
                    conditionsList.appendChild(li);
                });
            }
            
            // Metadata
            setText('cropSeason', cropInfo.season || 'N/A');
            setText('cropWater', cropInfo.water_requirement || 'N/A');
            setText('cropClimate', cropInfo.climate || 'N/A');
        }
        
        // Add animation
        resultCard.style.animation = 'none';
        setTimeout(() => {
            resultCard.style.animation = 'fadeInUp 0.6s ease-out';
        }, 10);
        
        // Scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Celebrate with confetti effect
        celebrateResult();
    }

    // Download PDF Report Handler
    downloadReportBtn.addEventListener('click', async function() {
        if (!lastPredictionData || !lastUserInputs) return;
        
        const originalBtnText = downloadReportBtn.innerHTML;
        downloadReportBtn.innerHTML = '⌛ Generating...';
        downloadReportBtn.disabled = true;
        
        try {
            const response = await fetch('/api/download_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prediction_data: lastPredictionData,
                    user_inputs: lastUserInputs
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                
                // Extract filename from header or use default
                const contentDisposition = response.headers.get('Content-Disposition');
                let fileName = `crop_report_${lastPredictionData.crop.toLowerCase()}.pdf`;
                if (contentDisposition && contentDisposition.indexOf('filename=') !== -1) {
                    fileName = contentDisposition.split('filename=')[1].replace(/"/g, '');
                }
                
                a.download = fileName;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            } else {
                alert('Failed to generate report. Please try again.');
            }
        } catch (error) {
            console.error('Error downloading report:', error);
            alert('An error occurred while generating the report.');
        } finally {
            downloadReportBtn.innerHTML = originalBtnText;
            downloadReportBtn.disabled = false;
        }
    });

    // Show Error Function
    function showError(message) {
        errorCard.style.display = 'block';
        resultCard.style.display = 'none';
        
        errorMessage.textContent = message;
        
        // Add animation
        errorCard.style.animation = 'none';
        setTimeout(() => {
            errorCard.style.animation = 'fadeInUp 0.6s ease-out';
        }, 10);
        
        // Scroll to error
        errorCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Hide Results Function
    function hideResults() {
        resultCard.style.display = 'none';
        errorCard.style.display = 'none';
    }

    // Celebration Effect (simple version)
    function celebrateResult() {
        // Create a simple pulse effect on the result icon
        const icon = document.getElementById('resultIcon');
        icon.style.animation = 'none';
        setTimeout(() => {
            icon.style.animation = 'scaleIn 0.5s ease-out, float 3s ease-in-out infinite 0.5s';
        }, 10);
    }

    // Input Validation - Real-time feedback
    const inputs = form.querySelectorAll('.input-field');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });
        
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });

    function validateInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        if (input.value && (isNaN(value) || value < min || value > max)) {
            input.style.borderColor = '#ef4444';
        } else {
            input.style.borderColor = '';
        }
    }

    // Sample Data Loader (for demo purposes)
    const sampleData = {
        rice: { nitrogen: 90, phosphorus: 42, potassium: 43, temperature: 20.8, humidity: 82, ph: 6.5, rainfall: 202.9 },
        maize: { nitrogen: 80, phosphorus: 43, potassium: 16, temperature: 23.5, humidity: 71.5, ph: 6.6, rainfall: 66.7 },
        chickpea: { nitrogen: 40, phosphorus: 72, potassium: 77, temperature: 17, humidity: 17, ph: 7.5, rainfall: 88.5 }
    };

    // Add sample data buttons (commented out - can be enabled if needed)
    /*
    const buttonContainer = document.querySelector('.button-container');
    Object.keys(sampleData).forEach(crop => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-secondary btn-small';
        btn.textContent = `Load ${crop} example`;
        btn.addEventListener('click', () => loadSampleData(crop));
        buttonContainer.appendChild(btn);
    });
    */

    function loadSampleData(crop) {
        const data = sampleData[crop];
        Object.keys(data).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = data[key];
                // Trigger animation
                input.style.animation = 'none';
                setTimeout(() => {
                    input.style.animation = 'fadeInUp 0.3s ease-out';
                }, 10);
            }
        });
    }

    // Keyboard Shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
        
        // Escape to close results
        if (e.key === 'Escape') {
            hideResults();
        }
    });

    // Add smooth number input increment/decrement
    inputs.forEach(input => {
        input.addEventListener('wheel', function(e) {
            if (document.activeElement === this) {
                e.preventDefault();
                const step = parseFloat(this.step) || 1;
                const currentValue = parseFloat(this.value) || 0;
                const newValue = e.deltaY < 0 ? currentValue + step : currentValue - step;
                
                // Respect min/max values
                const min = parseFloat(this.min);
                const max = parseFloat(this.max);
                this.value = Math.min(Math.max(newValue, min), max);
                
                validateInput(this);
            }
        });
    });

    console.log('Crop Recommendation System initialized successfully! 🌾');
});
