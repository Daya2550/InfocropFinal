document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('weatherForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultContainer = document.getElementById('resultContainer');
    const apiResponse = document.getElementById('apiResponse');
    const loader = document.querySelector('.loader');
    const btnText = document.querySelector('.btn-text');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI State: Loading
        setLoading(true);
        resultContainer.classList.add('hidden');
        apiResponse.innerHTML = '';

        const formData = {
            city: document.getElementById('city').value,
            state: document.getElementById('state').value,
            crop: document.getElementById('crop').value,
            season: document.getElementById('season').value,
            extra_info: document.getElementById('extra_info').value
        };

        try {
            const response = await fetch('/api/get_analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (response.ok) {
                // Parse Markdown and display
                apiResponse.innerHTML = marked.parse(data.result);
                resultContainer.classList.remove('hidden');
                
                // Smooth scroll to result
                resultContainer.scrollIntoView({ behavior: 'smooth' });
            } else if (response.status === 429) {
                // Rate limit hit - show countdown
                apiResponse.innerHTML = `<div class="error-message">⚠️ ${data.error}</div>`;
                resultContainer.classList.remove('hidden');
                startCooldown(60);
            } else {
                alert('Error: ' + (data.error || 'Something went wrong'));
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the server. Please ensure the backend is running.');
        } finally {
            // UI State: Reset if not in cooldown
            if (!submitBtn.disabled || submitBtn.textContent === 'Analyzing...') {
                setLoading(false);
            }
        }
    });

    function startCooldown(seconds) {
        submitBtn.disabled = true;
        let remaining = seconds;
        
        const interval = setInterval(() => {
            btnText.textContent = `Wait ${remaining}s`;
            remaining--;
            
            if (remaining < 0) {
                clearInterval(interval);
                submitBtn.disabled = false;
                btnText.textContent = 'Get Advice';
            }
        }, 1000);
    }

    function setLoading(isLoading) {

        if (isLoading) {
            submitBtn.disabled = true;
            loader.style.display = 'block';
            btnText.textContent = 'Analyzing...';
        } else {
            submitBtn.disabled = false;
            loader.style.display = 'none';
            btnText.textContent = 'Get Advice';
        }
    }
});
