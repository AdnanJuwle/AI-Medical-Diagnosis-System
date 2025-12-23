// BMI Calculator functionality
document.getElementById('calculate-bmi').addEventListener('click', function() {
    const height = parseFloat(document.getElementById('height').value);
    const weight = parseFloat(document.getElementById('weight').value);
    
    if (height && weight && height > 0 && weight > 0) {
        const bmi = weight / Math.pow(height / 100, 2);
        document.getElementById('bmi').value = bmi.toFixed(1);
        
        // Show BMI category with better styling
        let category = '';
        let categoryClass = '';
        if (bmi < 18.5) {
            category = 'Underweight';
            categoryClass = 'underweight';
        } else if (bmi < 25) {
            category = 'Normal weight';
            categoryClass = 'normal';
        } else if (bmi < 30) {
            category = 'Overweight';
            categoryClass = 'overweight';
        } else {
            category = 'Obese';
            categoryClass = 'obese';
        }
        
        // Show result in results card
        const resultDiv = document.getElementById('result');
        const resultsCard = document.getElementById('prediction-result');
        resultsCard.style.display = 'block';
        resultDiv.innerHTML = `<div class="bmi-result ${categoryClass}">📊 BMI: ${bmi.toFixed(1)} (${category})</div>`;
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
        alert('Please enter valid height and weight values first.');
    }
});

// Form submission
document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Get submit button and disable it
    const submitBtn = event.target.querySelector('.submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const originalText = btnText.textContent;
    
    submitBtn.disabled = true;
    btnText.textContent = 'Analyzing...';

    // Show loading state
    const resultDiv = document.getElementById('result');
    const resultsCard = document.getElementById('prediction-result');
    resultsCard.style.display = 'block';
    resultDiv.innerHTML = '<div class="loading">Processing your health data</div>';

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Collect form data
    let formData = {
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        height: parseFloat(document.getElementById('height').value),
        weight: parseFloat(document.getElementById('weight').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        exercise: document.getElementById('exercise').value,
        checkup: document.getElementById('checkup').value,
        smoking_history: document.getElementById('smoking_history').value,
        alcohol_consumption: document.getElementById('alcohol_consumption').value,
        fruit_consumption: parseFloat(document.getElementById('fruit_consumption').value),
        green_vegetables_consumption: parseFloat(document.getElementById('green_vegetables_consumption').value),
        fried_food_consumption: parseFloat(document.getElementById('fried_food_consumption').value)
    };

    console.log('Sending data:', formData);

    // Send request to backend
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Received response:', data);
        
        // Re-enable submit button
        submitBtn.disabled = false;
        btnText.textContent = originalText;
        
        if (data.error) {
            resultDiv.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
            return;
        }

        // Build results HTML
        let resultHTML = '<div class="results-container"><div class="diseases-grid">';
        
        const diseaseNames = {
            'Heart_Disease': 'Heart Disease',
            'Skin_Cancer': 'Skin Cancer',
            'Other_Cancer': 'Other Cancer',
            'Depression': 'Depression',
            'Diabetes': 'Diabetes',
            'Arthritis': 'Arthritis'
        };

        // Count high risk diseases
        let highRiskCount = 0;
        for (let disease in data) {
            if (data[disease] === 1) highRiskCount++;
        }

        // Add summary if there are high risk diseases
        if (highRiskCount > 0) {
            resultHTML += `<div class="risk-summary high-risk-summary">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                <div>
                    <strong>${highRiskCount} High Risk Condition${highRiskCount > 1 ? 's' : ''} Detected</strong>
                    <p>Please consult with a healthcare professional</p>
                </div>
            </div>`;
        } else {
            resultHTML += `<div class="risk-summary low-risk-summary">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
                <div>
                    <strong>Low Risk Profile</strong>
                    <p>Continue maintaining a healthy lifestyle</p>
                </div>
            </div>`;
        }

        // Add disease items
        for (let disease in data) {
            const riskLevel = data[disease] ? 'High Risk' : 'Low Risk';
            const riskClass = data[disease] ? 'high-risk' : 'low-risk';
            const diseaseName = diseaseNames[disease] || disease;
            
            resultHTML += `
                <div class="disease-item ${riskClass}">
                    <span class="disease-name">${diseaseName}</span>
                    <span class="risk-level">${riskLevel}</span>
                </div>
            `;
        }
        
        resultHTML += '</div><p class="disclaimer">⚠️ This is a predictive model for educational purposes. Please consult with healthcare professionals for medical advice.</p></div>';
        
        resultDiv.innerHTML = resultHTML;
    })
    .catch(error => {
        console.error("Error:", error);
        
        // Re-enable submit button
        submitBtn.disabled = false;
        btnText.textContent = originalText;
        
        resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}. Please make sure the backend server is running on http://127.0.0.1:5000</div>`;
    });
});

// Add some interactive enhancements
document.addEventListener('DOMContentLoaded', function() {
    // Add focus animations to inputs
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });
});
