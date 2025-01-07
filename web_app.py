from flask import Flask, render_template_string
import json
from datetime import datetime
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Kraken Trading Bot Status</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f5;
        }
        .status-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .header {
            color: #2c3e50;
            text-align: center;
        }
        .status-item {
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1 class="header">🤖 Kraken Trading Bot Status</h1>
    
    <div class="status-box">
        <h2>📊 Latest Status</h2>
        <div class="status-item">
            <p><strong>Last Check:</strong> {{ last_check }}</p>
        </div>
        
        {% if backtest_summary %}
        <h2>📈 Latest Backtest Results</h2>
        <div class="status-item">
            {% for key, value in backtest_summary.items() %}
            <p><strong>{{ key }}:</strong> {{ value }}</p>
            {% endfor %}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    # Get the latest backtest summary if it exists
    backtest_summary = None
    try:
        if os.path.exists('backtesting_results/backtest_summary.json'):
            with open('backtesting_results/backtest_summary.json', 'r') as f:
                backtest_summary = json.load(f)
    except Exception as e:
        backtest_summary = {"error": str(e)}

    return render_template_string(
        HTML_TEMPLATE,
        last_check=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        backtest_summary=backtest_summary
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 