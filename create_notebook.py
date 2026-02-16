#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///

import base64
from pathlib import Path

def image_to_base64(image_path):
    """Convert an image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Read the images
timeseries_b64 = image_to_base64('plot_timeseries.png')
histogram_b64 = image_to_base64('plot_histogram.png')
power_spectrum_b64 = image_to_base64('plot_power_spectrum.png')

# Create the HTML content
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rubin Observatory Mirror Temperature Analysis - Lab Notebook</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .date {{
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 20px;
        }}
        .finding {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 15px 0;
        }}
        .key-result {{
            font-weight: bold;
            color: #2980b9;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .code-link {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
        }}
        .code-link a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .code-link a:hover {{
            text-decoration: underline;
        }}
        .data-provenance {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        ul {{
            margin: 10px 0;
        }}
        li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Rubin Observatory Mirror Temperature Analysis</h1>
        <p class="date">Laboratory Notebook Entry - February 15, 2026</p>

        <h2>Overview</h2>
        <p>
            This notebook documents the analysis of temperature measurements from the Rubin Observatory
            primary mirror at Cerro Pachón, Chile. The dataset spans from June 1, 2025 to December 1, 2025,
            with measurements recorded at 15-minute intervals.
        </p>

        <h2>1. Initial Data Exploration</h2>
        <h3>Dataset Characteristics</h3>
        <div class="finding">
            <p><span class="key-result">Data file:</span> rubin_mirror_temps.csv</p>
            <p><span class="key-result">Total records:</span> 17,643 measurements (including header)</p>
            <p><span class="key-result">Time range:</span> June 1, 2025 00:00:00 to December 1, 2025 18:15:00</p>
            <p><span class="key-result">Sampling interval:</span> 15 minutes</p>
        </div>

        <h3>Data Structure</h3>
        <div class="finding">
            <p><span class="key-result">Columns:</span></p>
            <ul>
                <li><strong>timestamp</strong> - Date and time in format YYYY-MM-DD HH:MM:SS.mmm</li>
                <li><strong>temperature</strong> - Mirror temperature in degrees Celsius</li>
            </ul>
        </div>

        <h3>Temperature Range</h3>
        <div class="finding">
            <p><span class="key-result">Minimum temperature:</span> -9.0872 °C</p>
            <p><span class="key-result">Maximum temperature:</span> 20.2329 °C</p>
            <p><span class="key-result">Temperature span:</span> ~29.3 °C</p>
        </div>

        <h3>Data Quality</h3>
        <div class="finding">
            <p>✓ No missing values detected</p>
            <p>✓ No duplicate timestamps found</p>
            <p>✓ All temperature values are valid numeric entries</p>
            <p>✓ Data appears complete and well-formed</p>
        </div>

        <h2>2. Time Series Analysis - Recent Week</h2>
        <p>
            We analyzed the most recent week of data (November 25 - December 1, 2025) and calculated
            sunset times for Cerro Pachón (latitude -30.24°, longitude -70.74°) using the astral library.
            Sunset times are marked with red dots on the plot.
        </p>

        <div class="finding">
            <p><span class="key-result">Observation:</span> Temperature shows clear diurnal (daily) variation,
            with temperatures generally rising during the day and falling at night. Sunset times align with
            the evening cooling phase.</p>
        </div>

        <img src="data:image/png;base64,{timeseries_b64}" alt="Temperature Time Series - Recent Week">

        <h2>3. Temperature Distribution Analysis</h2>
        <p>
            A histogram of all temperature measurements reveals the overall distribution of mirror
            temperatures throughout the six-month observation period.
        </p>

        <div class="finding">
            <p><span class="key-result">Observation:</span> The temperature distribution is roughly
            bimodal with peaks around 10-12°C and 13-15°C, likely reflecting seasonal temperature
            variations and day/night cycles.</p>
        </div>

        <img src="data:image/png;base64,{histogram_b64}" alt="Temperature Distribution Histogram">

        <h2>4. Fourier Transform Analysis</h2>
        <p>
            We performed a Fast Fourier Transform (FFT) on the complete temperature time series to
            identify periodic components in the data. The power spectrum reveals the dominant
            periodicities present in the temperature variations.
        </p>

        <h3>Dominant Periods Identified</h3>
        <div class="finding">
            <p><span class="key-result">Top 5 Periodic Components:</span></p>
            <ol>
                <li><strong>61.26 days</strong> (Power: 4.09 × 10⁸) - Likely related to seasonal/weather patterns</li>
                <li><strong>183.77 days</strong> (Power: 3.37 × 10⁸) - ~6 months, semi-annual variation</li>
                <li><strong>36.75 days</strong> (Power: 3.10 × 10⁸) - Monthly-scale variation</li>
                <li><strong>1.00 days</strong> (Power: 2.52 × 10⁸) - <strong>Diurnal (daily) cycle</strong></li>
                <li><strong>16.71 days</strong> (Power: 2.08 × 10⁸) - Possible lunar or weather-related cycle</li>
            </ol>
            <p><span class="key-result">Key Finding:</span> The 1-day period confirms the expected
            diurnal temperature cycle driven by solar heating and nighttime cooling. The longer-period
            components (61 and 184 days) indicate significant seasonal temperature trends over the
            six-month observation window.</p>
        </div>

        <img src="data:image/png;base64,{power_spectrum_b64}" alt="Power Spectrum">

        <h2>5. Source Code</h2>
        <p>The analysis was performed using the following Python scripts:</p>

        <div class="code-link">
            <strong>Main plotting program:</strong> <a href="plot_mirror_temps.py">plot_mirror_temps.py</a>
            <br>Interactive version with real-time display of all three plots
        </div>

        <div class="code-link">
            <strong>Plot generation script:</strong> <a href="generate_plots.py">generate_plots.py</a>
            <br>Batch processing version that saves plots as PNG files
        </div>

        <h3>Dependencies</h3>
        <ul>
            <li><strong>pandas</strong> - Data loading and manipulation</li>
            <li><strong>matplotlib</strong> - Plotting and visualization</li>
            <li><strong>numpy</strong> - Numerical computing and FFT</li>
            <li><strong>astral</strong> - Astronomical calculations (sunset times)</li>
        </ul>

        <p>All scripts use <code>uv</code> for dependency management with inline script metadata (PEP 723).</p>

        <h2>6. Data Provenance</h2>
        <div class="data-provenance">
            <p><strong>Data File:</strong> <code>rubin_mirror_temps.csv</code></p>
            <p><strong>Source:</strong> Rubin Observatory Primary Mirror Temperature Monitoring System</p>
            <p><strong>Location:</strong> Cerro Pachón, Chile (Latitude: -30.24°, Longitude: -70.74°)</p>
            <p><strong>Observation Period:</strong> June 1, 2025 - December 1, 2025 (6 months)</p>
            <p><strong>Temporal Resolution:</strong> 15-minute intervals</p>
            <p><strong>Data Size:</strong> 560.8 KB (17,642 measurements)</p>
            <p><strong>File Format:</strong> CSV with two columns (timestamp, temperature)</p>
            <p><strong>Temperature Units:</strong> Degrees Celsius (°C)</p>
        </div>

        <h2>7. Conclusions</h2>
        <ul>
            <li>The mirror temperature data shows strong periodic behavior with dominant periods at
                1 day (diurnal cycle), ~2 months, and ~6 months (seasonal)</li>
            <li>Temperature range of approximately 29°C indicates significant thermal variation that
                must be managed for optimal telescope performance</li>
            <li>Clear correlation between sunset times and evening temperature decline confirms
                expected solar heating effects</li>
            <li>Data quality is excellent with no missing values or anomalies detected</li>
            <li>The bimodal temperature distribution suggests distinct thermal regimes, possibly
                related to seasonal transitions or operational states</li>
        </ul>

        <hr style="margin-top: 40px; border: none; border-top: 2px solid #ecf0f1;">
        <p style="text-align: center; color: #95a5a6; font-size: 0.9em; margin-top: 20px;">
            Notebook generated on February 15, 2026<br>
            Self-contained HTML document with embedded images
        </p>
    </div>
</body>
</html>
"""

# Write the HTML file
with open('lab_notebook.html', 'w') as f:
    f.write(html_content)

print("Lab notebook created successfully: lab_notebook.html")
