# Location Analysis Report Tool

This repository contains a small Gradio application to generate post campaign reports.

## Setup

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the interface
   ```bash
   python main.py
   ```

The application loads network and location data, allows you to create a campaign cart and downloads the results as CSV or Excel.
Pandas will use `openpyxl` by default for Excel output. If it is not installed,
`xlsxwriter` will be used instead.


