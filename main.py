import os
import pandas as pd
import gradio as gr
from datetime import datetime
import geohash2
from utils_holidays import load_holidays
import utils_report as ur

# Load data
try:
    network_df = pd.read_csv('network_list.csv')
    location_df = pd.read_csv('location_list.csv')
    otcRatio = pd.read_csv('jeki_truth_data_reference.csv')
    holiday = load_holidays([2024, 2025], 'JP')
    print("All files loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")
    network_df = pd.DataFrame({'NetworkId': ['Network1', 'Network2'], 'ReferenceId': ['Ref1', 'Ref2']})
    location_df = pd.DataFrame({
        'ReferenceId': ['Ref1', 'Ref2'],
        'lat': [35.6762, 35.6895],
        'lon': [139.6503, 139.6917],
        'spotsPerHour': [4, 6],
        'spotDuration': [15, 20],
        'dwellTime': [30, 45],
        'loopLength': [120, 180]
    })
    otcRatio = pd.DataFrame({'referenceId': ['Ref1', 'Ref2'], 'share': [0.8, 0.9], 'mediaRatio': [0.7, 0.8]})
    holiday = pd.DataFrame({'date': [], 'Day_Type_2': []})

# Compute geohash columns
location_df['geohash5'] = location_df.apply(lambda row: geohash2.encode(row['lat'], row['lon'], precision=5), axis=1)
location_df['geohash6'] = location_df.apply(lambda row: geohash2.encode(row['lat'], row['lon'], precision=6), axis=1)

network_names = sorted(network_df['NetworkId'].unique().tolist())
location_names = sorted(location_df['ReferenceId'].unique().tolist())

# Initialise util module globals
ur.initialize_data(network_df, location_df, otcRatio, holiday)

# --- Gradio UI ------------------------------------------------------------
with gr.Blocks(title="Post Campaign Report Tool") as demo:
    gr.Markdown("# Post Campaign Report Data Generation Tool")

    with gr.Tab("Add Network Campaign"):
        with gr.Row():
            with gr.Column():
                name_n = gr.Dropdown(choices=network_names, label="Select Network Name", value=None)
                start_date_n = gr.Textbox(label="Start Date (YYYY-MM-DD)")
                end_date_n = gr.Textbox(label="End Date (YYYY-MM-DD)")
            with gr.Column():
                hour_type_n = gr.CheckboxGroup(choices=['morning','afternoon','evening','night','full'], label="Hour Types", value=[])
                spots_n = gr.Number(label="Spots per Hour", value=1, minimum=1)
                btn_n = gr.Button("Add Network to Cart", variant="primary")
        out_n = gr.Textbox(label="Status Message")

    with gr.Tab("Add Location Campaign"):
        with gr.Row():
            with gr.Column():
                name_l = gr.Dropdown(choices=location_names, label="Select Location Name", value=None)
                start_date_l = gr.Textbox(label="Start Date (YYYY-MM-DD)")
                end_date_l = gr.Textbox(label="End Date (YYYY-MM-DD)")
            with gr.Column():
                hour_type_l = gr.CheckboxGroup(choices=['morning','afternoon','evening','night','full'], label="Hour Types", value=[])
                spots_l = gr.Number(label="Spots per Hour", value=1, minimum=1)
                btn_l = gr.Button("Add Location to Cart", variant="primary")
        out_l = gr.Textbox(label="Status Message")

    with gr.Tab("Cart Management"):
        gr.Markdown("## Current Cart Items")
        cart_display = gr.Dataframe(label="Cart Contents", interactive=False)
        with gr.Row():
            with gr.Column(scale=3):
                remove_dropdown = gr.Dropdown(label="Select Item to Remove", choices=[], interactive=True)
            with gr.Column(scale=1):
                remove_btn = gr.Button("Remove Selected", variant="secondary")
            with gr.Column(scale=1):
                clear_btn = gr.Button("Clear All", variant="stop")
        cart_message = gr.Textbox(label="Cart Status", interactive=False)

    with gr.Tab("Generate Report"):
        gr.Markdown("## Generate Impression Report")
        with gr.Row():
            report_btn = gr.Button("Generate Report", variant="primary", size="lg")
        report_output = gr.Dataframe(label="Campaign Report", interactive=False)
        gr.Markdown("## Download Options")
        with gr.Row():
            with gr.Column():
                download_csv_btn = gr.Button("Download Full Data (CSV)", variant="secondary")
                csv_download = gr.File(label="CSV Download", visible=False)
            with gr.Column():
                download_excel_btn = gr.Button("Download Excel Report", variant="secondary")
                excel_download = gr.File(label="Excel Download", visible=False)

    # -- Events --
    btn_n.click(
        fn=ur.add_to_cart,
        inputs=[gr.Textbox(value="Network", visible=False), name_n, start_date_n, end_date_n, hour_type_n, spots_n],
        outputs=[out_n, cart_display, remove_dropdown]
    )
    btn_l.click(
        fn=ur.add_to_cart,
        inputs=[gr.Textbox(value="Location", visible=False), name_l, start_date_l, end_date_l, hour_type_l, spots_l],
        outputs=[out_l, cart_display, remove_dropdown]
    )
    remove_btn.click(
        fn=ur.remove_from_cart,
        inputs=[remove_dropdown],
        outputs=[cart_message, cart_display, remove_dropdown]
    )
    clear_btn.click(
        fn=ur.clear_cart,
        outputs=[cart_message, cart_display, remove_dropdown]
    )
    report_btn.click(
        fn=ur.generate_report,
        outputs=[report_output]
    )
    download_csv_btn.click(
        fn=ur.download_full_data,
        outputs=[csv_download]
    ).then(
        fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False),
        inputs=[csv_download],
        outputs=[csv_download]
    )
    download_excel_btn.click(
        fn=ur.download_excel_report,
        outputs=[excel_download]
    ).then(
        fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False),
        inputs=[excel_download],
        outputs=[excel_download]
    )
    demo.load(
        fn=lambda: (ur.get_cart_display(), ur.update_remove_choices()),
        outputs=[cart_display, remove_dropdown]
    )

if __name__ == '__main__':
    demo.launch(debug=True, share=True)
