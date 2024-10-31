# Gradio frontend for app.py
import gradio as gr
import requests
import pandas as pd


# Define the interface
def submit_tuning_job(source_model, train_dataset, validation_dataset, tuned_model_display_name, epochs, learning_rate_multiplier, adapter_size):
    # Send a POST request to the backend API
    response = requests.post('http://localhost:8080/submit_tuning_job', json={
        'source_model': source_model,
        'train_dataset': train_dataset,
        'validation_dataset': validation_dataset,
        'tuned_model_display_name': tuned_model_display_name,
        'epochs': epochs,
        'learning_rate_multiplier': learning_rate_multiplier,
        'adapter_size': adapter_size
    })

    # Handle the response
    if response.status_code == 200:
        # Extract the message from the response
        message = response.json().get('message')
        # Extrac job_id
        job_id = response.json().get('job_id')
        # return message and job_id
        return f"Message: {message}\nJob ID: {job_id}"
    else:
        return 'Error submitting tuning job'

# Function to fetch and display tuning jobs
def list_tuning_jobs():
    try:
        response = requests.get('http://localhost:8080/list_tuning_jobs')
        if response.status_code == 200:
            tuning_jobs = response.json().get('tuning_jobs')
            # Create a Pandas DataFrame from the tuning jobs data
            df = pd.DataFrame(tuning_jobs)
            return df
        else:
            return 'Error fetching tuning jobs'
    except Exception as e:
        return f"An error occurred: {e}"
      

#  Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# Fine Tuning App")
    with gr.Row():
        with gr.Column():
            source_model = gr.Dropdown(choices=["gemini-1.5-pro-002", "gemini-1.5-flash-002"],value="gemini-1.5-pro-002", label='Source Model')
            train_dataset = gr.Textbox(label='Train Dataset',value="gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl")
            validation_dataset = gr.Textbox(label='Validation Dataset',value="gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_validation_data.jsonl")
            tuned_model_display_name = gr.Textbox(label='Tuned Model Display Name')
            epochs = gr.Number(label='Epochs',value=4)
            learning_rate_multiplier = gr.Number(label='Learning Rate Multiplier',value=1)
            adapter_size = gr.Dropdown(choices=[1,4,8,16],value=4,label='Adapter Size')
            submit_btn = gr.Button("Submit Tuning Job")
        with gr.Column():
            response_textbox = gr.Textbox(label='Response')
            with gr.Row():
                # Display tuning jobs
                tuning_jobs_output = gr.DataFrame(label="Tuning Jobs")
            
    
    # Define the behavior when the submit button is clicked
    submit_btn.click(
        fn=submit_tuning_job,
        inputs=[source_model, train_dataset, validation_dataset, tuned_model_display_name, epochs, learning_rate_multiplier, adapter_size],
        outputs=response_textbox,
    )
    demo.load(fn=list_tuning_jobs, inputs=None, outputs=tuning_jobs_output)


# Launch the interface
demo.launch(share=False)
 