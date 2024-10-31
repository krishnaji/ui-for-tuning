from flask import Flask, request, jsonify
import vertexai
from vertexai.tuning import sft, TuningJob


app = Flask(__name__)
# init vertex ai

vertexai.init(project='cloud-llm-preview1', location='us-central1')

# Define a route to submit a tuning job

@app.route('/submit_tuning_job', methods=['POST'])
def submit_tuning_job():
    data = request.get_json()

    source_model = data.get('source_model')
    train_dataset = data.get('train_dataset')
    validation_dataset = data.get('validation_dataset')
    tuned_model_display_name = data.get('tuned_model_display_name')
    epochs = data.get('epochs')
    learning_rate_multiplier = data.get('learning_rate_multiplier')
    adapter_size = data.get('adapter_size')

    try:
        tuning_job = sft.train(
            source_model=source_model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            tuned_model_display_name=tuned_model_display_name,
            epochs=epochs,
            learning_rate_multiplier=learning_rate_multiplier,
            adapter_size=adapter_size
        )

        return jsonify({'message': 'Tuning job submitted successfully!', 'job_id': tuning_job.resource_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_tuning_job', methods=['POST'])
def get_tuning_job():
    data = request.get_json()
    job_id = data.get('job_id')

    try:
        tuning_job = TuningJob(tuning_job_name=job_id)
        tuning_job.refresh()

        return jsonify({
            'job_id': tuning_job.resource_name,
            'state': tuning_job.state.name,
            'tuned_model_name': tuning_job.tuned_model_name,
            'tuned_model_endpoint_name': tuning_job.tuned_model_endpoint_name,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/list_tuning_jobs', methods=['GET'])
def list_tuning_jobs():
    try:
        filter = request.args.get('filter')  # Optional filter string
        tuning_jobs = TuningJob.list(filter=filter)

        job_list = []
        for job in tuning_jobs:
            job_list.append({
                'job_id': job.resource_name,
                'state': job.state.name,
                'tuned_model_name': job.tuned_model_name,
                'tuned_model_endpoint_name': job.tuned_model_endpoint_name,
            })

        return jsonify({'tuning_jobs': job_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)