from google.cloud import aiplatform
 

PROJECT_ID = "<your-project-id>"  # Replace with your project ID
REGION = "us-central1"   
STAGING_BUCKET = f"gs://{PROJECT_ID}-vertex-ai-staging"  
CONTAINER_IMAGE_URI = "us-central1-docker.pkg.dev/{PROJECT_ID}/llamafactory/llama-factory:latest"   
REPLICA_COUNT = 1   


def create_custom_job(
    project_id: str,
    region: str,
    staging_bucket: str,
    container_image_uri: str,
    hf_token: str,
):
    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",   
                "accelerator_type": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_T4,   
                "accelerator_count": 1,
            },
            "replica_count": REPLICA_COUNT,
            "container_spec": {
                "image_uri": container_image_uri,
                "env": [
                    {"name": "HF_TOKEN", "value": hf_token},
                    {"name": "HUGGING_FACE_HUB_TOKEN", "value": hf_token},
                    {"name": "PYTHONUNBUFFERED", "value": "0"},
                    {"name": "WORLD_SIZE", "value": str(REPLICA_COUNT)},
                ],
                "command": [
                    "bash",
                    "-c",
                    f"python -m torchrun --nproc_per_node=1 --nnodes={REPLICA_COUNT} --node_rank=${{CLOUD_ML_NODE_ID}} --master_addr=${{VERTEX_JOB_NAME}}-pytorch-workers-0-0 --master_port=3389 /app/llama-factory/llamafactory-cli train /app/llama-factory/examples/train_lora/llama3_lora_sft.yaml",
                ],
            },
        }
    ]

    custom_job = aiplatform.CustomJob(
        display_name="llama-factory-custom-job",   
        worker_pool_specs=worker_pool_specs,
        staging_bucket=staging_bucket,
    )

    custom_job.run()
    return custom_job


 
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = (
    f"projects/{PROJECT_ID}/secrets/huggingface/versions/latest" 
)
response = client.access_secret_version(request={"name": name})
hf_token = response.payload.data.decode("UTF-8")


# Create and run the custom job
create_custom_job(
    project_id=PROJECT_ID,
    region=REGION,
    staging_bucket=STAGING_BUCKET,
    container_image_uri=CONTAINER_IMAGE_URI,
    hf_token=hf_token,
)
print("Custom job submitted successfully.")   