from google.cloud import aiplatform
 

PROJECT_ID = "genai-380800"  # Replace with your project ID
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
    replica_count: int = 1,  
    machine_type: str = "a2-highgpu-8g", 
    accelerator_type: str = "NVIDIA_TESLA_A100",  
    accelerator_count: int = 8,     
    llamafactory_yaml_path: str = "/gcs/shkhose-test-bucket-unique/llama3_lora_sft.yaml"
):
    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,   
                "accelerator_type":  getattr(aiplatform.gapic.AcceleratorType, accelerator_type),   
                "accelerator_count": accelerator_count,
            },
            "replica_count": replica_count,
            "container_spec": {
                "image_uri": container_image_uri,
                "env": [
                    {"name": "HF_TOKEN", "value": hf_token}, 
                    {"name": "PYTHONUNBUFFERED", "value": "0"},
                    {"name": "WORLD_SIZE", "value": str(replica_count)},
                    { "name": "FORCE_TORCHRUN","value":"1"},
                    {"name": "NNODES", "value":"1"},
                    {"name":"RANK","value" :"0"}
                ],
                "command": [
                    "bash",
                    "-c",
                    f"/usr/local/bin/llamafactory-cli train {llamafactory_yaml_path}",
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

print(hf_token)

# Create and run the custom job
create_custom_job(
    project_id=PROJECT_ID,
    region=REGION,
    staging_bucket=STAGING_BUCKET,
    container_image_uri=CONTAINER_IMAGE_URI,
    hf_token=hf_token,
    replica_count=REPLICA_COUNT,

)
print("Custom job submitted successfully.")   