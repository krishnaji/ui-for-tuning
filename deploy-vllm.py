from google.cloud import aiplatform
from typing import Tuple

PROJECT_ID ="your-project-id"
REGION = "us-central1"
STAGING_BUCKET =f"gs://{PROJECT_ID}-vertex-ai-staging" 
SERVICE_ACCOUNT = f"SERVICE_ACCOUNT"
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
models, endpoints = {}, {}

VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240819_0916_RC00"

# Find Vertex AI prediction supported accelerators and regions [here](https://cloud.google.com/vertex-ai/docs/predictions/configure-compute).
if "8b" in base_model_id.lower():
    machine_type = "g2-standard-12"
    accelerator_type = "NVIDIA_L4"
    per_node_accelerator_count = 1
else:
    raise ValueError(f"Unsupported model ID or GCS path: {base_model_id}.")


# Dedicated endpoint not supported yet.
use_dedicated_endpoint = False

gpu_memory_utilization = 0.95
max_model_len = 8192  # Maximum context length.

# Ensure max_model_len does not exceed the limit
if max_model_len > 8192:
    raise ValueError("max_model_len cannot exceed 8192")


def deploy_model_vllm(
    model_name: str,
    model_id: str,
    service_account: str,
    base_model_id: str = None,
    machine_type: str = "g2-standard-8",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    dtype: str = "auto",
    enable_trust_remote_code: bool = False,
    enforce_eager: bool = False,
    enable_lora: bool = False,
    max_loras: int = 1,
    max_cpu_loras: int = 8,
    use_dedicated_endpoint: bool = False,
    max_num_seqs: int = 256,
    model_type: str = None,
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    """Deploys trained models with vLLM into Vertex AI."""
    endpoint = aiplatform.Endpoint.create(
        display_name=f"{model_name}-endpoint",
        dedicated_endpoint_enabled=use_dedicated_endpoint,
    )

    if not base_model_id:
        base_model_id = model_id

    # See https://docs.vllm.ai/en/latest/models/engine_args.html for a list of possible arguments with descriptions.
    vllm_args = [
        "python",
        "-m",
        "vllm.entrypoints.api_server",
        "--host=0.0.0.0",
        "--port=8080",
        f"--model={model_id}",
        f"--tensor-parallel-size={accelerator_count}",
        "--swap-space=16",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
        f"--max-model-len={max_model_len}",
        f"--dtype={dtype}",
        f"--max-loras={max_loras}",
        f"--max-cpu-loras={max_cpu_loras}",
        f"--max-num-seqs={max_num_seqs}",
        "--disable-log-stats",
    ]

    if enable_trust_remote_code:
        vllm_args.append("--trust-remote-code")

    if enforce_eager:
        vllm_args.append("--enforce-eager")

    if enable_lora:
        vllm_args.append("--enable-lora")

    if model_type:
        vllm_args.append(f"--model-type={model_type}")

    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = (
        f"projects/{PROJECT_ID}/secrets/huggingface/versions/latest" 
    )
    response = client.access_secret_version(request={"name": name})
    hf_token = response.payload.data.decode("UTF-8")


    env_vars = {
        "MODEL_ID": base_model_id,
        "DEPLOY_SOURCE": "notebook",
        "HF_TOKEN":hf_token
    }



    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=VLLM_DOCKER_URI,
        serving_container_args=vllm_args,
        serving_container_ports=[8080],
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_environment_variables=env_vars,
        serving_container_shared_memory_size_mb=(16 * 1024),  # 16 GB
        serving_container_deployment_timeout=7200,
    )
    print(
        f"Deploying {model_name} on {machine_type} with {accelerator_count} {accelerator_type} GPU(s)."
    )
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        deploy_request_timeout=1800,
        service_account=service_account,
    )
    print("endpoint_name:", endpoint.name)

    return model, endpoint




deploy_pretrained_model_id = base_model_id


models["vllm_gpu"], endpoints["vllm_gpu"] = deploy_model_vllm(
    model_name="llama3_1-vllm-serve",
    model_id=deploy_pretrained_model_id,
    service_account=SERVICE_ACCOUNT,
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=per_node_accelerator_count,
    gpu_memory_utilization=gpu_memory_utilization,
    max_model_len=max_model_len,
    enable_lora=True,
    use_dedicated_endpoint=use_dedicated_endpoint,
)

print(models["vllm_gpu"], endpoints["vllm_gpu"] )
