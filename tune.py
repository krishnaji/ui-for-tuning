import datetime
import importlib
import os
import subprocess
import uuid
from typing import Tuple

from google.cloud import aiplatform
from google.cloud.aiplatform.compat.types import \
    custom_job as gca_custom_job_compat

# Check if vertex-ai-samples exists and has the correct commit
if not os.path.exists("vertex-ai-samples") or (
    os.path.exists("vertex-ai-samples")
    and subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd="vertex-ai-samples",
        capture_output=True,
        text=True,
    ).stdout.strip()
    != "0727e19520cf7957bceb701c248221bd3dbe4f1f"
):
    print("Cloning or updating vertex-ai-samples repository...")
    if os.path.exists("vertex-ai-samples"):
        subprocess.run(["rm", "-rf", "vertex-ai-samples"], check=True)
    subprocess.run(
        ["git", "clone", "https://github.com/GoogleCloudPlatform/vertex-ai-samples.git"],
        check=True,
    )
    subprocess.run(
        ["git", "reset", "--hard", "0727e19520cf7957bceb701c248221bd3dbe4f1f"],
        cwd="vertex-ai-samples",
        check=True,
    )

common_util = importlib.import_module(
    "vertex-ai-samples.community-content.vertex_model_garden.model_oss.notebook_util.common_util"
)

models, endpoints = {}, {}


def train_llama_model(
    base_model_id: str,
    MODEL_BUCKET: str ,
    job_name: str,
    pretrained_model_id: str,
    train_dataset_name: str,
    train_split_name: str,
    eval_dataset_name: str,
    eval_split_name: str,
    instruct_column_in_dataset: str,
    template: str,
    base_output_dir: str,
    lora_output_dir: str,
    final_checkpoint: str,
    accelerator_type: str,
    replica_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    max_steps: int,
    num_epochs: float,
    finetuning_precision_mode: str,
    learning_rate: float,
    lr_scheduler_type: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    enable_gradient_checkpointing: bool,
    attn_implementation: str,
    optimizer: str,
    warmup_ratio: str,
    report_to: str,
    save_steps: int,
    logging_steps: int,
    SERVICE_ACCOUNT: str = None,   
):
    """Trains a Llama 3.1 model with LoRA using Vertex AI."""


    # Derive SERVICE_ACCOUNT if not provided
    SERVICE_ACCOUNT = f"801452371447-compute@developer.gserviceaccount.com"

    pretrained_model_id = os.path.join(MODEL_BUCKET, base_model_id.split("/")[-1])

    # Acceletor type to use for training.
    if accelerator_type == "NVIDIA_A100_80GB":
        repo = "us-docker.pkg.dev/vertex-ai-restricted"
        is_restricted_image = True
        is_dynamic_workload_scheduler = False
        dws_kwargs = {}
        if "405b" in base_model_id.lower():  # Use the passed base_model_id
            raise ValueError(
                "405B model is not supported with Nvidia A100 GPUs. Use Nvidia H100 GPUs instead."
            )
    else:  # Assuming NVIDIA_H100_80GB or other supported type
        repo = "us-docker.pkg.dev/vertex-ai"
        is_restricted_image = False
        is_dynamic_workload_scheduler = True
        dws_kwargs = {
            "max_wait_duration": 1800,  # 30 minutes
            "scheduling_strategy": gca_custom_job_compat.Scheduling.Strategy.FLEX_START,
        }

    TRAIN_DOCKER_URI = (
        f"{repo}/vertex-vision-model-garden-dockers/pytorch-peft-train:stable_20240909"
    )

    # Worker pool spec.
    if accelerator_type == "NVIDIA_A100_80GB":
        per_node_accelerator_count = 8
        machine_type = "a2-ultragpu-8g"
        boot_disk_size_gb = 500
    elif accelerator_type == "NVIDIA_H100_80GB":
        per_node_accelerator_count = 8
        machine_type = "a3-highgpu-8g"
        boot_disk_size_gb = 2000
    else:
        raise ValueError(
            f"Recommended machine settings not found for: {accelerator_type}. To use another accelerator type, edit this code block to pass in an appropriate `machine_type`, `accelerator_type`, and `per_node_accelerator_count` to the deploy_model_vllm function by clicking `Show Code` and then modifying the code."
        )

    # Set config file.
    if accelerator_type == "NVIDIA_A100_80GB":
        if replica_count == 1:
            config_file = "vertex_vision_model_garden_peft/llama_fsdp_8gpu.yaml"
        elif replica_count <= 4:
            config_file = (
                "vertex_vision_model_garden_peft/"
                f"llama_hsdp_{replica_count * per_node_accelerator_count}gpu.yaml"
            )
        else:
            raise ValueError(
                f"Recommended config settings not found for replica_count: {replica_count}."
            )
    elif accelerator_type == "NVIDIA_H100_80GB":
        if replica_count == 1:
            config_file = "vertex_vision_model_garden_peft/llama_fsdp_8gpu_h100.yaml"
        elif replica_count == 2:
            config_file = "vertex_vision_model_garden_peft/llama_fsdp_16gpu_h100.yaml"
        # Add other configurations for H100 based on replica_count as needed
        else:
            raise ValueError(
                f"Recommended config settings not found for replica_count: {replica_count} with {accelerator_type}"
            )
    else:
        raise ValueError(
            f"Recommended config settings not found for accelerator_type: {accelerator_type}."
        )

    # Add labels for the finetuning job.
    labels = {
        "finetuning_source": "notebook",
        "notebook_name": "model_garden_pytorch_llama3_1_finetuning.ipynb".split(".")[0],
    }

    eval_args = [
        f"--eval_dataset_path={eval_dataset_name}",
        f"--eval_column={instruct_column_in_dataset}",
        f"--eval_template={template}",
        f"--eval_split={eval_split_name}",
        f"--eval_steps={save_steps}",
        "--eval_tasks=builtin_eval",
        "--eval_metric_name=loss",
    ]

    train_job_args = [
        f"--config_file={config_file}",
        "--task=instruct-lora",
        "--completion_only=True",
        f"--pretrained_model_id={pretrained_model_id}",
        f"--dataset_name={train_dataset_name}",
        f"--train_split_name={train_split_name}",
        f"--instruct_column_in_dataset={instruct_column_in_dataset}",
        f"--output_dir={lora_output_dir}",
        f"--per_device_train_batch_size={per_device_train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--lora_rank={lora_rank}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_dropout={lora_dropout}",
        f"--max_steps={max_steps}",
        f"--max_seq_length={max_seq_length}",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler_type={lr_scheduler_type}",
        f"--precision_mode={finetuning_precision_mode}",
        f"--enable_gradient_checkpointing={enable_gradient_checkpointing}",
        f"--num_epochs={num_epochs}",
        f"--attn_implementation={attn_implementation}",
        f"--optimizer={optimizer}",
        f"--warmup_ratio={warmup_ratio}",
        f"--report_to={report_to}",
        f"--logging_output_dir={base_output_dir}",
        f"--save_steps={save_steps}",
        f"--logging_steps={logging_steps}",
        f"--template={template}",
    ] + eval_args

    # Pass training arguments and launch job.
    train_job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=TRAIN_DOCKER_URI,
        labels=labels,
    )

    print("Running training job with args:")
    print(" \\\n".join(train_job_args))
    train_job.run(
        args=train_job_args,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=per_node_accelerator_count,
        boot_disk_size_gb=boot_disk_size_gb,
        service_account=SERVICE_ACCOUNT,
        base_output_dir=base_output_dir,
        sync=False,  # Non-blocking call to run.
        **dws_kwargs,
    )

    # Wait until resource has been created.
    train_job.wait_for_resource_creation()

    print("LoRA adapter will be saved in:", lora_output_dir)
    print("Final checkpoint will be saved in:", final_checkpoint)

    print(f"Command to copy: tensorboard --logdir {base_output_dir}/logs")

    # if train_job.end_time is None:
    #     print("Waiting for the training job to finish...")
    #     train_job.wait()
    #     print("The training job has finished.")

    return train_job
 
# Define training parameters
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
max_seq_length = 4096
max_steps = -1
num_epochs = 1.0
finetuning_precision_mode = "4bit"
learning_rate = 5e-5
lr_scheduler_type = "cosine"
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
enable_gradient_checkpointing = True
attn_implementation = "flash_attention_2"
optimizer = "adamw_torch"
warmup_ratio = "0.01"
report_to = "tensorboard"
save_steps = 10
logging_steps = save_steps
base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
accelerator_type = "NVIDIA_A100_80GB"  
replica_count = 1  
template = "llama3-text-bison"   
train_dataset_name = "gs://cloud-samples-data/vertex-ai/model-evaluation/peft_train_sample.jsonl"
train_split_name = "train"  
eval_dataset_name = "gs://cloud-samples-data/vertex-ai/model-evaluation/peft_eval_sample.jsonl"
eval_split_name = "train"  
instruct_column_in_dataset = "input_text"  

MODEL_BUCKET ="gs://shkhose-tune/llama3.1/"
PROJECT_ID ="cloud-llm-preview1"

print("Initializing Vertex AI API.")

aiplatform.init(project=PROJECT_ID, location="us-central1",staging_bucket=MODEL_BUCKET)

train_llama_model(
    base_model_id=base_model_id,
    MODEL_BUCKET=MODEL_BUCKET,
    job_name=f"llama-3.1-8b-instruct-finetuning-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
    pretrained_model_id=base_model_id,
    train_dataset_name=train_dataset_name,
    train_split_name=train_split_name,
    eval_dataset_name=eval_dataset_name,
    eval_split_name=eval_split_name,
    instruct_column_in_dataset=instruct_column_in_dataset,
    template=template,
    base_output_dir=f"{MODEL_BUCKET}/llama-3.1-8b-instruct-finetuning-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
    lora_output_dir=f"{MODEL_BUCKET}/llama-3.1-8b-instruct-finetuning-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}/lora_output",
    final_checkpoint=f"{MODEL_BUCKET}/llama-3.1-8b-instruct-finetuning-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}/final_checkpoint",
    accelerator_type=accelerator_type,
    replica_count=replica_count,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_seq_length=max_seq_length,
    max_steps=max_steps,
    num_epochs=num_epochs,
    finetuning_precision_mode=finetuning_precision_mode,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    enable_gradient_checkpointing=enable_gradient_checkpointing,
    attn_implementation=attn_implementation,
    optimizer=optimizer,
    warmup_ratio=warmup_ratio,
    report_to=report_to,
    save_steps=save_steps,
    logging_steps=logging_steps,
)