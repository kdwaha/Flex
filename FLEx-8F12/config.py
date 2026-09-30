"""Command-line configuration for federated instruction tuning.

The original release mixed versions of TRL and Transformers and exposed a few
arguments that were not wired into the trainer.  This module keeps the original
surface where possible, while making the executable configuration explicit for
FedAvg, FedSA-LoRA, and FRLoRA experiments.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from transformers import BitsAndBytesConfig, HfArgumentParser
from trl import SFTConfig
from peft import LoraConfig
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime, timedelta
from utils import select_random_expert_modules

# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(
        default="fedavg",
        metadata={
            "help": (
                "Federated algorithm: fedavg, fedsa (FedSA-LoRA), frlora, "
                "frlora_scaffold, scaffold_reset, "
                "pflalign, fedavgm, fedadagrad, fedyogi, fedadam, fedprox, scaffold, or flexlora"
            )
        },
    )
    num_rounds: Optional[int] = field(default=500, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(
        default="iid",
        metadata={"help": "Client split: iid, no-iid/task (one task per client), or dirichlet."},
    )
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    save_model_freq: Optional[int] = field(default=10, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})
    whether_same_sample: Optional[bool] = field(default=False, metadata={"help": "whether every client has same samples"})
    category_list: List[str] = field(
        default_factory=lambda: [
            "brainstorming",
            "classification",
            "closed_qa",
            "creative_writing",
            "information_extraction",
            "open_qa",
            "summarization",
        ],
        metadata={"help": "Task/category assigned to each client for no-iid/task splitting."},
    )
    eval_fraction: Optional[float] = field(
        default=0.1,
        metadata={"help": "Per-client held-out fraction for final task-wise evaluation; set 0 to disable."},
    )
    aggregation_weighting: Optional[str] = field(
        default="sample",
        metadata={"help": "Aggregation weighting for FRLoRA: sample (default) or uniform."},
    )
    pflalign_beta: float = field(default=0.9, metadata={"help": "pFLAlign moment decay beta."})
    pflalign_epsilon: float = field(default=1e-12, metadata={"help": "pFLAlign numerical epsilon."})
    pflalign_variant: str = field(default="full", metadata={"help": "pFLAlign component ablation; full preserves Algorithm 1."})

@dataclass
class ScriptArguments:

    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Optional Hugging Face dataset configuration/subset name."}
    )
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})    # vicuna and alpaca use 2e-5
    round_lr_schedule: str = field(default="constant", metadata={"help": "constant or cosine across federated rounds"})
    diagnostics_every: int = field(default=0, metadata={"help": "Periodic local/global evaluation interval; 0 disables."})
    hessian_lanczos_steps: int = field(default=20)
    hessian_probes: int = field(default=2)
    hessian_trace_probes: int = field(default=4)
    hessian_every: int = field(default=0, metadata={"help": "0: final round only; positive: round interval"})
    baseline_eval_only: bool = field(default=False, metadata={"help": "Evaluate the untouched pretrained model and exit without training."})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Deprecated alias for max_length; when max_length is unset this value is used."},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "Load the model in 8-bit precision (not supported by FRLoRA)."})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "Load the model in 4-bit precision (not supported by FRLoRA)."})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    add_side_experts: Optional[bool] = field(default=False, metadata={"help": "add side experts"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    max_length: Optional[int] = field(default=None, metadata={"help": "Maximum token length. Defaults to seq_length."})
    eval_steps: Optional[int] = field(default=10, metadata={"help": "the number of evaluation steps"})
    eval_strategy: Optional[str] = field(default="no", metadata={"help": "eval strategy"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})
    dataset_sample: Optional[int] = field(default=20000, metadata={"help": "the number of samples to use from the dataset"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})
    test_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional separate test dataset path. When provided, it is partitioned by "
                "client task and is never used for local training."
            )
        },
    )
    test_dataset_sample: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap for examples loaded from --test_data_dir."},
    )
    task_column: Optional[str] = field(
        default=None,
        metadata={"help": "Source task/category column. If omitted, the dataset adapter auto-detects it."},
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Model dtype: auto, float32, float16, or bfloat16."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Trainer checkpoint strategy; use 'no' to disable intermediate checkpoints."},
    )
    save_final_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Save a final checkpoint. FRLoRA saves a full model because its base weights change."},
    )
    generation_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Compute prompt-only ROUGE-L and BLEU on each client's held-out task "
                "after the final round."
            )
        },
    )
    post_local_generation_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "After every selected client's local update and before aggregation, "
                "compute ROUGE-L and BLEU on that client's held-out task."
            )
        },
    )
    post_local_generation_eval_final_round_only: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "With --post_local_generation_eval, evaluate only the final round's "
                "post-local client states before aggregation."
            )
        },
    )
    generation_max_new_tokens: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum tokens generated per held-out prompt for ROUGE-L/BLEU."},
    )
    generation_eval_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Generation batch size for held-out ROUGE-L/BLEU evaluation."},
    )

    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    random_select_experts: Optional[bool] = field(default=False, metadata={"help": "Whether random select experts"})
    personalized: Optional[bool] = field(default=True, metadata={"help": "Whether use personalized experts weight"})

parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()

if script_args.max_length is None:
    script_args.max_length = script_args.seq_length
if script_args.round_lr_schedule not in {"constant", "cosine"}:
    raise ValueError("round_lr_schedule must be constant or cosine")
if script_args.diagnostics_every < 0 or min(script_args.hessian_lanczos_steps, script_args.hessian_probes, script_args.hessian_trace_probes) < 1:
    raise ValueError("Invalid diagnostic interval or Hessian budget")

if fed_args.fed_alg not in {
    "fedavg", "fedsa", "frlora", "frlora_scaffold", "scaffold_reset", "fedavgm", "fedadagrad", "fedyogi",
    "fedadam", "fedprox", "scaffold", "flexlora", "pflalign",
} and not fed_args.fed_alg.startswith("local"):
    raise ValueError(f"Unsupported `fed_alg`: {fed_args.fed_alg}")

if fed_args.aggregation_weighting not in {"sample", "uniform"}:
    raise ValueError("`aggregation_weighting` must be `sample` or `uniform`.")
if not 0.0 <= fed_args.pflalign_beta < 1.0 or fed_args.pflalign_epsilon <= 0.0:
    raise ValueError("pFLAlign requires 0 <= beta < 1 and epsilon > 0.")
if fed_args.pflalign_variant not in {"full", "no_preconditioner", "no_correction", "constant_gamma", "constant_gamma_one", "hard_gamma", "no_personalization", "sgd"}:
    raise ValueError("Unsupported pFLAlign ablation variant")

if fed_args.fed_alg in {"frlora", "frlora_scaffold"} and (script_args.load_in_8bit or script_args.load_in_4bit):
    raise ValueError("FRLoRA updates frozen base weights and cannot be used with 4/8-bit quantization.")

if script_args.test_dataset_sample is not None and script_args.test_dataset_sample <= 0:
    raise ValueError("`test_dataset_sample` must be positive when provided.")
if script_args.generation_max_new_tokens <= 0:
    raise ValueError("`generation_max_new_tokens` must be positive.")
if script_args.generation_eval_batch_size <= 0:
    raise ValueError("`generation_eval_batch_size` must be positive.")

# ===== Define the LoraConfig =====
if script_args.use_peft:
    if not script_args.random_select_experts :
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=script_args.target_modules,
        )
    else :
        num_layers = 24
        num_experts = 60

        target_modules_for_lora = select_random_expert_modules(
            num_layers=num_layers,
            num_experts_per_layer=num_experts,
            base_path_template="model.layers.{layer_idx}.mlp.experts.{expert_idx}.{linear_name}" 
        )
        script_args.target_modules += target_modules_for_lora
        
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=script_args.target_modules,
        )
else:
    peft_config = None

def get_config():
    return script_args, fed_args, peft_config

# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = SFTConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_strategy=script_args.save_strategy,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
        max_seq_length=script_args.max_length,
        eval_steps=script_args.eval_steps,
        eval_strategy=script_args.eval_strategy,
        bf16=(script_args.torch_dtype == "bfloat16" and torch.cuda.is_available()),
        fp16=(script_args.torch_dtype == "float16" and torch.cuda.is_available()),
        use_cpu=not torch.cuda.is_available(),
    )
    return training_args

def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    elif script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = "auto" if torch.cuda.is_available() else None
        quantization_config = None
        if script_args.torch_dtype == "float32":
            torch_dtype = torch.float32
        elif script_args.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif script_args.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return device_map, quantization_config, torch_dtype

def save_config(script_args, fed_args):
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    dataset_name_split = os.path.basename(script_args.dataset_name.rstrip("/"))
    output_dir = f"{script_args.output_dir}/{dataset_name_split}_{script_args.dataset_sample}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
    os.makedirs(script_args.output_dir, exist_ok=True)
    while True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            break
        else:
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_{now_time}"

    script_args.output_dir = output_dir
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
        }
        json.dump(combined_dict, f, indent=4)
