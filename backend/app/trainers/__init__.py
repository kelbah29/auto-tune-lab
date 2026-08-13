from app.core.schemas import Backend, TechniqueName, TechniqueSpec
from app.trainers.base import Trainer
from app.trainers.colab_trainer import ColabTrainer
from app.trainers.heretic_abliteration import HereticAbliterationTrainer
from app.trainers.hf_zerogpu_trainer import HFZeroGPUTrainer
from app.trainers.mlx_lora import MLXLoraTrainer
from app.trainers.tinker_trainer import TinkerLoraSftTrainer, TinkerPreferenceTrainer


def get_trainer(spec: TechniqueSpec) -> Trainer:
    if spec.backend == Backend.HERETIC_LOCAL:
        return HereticAbliterationTrainer()
    if spec.backend == Backend.MLX_LOCAL:
        return MLXLoraTrainer()
    if spec.backend == Backend.TINKER:
        if spec.name == TechniqueName.DPO:
            return TinkerPreferenceTrainer()
        return TinkerLoraSftTrainer()
    if spec.backend == Backend.HF_ZEROGPU:
        return HFZeroGPUTrainer(mode="dpo" if spec.name == TechniqueName.DPO else "sft")
    if spec.backend == Backend.COLAB:
        return ColabTrainer(mode="dpo" if spec.name == TechniqueName.DPO else "sft")
    raise ValueError(f"No trainer registered for backend={spec.backend}, technique={spec.name}")
