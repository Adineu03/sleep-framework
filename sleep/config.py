"""
SLEEP System Configuration

All hyperparameters from the Master Hyperparameter Table (SLEEP_Formalization.md, Final Synthesis).
Organized by module. Defaults match the formalization exactly — any deviation must be
logged in the experiment logbook.
"""

from dataclasses import dataclass, field


@dataclass
class TaggingConfig:
    """Part 1: The Tagging Layer (Q1.1–Q1.6)"""

    # Key projection
    d_tag: int = 128                    # Q1.1: tag key dimensionality

    # Adaptive threshold
    beta: float = 0.99                  # Q1.2: EMA smoothing factor
    kappa: float = 1.5                  # Q1.2: z-score sensitivity (std devs above mean)
    gap_tolerance: int = 3              # Q1.2: max gap between flagged tokens to merge
    min_span: int = 4                   # Q1.2: minimum span length (tokens)

    # Tag creation
    alpha_init: float = 2.0             # Q1.3: prediction error → initial strength scaling
    kappa_wfast: float = 2.5            # Q1.3: z-score threshold for W_fast updates

    # Decay
    tau_base: int = 1000                # Q1.4: base decay time constant (inference steps)
    gamma_decay: float = 0.5            # Q1.4: error-dependent decay scaling
    epsilon: float = 0.01               # Q1.4: decay floor
    epsilon_gc: float = 0.02            # Q1.4: garbage collection threshold
    gc_interval: int = 100              # Q1.4: steps between GC runs

    # Reinforcement
    delta_s: float = 0.3                # Q1.5: base reinforcement amount per access
    theta_access: float = 0.7           # Q1.5: cosine similarity threshold for tag activation

    # Capacity
    c_tag: int = 5000                   # Q1.6: tags per billion model parameters


@dataclass
class PRPConfig:
    """Part 2: PRP Allocation (Q2.1–Q2.5)"""

    # Scoring weights
    w_error: float = 0.35               # Q2.2: weight of cumulative prediction error
    w_access: float = 0.30              # Q2.2: weight of access frequency
    w_crossref: float = 0.15            # Q2.2: weight of cross-reference density
    w_recency: float = 0.20             # Q2.2: weight of recency-weighted utility

    # Cross-reference
    theta_xref: float = 0.5             # Q2.2: cosine threshold for cross-reference edges
    crossref_interval: int = 500        # Q2.2: steps between batch cross-ref computation

    # Recency
    tau_recency: int = 500              # Q2.2: decay constant for recency-weighted utility

    # Budget
    c_prp: int = 500                    # Q2.3: PRP budget per billion model parameters

    # Allocation
    delta_steal: float = 0.05           # Q2.4: min score gap for PRP reallocation
    allocation_interval: int = 100      # Q2.4: steps between PRP allocation evaluations

    # Threshold
    kappa_prp: float = 0.5              # Q2.5: std devs above mean for PRP threshold
    theta_floor: float = 0.2            # Q2.5: minimum PRP threshold


@dataclass
class WeightsConfig:
    """Part 3: Dual Weight System (Q3.1–Q3.4)"""

    # LoRA architecture
    lora_rank: int = 16                 # Q3.1: rank of W_fast/W_cons adapters
    lora_alpha: int = 32                # Q3.1: LoRA scaling factor
    adapted_fraction: float = 1 / 3     # Q3.1: fraction of top layers adapted
    adapted_matrices: list = field(      # Q3.1: which attention matrices get LoRA
        default_factory=lambda: ["v_proj", "o_proj"]
    )

    # W_fast learning
    alpha_fast: float = 1e-4            # Q3.3: base learning rate for W_fast
    momentum_fast: float = 0.9          # Q3.3: SGD momentum for W_fast

    # W_slow / W_cons plasticity
    alpha_slow: float = 1e-5            # Q3.4: base learning rate for sleep training
    delta_max: float = 0.001            # Q3.4: max relative weight change per layer
    phi_min: float = 0.10               # Q3.4: plasticity floor for lowest layers

    # EWC
    lambda_ewc: float = 100.0           # Q3.4: Fisher information penalty strength
    fisher_refresh_interval: int = 10   # Q3.4: sleep cycles between Fisher recomputation
    fisher_calibration_mix: float = 0.7 # Q3.4: fraction of calibration data in Fisher refresh

    # Safety
    epsilon_degrade: float = 0.02       # Q3.4: max PPL increase before rollback


@dataclass
class SleepConfig:
    """Part 4: The Sleep Engine (Q4.1–Q4.6)"""

    # Replay generation
    replay_temperature: float = 0.7     # Q4.1: sampling temperature for replay
    replay_top_p: float = 0.9           # Q4.1: nucleus sampling threshold
    seed_length_max: int = 32           # Q4.1: max tokens from original used as seed
    theta_quality: float = 0.5          # Q4.1: min cosine similarity for replay acceptance
    max_generation_attempts: int = 3    # Q4.1: retries for failed quality checks

    # Compression
    compression_target: int = 5         # Q4.2: target compression ratio (original:replay)
    min_replay_length: int = 64         # Q4.2: minimum generated replay tokens

    # Interleaving
    eta_default: int = 4                # Q4.3: old-to-new ratio (4:1 = 80% old, 20% new)
    proximity_fraction: float = 0.5     # Q4.3: fraction of old samples that are proximity-weighted

    # Training
    sleep_optimizer: str = "adamw"      # Q4.4: optimizer for W_cons during sleep
    sleep_weight_decay: float = 0.01    # Q4.4: AdamW weight decay
    grad_clip_norm: float = 1.0         # Q4.4: max gradient norm
    steps_per_memory: int = 5           # Q4.4: gradient steps per consolidated memory
    batch_size: int = 32                # Q4.4: training batch size

    # Triggers
    schedule_interval: int = 10_000     # Q4.5: max steps between sleep cycles
    pressure_threshold: float = 0.7     # Q4.5: tag buffer occupancy triggering sleep
    budget_threshold: float = 0.8       # Q4.5: PRP budget usage triggering sleep
    idle_timeout: int = 300             # Q4.5: seconds of inactivity triggering sleep

    # Validation
    epsilon_learn: float = 0.1          # Q4.6: min surprise reduction for consolidation pass
    theta_validate: float = 0.4         # Q4.6: min cosine sim for consolidation validation
    max_failures: int = 3               # Q4.6: failures before permanent tag removal

    # Curriculum (warmup/consolidate/stabilize split)
    curriculum_warmup: float = 0.3      # Q4.3: fraction of steps at η=9
    curriculum_consolidate: float = 0.5 # Q4.3: fraction of steps at η=4
    # remaining fraction (0.2) at η=9 (stabilize)


@dataclass
class ColdStartConfig:
    """Part 6: Cold Start (Q6.2)"""

    kappa_cold: float = 3.0             # Q6.2: elevated κ during burn-in
    n_burnin: int = 50                  # Q6.2: interactions before normal tagging
    n_ramp: int = 50                    # Q6.2: interactions for threshold relaxation
    n_mature: int = 500                 # Q6.2: interactions for full PRP budget ramp


@dataclass
class RevisionConfig:
    """Part 6: Memory Revision (Q6.1)"""

    w_revision_bonus: float = 0.3       # Q6.1: PRP score bonus for contradiction tags
    replay_multiplier: int = 2          # Q6.1: extra replay frequency for revisions
    contradiction_confidence: float = 0.6  # Q6.1: min confidence for contradiction detection


@dataclass
class SLEEPConfig:
    """Complete SLEEP system configuration."""

    # Model
    model_name: str = "gpt2"            # HuggingFace model ID
    device: str = "cpu"                 # "cpu", "cuda", "cuda:0", etc.

    # Sub-configs
    tagging: TaggingConfig = field(default_factory=TaggingConfig)
    prp: PRPConfig = field(default_factory=PRPConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    cold_start: ColdStartConfig = field(default_factory=ColdStartConfig)
    revision: RevisionConfig = field(default_factory=RevisionConfig)

    # Experiment tracking
    wandb_project: str = "sleep-memory"
    wandb_enabled: bool = True
    log_dir: str = "experiments/results"

    @property
    def model_params_billions(self) -> float:
        """Estimated model size in billions. Set after model loading."""
        return self._model_params_billions

    @model_params_billions.setter
    def model_params_billions(self, value: float):
        self._model_params_billions = value

    @property
    def n_max_tags(self) -> int:
        return int(self.tagging.c_tag * self._model_params_billions)

    @property
    def prp_budget(self) -> int:
        return int(self.prp.c_prp * self._model_params_billions)