"""
Baseline implementations for SLEEP evaluation comparisons.

Baselines from Q5.4 (SLEEP_Formalization.md) plus the three the mentor review
(item #5) flagged as missing from the paper:
    1. RAG baseline — store documents, retrieve by similarity, append to context.
       Expected DRA ~0.9 at all delays (always has access to original document).
    2. Naive LoRA — fine-tune on every input with no tagging, no sleep scheduling.
       Expected to show catastrophic forgetting over many inputs.
    3. EWC-only — naive LoRA plus an Elastic Weight Consolidation penalty, with
       none of the rest of the SLEEP pipeline. Isolates how much of SLEEP's
       preservation advantage is EWC alone vs the full safety machinery.
    4. In-context injection — no training at all; prepend the gold fact to the
       prompt at query time. A cheap reference point for "what if the fact is
       simply in context", and the natural lower bound the paper's RAG
       contrast implies but never measured.

These are intentionally simple and correct — they are comparison points,
not production systems.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sleep.utils.logging import get_logger, metrics

logger = get_logger("sleep.evaluation.baselines")


# ---------------------------------------------------------------------------
# RAG Baseline
# ---------------------------------------------------------------------------

class RAGBaseline:
    """Simple RAG baseline: store documents, retrieve by similarity, append to context.

    Uses mean-pooled last-hidden-state embeddings for document and query
    representation, with cosine similarity for retrieval.

    This baseline should achieve high recall at all delays because it always
    has access to the original document text. It serves as an upper-bound
    comparison for the SLEEP system's consolidation quality.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.documents: list[dict] = []  # {id, text, embedding}

    @torch.no_grad()
    def _embed(self, text: str) -> Tensor:
        """Compute a mean-pooled embedding for a text string.

        Args:
            text: Input text to embed.

        Returns:
            1-D tensor of shape (d_model,).
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        # Use last hidden state, mean-pool over non-padding tokens
        hidden = outputs.hidden_states[-1]  # (1, seq_len, d_model)
        mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.squeeze(0)  # (d_model,)

    def add_document(self, text: str, doc_id: str = "") -> None:
        """Store a document with its embedding.

        Args:
            text:   Document text to store.
            doc_id: Optional identifier for the document.
        """
        embedding = self._embed(text)
        self.documents.append({
            "id": doc_id or f"doc_{len(self.documents)}",
            "text": text,
            "embedding": embedding,
        })
        logger.info(
            "RAG: added document '%s' (%d tokens)",
            doc_id, len(self.tokenizer.encode(text)),
        )

    @torch.no_grad()
    def query(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 100,
    ) -> str:
        """Retrieve relevant docs and generate an answer.

        Retrieves the top-k most similar documents by cosine similarity,
        prepends them to the question as context, and generates a response.

        Args:
            question:       The question to answer.
            top_k:          Number of documents to retrieve.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated response string.
        """
        if not self.documents:
            logger.warning("RAG: no documents stored, generating without context")
            context_text = ""
        else:
            # Compute query embedding and retrieve top-k
            query_emb = self._embed(question)
            similarities = []
            for doc in self.documents:
                sim = F.cosine_similarity(
                    query_emb.unsqueeze(0),
                    doc["embedding"].unsqueeze(0),
                ).item()
                similarities.append((sim, doc))

            similarities.sort(key=lambda x: x[0], reverse=True)
            top_docs = similarities[:top_k]

            context_text = "\n\n".join(
                f"[Document: {doc['id']}]\n{doc['text']}"
                for _, doc in top_docs
            )

        # Build prompt with retrieved context
        if context_text:
            prompt = (
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
        else:
            prompt = f"Question: {question}\nAnswer:"

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to(self.device)

        self.model.eval()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response


# ---------------------------------------------------------------------------
# Naive LoRA Baseline
# ---------------------------------------------------------------------------

class NaiveLoRABaseline:
    """Naive LoRA fine-tuning: update on every input, no tagging or sleep.

    Applies a simple LoRA adapter and fine-tunes on every input text
    immediately. This baseline is expected to show catastrophic forgetting
    over many inputs, as there is no replay interleaving, no EWC
    regularization, and no selective consolidation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Any,
        device: str = "cpu",
    ) -> None:
        """Apply LoRA adapters and set up the optimizer.

        Args:
            model:     A HuggingFace-style causal LM.
            tokenizer: The corresponding tokenizer.
            config:    A :class:`SLEEPConfig` (uses ``weights.lora_rank``,
                       ``weights.lora_alpha``, ``weights.alpha_fast``).
            device:    Device string.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Apply simple LoRA adapters to the model
        self.lora_modules = self._apply_lora(
            rank=config.weights.lora_rank,
            alpha=config.weights.lora_alpha,
        )

        # Freeze all base parameters, only train LoRA
        for param in self.model.parameters():
            param.requires_grad = False
        for module in self.lora_modules:
            for param in module.parameters():
                param.requires_grad = True

        trainable = [p for m in self.lora_modules for p in m.parameters()]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.weights.alpha_fast,
            weight_decay=0.01,
        )

        logger.info(
            "NaiveLoRA: %d LoRA modules, %d trainable parameters",
            len(self.lora_modules),
            sum(p.numel() for p in trainable),
        )

    def _apply_lora(self, rank: int, alpha: int) -> list[nn.Module]:
        """Apply simple LoRA adapters to linear layers.

        This is a minimal LoRA implementation for baseline comparison.
        It wraps selected linear layers with low-rank additive adapters.

        Args:
            rank:  LoRA rank.
            alpha: LoRA scaling factor.

        Returns:
            List of LoRA adapter modules that were created.
        """
        lora_modules: list[nn.Module] = []
        scaling = alpha / rank

        for name, module in self.model.named_modules():
            # Target value and output projections in attention layers
            if any(target in name for target in ["v_proj", "o_proj"]):
                if isinstance(module, nn.Linear):
                    adapter = _LoRAAdapter(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=rank,
                        scaling=scaling,
                        # Match the wrapped layer's dtype as well as device:
                        # a float32 adapter against a bfloat16 model raises
                        # "mat1 and mat2 must have the same dtype" at forward.
                    ).to(device=self.device, dtype=module.weight.dtype)

                    # Hook the adapter into the module's forward
                    _install_lora_hook(module, adapter)
                    lora_modules.append(adapter)

        return lora_modules

    def train_on_input(self, text: str) -> float:
        """Fine-tune on a single input text. Returns the training loss.

        Args:
            text: Input text to train on.

        Returns:
            Cross-entropy loss value (float).
        """
        self.model.train()
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(self.device)

        outputs = self.model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for m in self.lora_modules for p in m.parameters()],
            max_norm=1.0,
        )
        self.optimizer.step()

        loss_val = loss.item()
        logger.debug("NaiveLoRA train step: loss=%.4f", loss_val)
        return loss_val

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate from the fine-tuned model.

        Args:
            prompt:         Input prompt string.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated response string (excluding the prompt).
        """
        self.model.eval()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# EWC-only Baseline (mentor item #5b)
# ---------------------------------------------------------------------------

class EWCOnlyBaseline(NaiveLoRABaseline):
    """Naive LoRA fine-tuning plus an EWC penalty — and nothing else.

    This isolates the contribution of Elastic Weight Consolidation from the
    rest of SLEEP's safety machinery (hard clipping, interleaving, validation
    rollback, tagging, sleep scheduling). If EWC-only already recovers most of
    SLEEP's multi-cycle preservation advantage, the extra machinery is doing
    little; if it does not, the machinery earns its keep.

    The penalty is the standard diagonal-Fisher form,
    ``L_EWC = (lambda/2) * sum_p F_p (theta_p - theta*_p)^2``, anchored to the
    adapter state captured by :meth:`consolidate_task` (call it once after a
    task/batch to snapshot ``theta*`` and estimate ``F`` on that task's data).

    Args:
        model, tokenizer, config, device: As for :class:`NaiveLoRABaseline`.
            Uses ``config.weights.lambda_ewc`` for the penalty strength.
    """

    def __init__(self, model, tokenizer, config, device: str = "cpu") -> None:
        super().__init__(model, tokenizer, config, device)
        self.lambda_ewc: float = float(config.weights.lambda_ewc)
        # Anchors and Fisher, one snapshot per consolidated task. Parameters
        # are referenced by (module_index, "A"/"B") so the mapping is stable.
        self._theta_star: list[dict[str, Tensor]] = []
        self._fisher: list[dict[str, Tensor]] = []

    def _named_adapter_params(self) -> list[tuple[str, nn.Parameter]]:
        """Flat, stably-ordered list of ``("<i>.A"/"<i>.B", param)`` pairs."""
        pairs: list[tuple[str, nn.Parameter]] = []
        for i, module in enumerate(self.lora_modules):
            pairs.append((f"{i}.A", module.A))
            pairs.append((f"{i}.B", module.B))
        return pairs

    @torch.enable_grad()
    def consolidate_task(self, texts: list[str], max_batches: int = 32) -> None:
        """Snapshot ``theta*`` and estimate the diagonal Fisher on ``texts``.

        Call after finishing a task so subsequent tasks are penalised for
        moving parameters that mattered for this one.

        Args:
            texts:       The just-learned task's texts (used to estimate Fisher).
            max_batches: Cap on sequences used for the Fisher estimate.
        """
        params = self._named_adapter_params()
        fisher: dict[str, Tensor] = {
            name: torch.zeros_like(p) for name, p in params
        }
        n = 0
        self.model.eval()
        for text in texts[:max_batches]:
            encoded = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            )
            input_ids = encoded["input_ids"].to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
            for name, p in params:
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
            n += 1
        if n > 0:
            for name in fisher:
                fisher[name] /= n
        self.optimizer.zero_grad()

        self._theta_star.append({name: p.detach().clone() for name, p in params})
        self._fisher.append(fisher)
        logger.info(
            "EWC-only: consolidated task (%d seqs); %d anchors stored",
            n, len(self._theta_star),
        )

    def _ewc_penalty(self) -> Tensor:
        """Sum of Fisher-weighted quadratic penalties across all anchors."""
        params = dict(self._named_adapter_params())
        total = torch.zeros((), device=self.device)
        for theta_star, fisher in zip(self._theta_star, self._fisher):
            for name, p in params.items():
                total = total + (fisher[name] * (p - theta_star[name]) ** 2).sum()
        return 0.5 * self.lambda_ewc * total

    def train_on_input(self, text: str) -> float:
        """Fine-tune on one input with the EWC penalty added to the LM loss."""
        self.model.train()
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        )
        input_ids = encoded["input_ids"].to(self.device)

        lm_loss = self.model(input_ids=input_ids, labels=input_ids).loss
        loss = lm_loss
        if self._theta_star:
            loss = loss + self._ewc_penalty()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for m in self.lora_modules for p in m.parameters()], max_norm=1.0,
        )
        self.optimizer.step()
        return float(lm_loss.item())


# ---------------------------------------------------------------------------
# In-context injection Baseline (mentor item #5c)
# ---------------------------------------------------------------------------

class InContextBaseline:
    """Zero-training reference: prepend the gold fact to the prompt.

    Unlike :class:`RAGBaseline`, there is no retrieval and no embedding — the
    fact is looked up by id and placed directly in context. This is the cheap
    lower bound the paper's RAG framing implies: if the answer is literally in
    the prompt, what recall do we get? Any consolidation method that cannot beat
    "just put it in context" has not earned its complexity.

    Args:
        model, tokenizer, device: Standard.
    """

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.facts: dict[str, str] = {}

    def add_fact(self, fact_id: str, text: str) -> None:
        """Register a fact's text under its id for later in-context lookup."""
        self.facts[fact_id] = text

    @torch.no_grad()
    def query(
        self,
        question: str,
        fact_id: str | None = None,
        fact_text: str | None = None,
        max_new_tokens: int = 50,
    ) -> str:
        """Answer with the gold fact placed in context.

        Args:
            question:       The question to answer.
            fact_id:        Id of a previously-added fact to inject. Ignored if
                            ``fact_text`` is given.
            fact_text:      Explicit fact text to inject (overrides ``fact_id``).
            max_new_tokens: Generation length.

        Returns:
            The generated answer (prompt excluded).
        """
        context = fact_text if fact_text is not None else self.facts.get(fact_id or "", "")
        if context:
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to(self.device)

        self.model.eval()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True,
        )


# ---------------------------------------------------------------------------
# Minimal LoRA adapter (internal)
# ---------------------------------------------------------------------------

class _LoRAAdapter(nn.Module):
    """Minimal low-rank adapter: output += scaling * (x @ A^T @ B^T).

    A is initialized with Kaiming uniform, B is initialized to zero so the
    adapter starts as a no-op.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = scaling
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """Compute the low-rank delta: scaling * x @ A^T @ B^T.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Additive correction of shape (..., out_features).
        """
        return self.scaling * (x @ self.A.T @ self.B.T)


def _install_lora_hook(
    linear: nn.Linear,
    adapter: _LoRAAdapter,
) -> None:
    """Install a forward hook on a linear layer to add the LoRA correction.

    Args:
        linear:  The target nn.Linear module.
        adapter: The LoRA adapter to add.
    """
    def hook(module: nn.Module, input: tuple, output: Tensor) -> Tensor:
        x = input[0]
        return output + adapter(x)

    linear.register_forward_hook(hook)
