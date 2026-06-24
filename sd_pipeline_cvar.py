"""
sd_pipeline_cvar.py

CVaR/tail-controlled non-batched SVDD decoding pipeline.

This file intentionally keeps the non-batched duplicate-generation pattern from
Decoding_nonbatch_SDPipeline, but replaces reward maximization / reward-softmax
with a CVaR cost objective.

Important notation clash:
    - `eta` in the __call__ signature is the DDIM eta used by the scheduler.
    - `cvar_eta` is the CVaR threshold eta from the tail objective.

For PM mode, no value-network training is needed:
    x0_hat = E[x0 | x_t] approximated by predict_x0_from_xt(...)
    cost  = - reward(x0_hat)
    V_eta(x_t) ~= ((cost - cvar_eta)^+) / (1 - beta)
    log weight = - V_eta(x_t) / alpha
               = - ((cost - cvar_eta)^+) / (alpha * (1 - beta))

MC mode is left as a TODO hook for a trained value/log-weight model.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from diffusers import StableDiffusionPipeline
from PIL import Image

from diffusers_patch.ddim_with_kl import ddim_step_KL, predict_x0_from_xt


ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...], torch.Tensor]


def _as_numpy_1d(x: ArrayLike, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert tensor/list/array scorer output to a finite 1D numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=dtype).reshape(-1)
    return np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)


def logmeanexp(log_values: np.ndarray) -> float:
    """Numerically stable log(mean(exp(log_values)))."""
    log_values = np.asarray(log_values, dtype=np.float32).reshape(-1)
    if log_values.size == 0:
        raise ValueError("logmeanexp received an empty array.")
    m = np.max(log_values)
    return float(m + np.log(np.mean(np.exp(log_values - m))))


def cvar_eta_time0_objective(
    eta_value: float,
    costs: ArrayLike,
    alpha: float,
    beta: float,
) -> float:
    """
    Empirical time-0 objective for the CVaR/tail-controlled eta.

        J(eta) = eta - alpha * log E_pre[ exp(-((c(x0)-eta)^+) / (alpha*(1-beta))) ]

    The expectation is approximated by an empirical average over pre-trained
    samples' costs.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must satisfy 0 <= beta < 1.")

    costs_np = _as_numpy_1d(costs)
    denom = alpha * (1.0 - beta)
    log_terms = -np.maximum(costs_np - float(eta_value), 0.0) / denom
    return float(eta_value - alpha * logmeanexp(log_terms))


def solve_cvar_eta_time0_from_costs(
    costs: ArrayLike,
    alpha: float,
    beta: float,
    grid_size: int = 2001,
    eta_bounds: Optional[Tuple[float, float]] = None,
    include_cost_knots: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Solve the empirical time-0 eta optimization using a global grid search.

    The notes indicate the eta problem can be non-convex, so this function avoids
    assuming convexity.  By default it searches the empirical cost range; for the
    empirical objective, a minimizer lies in [min(costs), max(costs)].

    Returns:
        eta_hat, info
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must satisfy 0 <= beta < 1.")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")

    costs_np = _as_numpy_1d(costs)
    if costs_np.size == 0:
        raise ValueError("Need at least one pre-trained cost sample to solve eta.")

    finite_costs = costs_np[np.isfinite(costs_np)]
    if finite_costs.size == 0:
        raise ValueError("All cost samples are non-finite.")

    if eta_bounds is None:
        lo = float(np.min(finite_costs))
        hi = float(np.max(finite_costs))
    else:
        lo, hi = map(float, eta_bounds)
        if lo > hi:
            lo, hi = hi, lo

    if np.isclose(lo, hi):
        eta_hat = 0.5 * (lo + hi)
        obj = cvar_eta_time0_objective(eta_hat, finite_costs, alpha, beta)
        return eta_hat, {
            "objective": obj,
            "num_cost_samples": int(finite_costs.size),
            "cost_min": float(np.min(finite_costs)),
            "cost_max": float(np.max(finite_costs)),
            "grid_size": int(grid_size),
            "eta_bounds": (lo, hi),
        }

    grid = np.linspace(lo, hi, int(grid_size), dtype=np.float32)
    if include_cost_knots:
        # Add the empirical kink locations. This is cheap and makes the search
        # more stable when grid_size is small.
        grid = np.unique(np.concatenate([grid, finite_costs]))

    objectives = np.asarray(
        [cvar_eta_time0_objective(eta_val, finite_costs, alpha, beta) for eta_val in grid],
        dtype=np.float32,
    )
    best_idx = int(np.argmin(objectives))
    eta_hat = float(grid[best_idx])
    best_obj = float(objectives[best_idx])

    return eta_hat, {
        "objective": best_obj,
        "num_cost_samples": int(finite_costs.size),
        "cost_min": float(np.min(finite_costs)),
        "cost_max": float(np.max(finite_costs)),
        "cost_mean": float(np.mean(finite_costs)),
        "cost_std": float(np.std(finite_costs)),
        "grid_size": int(grid.size),
        "eta_bounds": (lo, hi),
        "best_grid_index": best_idx,
    }


class Decoding_nonbatch_SDPipeline_CVaR(StableDiffusionPipeline):
    """
    Non-batched SVDD decoding with CVaR/tail-controlled resampling.

    Compared with Decoding_nonbatch_SDPipeline, this class changes the duplicate
    selection score.  It samples a duplicate using

        p(candidate) proportional to exp(-((cost(candidate)-cvar_eta)^+) /
                                       (alpha * (1-beta)))

    where cost = -reward.
    """

    # ---------------------------------------------------------------------
    # Basic configuration helpers
    # ---------------------------------------------------------------------

    def _ensure_cvar_defaults(self) -> None:
        if not hasattr(self, "duplicate"):
            self.duplicate = 1
        if not hasattr(self, "batch_size"):
            self.batch_size = None
        if not hasattr(self, "alpha"):
            self.alpha = 10.0
        if not hasattr(self, "beta"):
            self.beta = 0.8
        if not hasattr(self, "cvar_eta"):
            self.cvar_eta = None
        if not hasattr(self, "variant"):
            self.variant = "PM"

    @staticmethod
    def _validate_alpha_beta(alpha: float, beta: float) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must satisfy 0 <= beta < 1.")

    def set_parameters(
        self,
        batch_size: int,
        duplicate_size: int,
        alpha: float = 10.0,
        beta: float = 0.8,
        cvar_eta: Optional[float] = None,
    ) -> None:
        self._validate_alpha_beta(alpha, beta)
        if duplicate_size < 1:
            raise ValueError("duplicate_size must be at least 1.")
        self.batch_size = batch_size
        self.duplicate = int(duplicate_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.cvar_eta = None if cvar_eta is None else float(cvar_eta)

    def set_cvar_eta(self, cvar_eta: float) -> None:
        self.cvar_eta = float(cvar_eta)

    def set_cvar_beta(self, beta: float) -> None:
        self._ensure_cvar_defaults()
        self._validate_alpha_beta(float(self.alpha), float(beta))
        self.beta = float(beta)

    def setup_oracle(self, scorer: Any) -> None:
        self.scorer = scorer

    def setup_scorer(self, scorer: Any) -> None:
        self.scorer = scorer
        if hasattr(self.scorer, "requires_grad_"):
            self.scorer.requires_grad_(False)
        if hasattr(self.scorer, "eval"):
            self.scorer.eval()

    def setup_cvar_value_model(self, value_model: Any) -> None:
        """Placeholder hook for future MC-mode value/log-weight model."""
        self.cvar_value_model = value_model
        if hasattr(self.cvar_value_model, "requires_grad_"):
            self.cvar_value_model.requires_grad_(False)
        if hasattr(self.cvar_value_model, "eval"):
            self.cvar_value_model.eval()

    def set_reward(self, reward: str) -> None:
        self.reward = reward

    def set_target(self, target: Any) -> None:
        self.target = target

    def set_guidance(self, guidance: Any) -> None:
        self.target_guidance = guidance

    def set_variant(self, variant: str) -> None:
        if variant not in {"PM", "MC"}:
            raise ValueError("variant must be 'PM' or 'MC'.")
        self.variant = variant

    # ---------------------------------------------------------------------
    # Eta solving from pre-trained samples
    # ---------------------------------------------------------------------

    def solve_cvar_eta_from_costs(
        self,
        costs: ArrayLike,
        grid_size: int = 2001,
        eta_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve and store cvar_eta from already-computed pre-trained costs."""
        self._ensure_cvar_defaults()
        eta_hat, info = solve_cvar_eta_time0_from_costs(
            costs=costs,
            alpha=float(self.alpha),
            beta=float(self.beta),
            grid_size=grid_size,
            eta_bounds=eta_bounds,
        )
        self.cvar_eta = float(eta_hat)
        self.cvar_eta_info = info
        self.pretrained_cost_samples = _as_numpy_1d(costs)
        return self.cvar_eta, info

    @torch.no_grad()
    def solve_cvar_eta_from_pretrained(
        self,
        prompt: Union[str, List[str]],
        num_pretrained_samples: int = 64,
        pretrained_batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        grid_size: int = 2001,
        eta_bounds: Optional[Tuple[float, float]] = None,
        disable_safety_checker: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Approximate E_pre[...] by sampling from the unmodified pre-trained
        StableDiffusionPipeline, scoring those images, negating rewards to costs,
        and solving the empirical time-0 eta problem.
        """
        self._ensure_cvar_defaults()
        costs = self.estimate_pretrained_costs(
            prompt=prompt,
            num_pretrained_samples=num_pretrained_samples,
            pretrained_batch_size=pretrained_batch_size,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            disable_safety_checker=disable_safety_checker,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        eta_hat, info = self.solve_cvar_eta_from_costs(
            costs=costs,
            grid_size=grid_size,
            eta_bounds=eta_bounds,
        )
        info["num_pretrained_samples_requested"] = int(num_pretrained_samples)
        return eta_hat, info

    @torch.no_grad()
    def estimate_pretrained_costs(
        self,
        prompt: Union[str, List[str]],
        num_pretrained_samples: int = 64,
        pretrained_batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        disable_safety_checker: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Generate pre-trained samples and return cost = -reward for each image."""
        self._ensure_cvar_defaults()
        if num_pretrained_samples < 1:
            raise ValueError("num_pretrained_samples must be at least 1.")
        if pretrained_batch_size is None:
            pretrained_batch_size = min(num_pretrained_samples, self.batch_size or num_pretrained_samples)
        pretrained_batch_size = int(max(1, pretrained_batch_size))

        cost_chunks: List[np.ndarray] = []
        made = 0
        with self._maybe_disable_safety_checker(disable_safety_checker):
            while made < num_pretrained_samples:
                batch_n = min(pretrained_batch_size, num_pretrained_samples - made)
                batch_prompt = self._make_repeated_batch(prompt, batch_n, offset=made)
                batch_negative_prompt = self._make_repeated_batch(negative_prompt, batch_n, offset=made)
                batch_generator = self._slice_generator(generator, made, batch_n)

                # Important: bypass this subclass and call the original
                # StableDiffusionPipeline sampling path.
                output = StableDiffusionPipeline.__call__(
                    self,
                    prompt=batch_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=batch_negative_prompt,
                    num_images_per_prompt=1,
                    eta=0.0,
                    generator=batch_generator,
                    output_type="np",
                    return_dict=True,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
                images = output.images if hasattr(output, "images") else output[0]
                rewards = self.reward_from_images(images)
                costs = -_as_numpy_1d(rewards)
                cost_chunks.append(costs)
                made += batch_n

        costs_all = np.concatenate(cost_chunks, axis=0)[:num_pretrained_samples]
        self.pretrained_cost_samples = costs_all
        return costs_all

    @staticmethod
    def _make_repeated_batch(
        value: Optional[Union[str, List[str]]],
        batch_n: int,
        offset: int = 0,
    ) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value] * batch_n
        if len(value) == 0:
            raise ValueError("Prompt/negative_prompt list cannot be empty.")
        return [value[(offset + i) % len(value)] for i in range(batch_n)]

    @staticmethod
    def _slice_generator(
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        offset: int,
        batch_n: int,
    ) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
        if isinstance(generator, list):
            return generator[offset : offset + batch_n]
        return generator

    @contextmanager
    def _maybe_disable_safety_checker(self, disable: bool):
        if not disable:
            yield
            return
        old_safety_checker = getattr(self, "safety_checker", None)
        try:
            self.safety_checker = None
            yield
        finally:
            self.safety_checker = old_safety_checker

    # ---------------------------------------------------------------------
    # Reward/cost calculation
    # ---------------------------------------------------------------------

    def reward_from_images(self, images: Union[np.ndarray, List[Image.Image]]) -> np.ndarray:
        """Score final generated images as rewards.  Costs are -rewards."""
        if not hasattr(self, "scorer"):
            raise RuntimeError("Call setup_scorer(...) before scoring images.")
        if not hasattr(self, "reward"):
            raise RuntimeError("Call set_reward('compressibility' or 'aesthetic') first.")

        if self.reward == "compressibility":
            images_uint8 = self._images_to_uint8_numpy(images)
            rewards = self.scorer(images_uint8)
            return _as_numpy_1d(rewards)

        if self.reward == "aesthetic":
            im_pix = self._images_to_clip_tensor(images, size=224)
            rewards, _ = self.scorer(im_pix)
            return _as_numpy_1d(rewards)

        raise ValueError("Invalid reward type. Expected 'compressibility' or 'aesthetic'.")

    def calculate_pm_reward(
        self,
        latents: torch.FloatTensor,
        new_noise_pred: torch.FloatTensor,
        t: torch.Tensor,
    ) -> np.ndarray:
        """PM reward r(E[x0|x_t]) using predict_x0_from_xt; no extra training."""
        if not hasattr(self, "scorer"):
            raise RuntimeError("Call setup_scorer(...) before calculate_pm_reward(...).")
        if not hasattr(self, "reward"):
            raise RuntimeError("Call set_reward('compressibility' or 'aesthetic') first.")

        pred_original_sample = predict_x0_from_xt(
            self.scheduler,
            new_noise_pred,
            t,
            latents,
        )

        if self.reward == "compressibility":
            images = self.decode_latents(pred_original_sample)
            images = (images * 255).round().astype("uint8")
            rewards = self.scorer(images)
            return _as_numpy_1d(rewards)

        if self.reward == "aesthetic":
            im_pix_un = self.vae.decode(pred_original_sample.to(self.vae.dtype) / 0.18215).sample
            im_pix = ((im_pix_un / 2.0) + 0.5).clamp(0.0, 1.0)
            resize = torchvision.transforms.Resize(224, antialias=False)
            im_pix = resize(im_pix)
            normalize = torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            im_pix = normalize(im_pix).to(im_pix_un.dtype)
            rewards, _ = self.scorer(im_pix)
            return _as_numpy_1d(rewards)

        raise ValueError("Invalid reward type. Expected 'compressibility' or 'aesthetic'.")

    def calculate_pm_cost(
        self,
        latents: torch.FloatTensor,
        new_noise_pred: torch.FloatTensor,
        t: torch.Tensor,
    ) -> np.ndarray:
        """All rewards are negated to costs: c = -r."""
        rewards = self.calculate_pm_reward(latents=latents, new_noise_pred=new_noise_pred, t=t)
        return -rewards

    @torch.no_grad()
    def calculate_mc_cvar_value(
        self,
        latents: torch.FloatTensor,
        new_noise_pred: torch.FloatTensor,
        t: torch.Tensor,
        cvar_eta: float,
    ) -> np.ndarray:
        """
        MC-mode CVaR value estimated using the same scorer input convention as
        the risk-neutral MC setting.

        Risk-neutral MC logic:
            compressibility: scorer(latents, timesteps=t)
            aesthetic:       scorer(decoded_latent_image, timesteps=t)

        Here:
            reward = same MC reward estimate
            cost   = -reward
            value  = ((cost - cvar_eta)^+) / (1 - beta)

        new_noise_pred is unused in MC mode, but kept in the signature so PM and
        MC share the same calculate_cvar_value(...) interface.
        """
        if not hasattr(self, "scorer"):
            raise RuntimeError("Call setup_scorer(...) before calculate_mc_cvar_value(...).")
        if not hasattr(self, "reward"):
            raise RuntimeError("Call set_reward('compressibility' or 'aesthetic') first.")

        batch_n = latents.shape[0]

        # Make timesteps shape [batch_n], matching the original MC scorer call.
        if not isinstance(t, torch.Tensor):
            timesteps = torch.tensor(t, device=latents.device)
        else:
            timesteps = t.to(latents.device)

        if timesteps.ndim == 0:
            timesteps = timesteps.repeat(batch_n)
        elif timesteps.numel() == 1:
            timesteps = timesteps.reshape(1).repeat(batch_n)
        elif timesteps.numel() == batch_n:
            timesteps = timesteps.reshape(batch_n)
        else:
            raise ValueError(
                f"Expected timestep to be scalar or length {batch_n}, "
                f"but got shape {tuple(timesteps.shape)}."
            )

        if self.reward == "compressibility":
            rewards, _ = self.scorer(latents, timesteps=timesteps)

        elif self.reward == "aesthetic":
            im_pix_un = self.vae.decode(latents.to(self.vae.dtype) / 0.18215).sample
            im_pix = ((im_pix_un / 2.0) + 0.5).clamp(0.0, 1.0)

            resize = torchvision.transforms.Resize(224, antialias=False)
            im_pix = resize(im_pix)

            normalize = torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            im_pix = normalize(im_pix).to(im_pix_un.dtype)

            rewards, _ = self.scorer(im_pix, timesteps=timesteps)

        else:
            raise ValueError("Invalid reward type. Expected 'compressibility' or 'aesthetic'.")

        costs = -_as_numpy_1d(rewards)
        excess = np.maximum(costs - float(cvar_eta), 0.0)
        values = excess / (1.0 - float(self.beta))

        return values

    def calculate_cvar_value(
        self,
        latents: torch.FloatTensor,
        new_noise_pred: torch.FloatTensor,
        t: torch.Tensor,
        cvar_eta: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return V_{t,eta}(x_t).  In PM mode:
            V ~= ((c(E[x0|x_t]) - eta)^+) / (1 - beta)
        """
        self._ensure_cvar_defaults()
        self._validate_alpha_beta(float(self.alpha), float(self.beta))
        if cvar_eta is None:
            cvar_eta = self.cvar_eta
        if cvar_eta is None:
            raise RuntimeError("cvar_eta is not set. Call solve_cvar_eta_from_pretrained(...) or set_cvar_eta(...).")

        if self.variant == "PM":
            costs = self.calculate_pm_cost(latents=latents, new_noise_pred=new_noise_pred, t=t)
            excess = np.maximum(costs - float(cvar_eta), 0.0)
            return excess / (1.0 - float(self.beta))

        if self.variant == "MC":
            return self.calculate_mc_cvar_value(
                    latents=latents,
                    new_noise_pred=new_noise_pred,
                    t=t,
                    cvar_eta=float(cvar_eta),
                )

        raise ValueError("variant must be 'PM' or 'MC'.")

    def calculate_cvar_log_weight(
        self,
        latents: torch.FloatTensor,
        new_noise_pred: torch.FloatTensor,
        t: torch.Tensor,
        cvar_eta: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return log weights for resampling.

        Correct sign/convention for minimization:
            log w = - V / alpha
                  = - ((c - eta)^+) / (alpha * (1 - beta))   in PM mode.

        Therefore the positive quantity ((c-eta)^+) / (alpha*(1-beta)) is a
        penalty; the sampling log-weight is its negative.
        """
        values = self.calculate_cvar_value(
            latents=latents,
            new_noise_pred=new_noise_pred,
            t=t,
            cvar_eta=cvar_eta,
        )
        return - values / float(self.alpha)

    @staticmethod
    def log_weights_to_probs(log_weights: ArrayLike) -> np.ndarray:
        """Normalize log-weights into a probability vector."""
        lw = _as_numpy_1d(log_weights)
        if lw.size == 0:
            raise ValueError("Cannot normalize empty log_weights.")
        lw = np.nan_to_num(lw, nan=-1e9, posinf=1e9, neginf=-1e9)
        m = np.max(lw)
        weights = np.exp(lw - m)
        total = np.sum(weights)
        if total <= 0.0 or not np.isfinite(total):
            return np.ones_like(weights, dtype=np.float32) / float(weights.size)
        return weights / total

    @staticmethod
    def _images_to_uint8_numpy(images: Union[np.ndarray, List[Image.Image]]) -> np.ndarray:
        arr = CVaR_Decoding_nonbatch_SDPipeline._images_to_float_numpy(images)
        return (arr * 255.0).round().clip(0, 255).astype("uint8")

    @staticmethod
    def _images_to_float_numpy(images: Union[np.ndarray, List[Image.Image]]) -> np.ndarray:
        if isinstance(images, np.ndarray):
            arr = images
        elif isinstance(images, list):
            arrays = []
            for image in images:
                if isinstance(image, Image.Image):
                    arrays.append(np.asarray(image.convert("RGB")))
                else:
                    arrays.append(np.asarray(image))
            arr = np.stack(arrays, axis=0)
        else:
            raise TypeError("images must be a numpy array or a list of PIL images.")

        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.shape[-1] != 3:
            raise ValueError("Expected images in NHWC format with 3 channels.")

        arr = arr.astype(np.float32)
        if float(np.max(arr)) > 1.5:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def _images_to_clip_tensor(self, images: Union[np.ndarray, List[Image.Image]], size: int) -> torch.Tensor:
        arr = self._images_to_float_numpy(images)
        device = self._execution_device
        dtype = getattr(self.vae, "dtype", torch.float32)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        resize = torchvision.transforms.Resize(size, antialias=False)
        tensor = resize(tensor)
        normalize = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        return normalize(tensor).to(dtype)

    # ---------------------------------------------------------------------
    # Main CVaR non-batched decoding call
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        cvar_eta: Optional[float] = None,
        auto_solve_cvar_eta: bool = False,
        pretrained_costs: Optional[ArrayLike] = None,
        num_pretrained_samples: int = 64,
        pretrained_batch_size: Optional[int] = None,
        eta_grid_size: int = 2001,
    ):
        """
        Generate images using CVaR/tail-controlled non-batched decoding.

        Args added relative to Decoding_nonbatch_SDPipeline:
            cvar_eta: CVaR threshold eta. If None, uses self.cvar_eta.
            auto_solve_cvar_eta: If True, solve eta before decoding.  If
                pretrained_costs is provided, solve from those; otherwise sample
                from the pre-trained StableDiffusionPipeline.
            pretrained_costs: Optional precomputed costs from pre-trained samples.
            num_pretrained_samples: Number of pre-trained samples for eta solving.
            pretrained_batch_size: Batch size for pre-trained eta samples.
            eta_grid_size: Grid size for empirical eta search.

        Note: the existing argument `eta` remains DDIM eta, not CVaR eta.
        """
        self._ensure_cvar_defaults()
        self._validate_alpha_beta(float(self.alpha), float(self.beta))

        # 0. Default height and width to unet.
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs.
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # Optional time-0 eta solving before the controlled rollout.
        print("Obtaining CVaR_eta...")

        if cvar_eta is not None:
            self.cvar_eta = float(cvar_eta)
        elif auto_solve_cvar_eta:
            if pretrained_costs is not None:
                self.solve_cvar_eta_from_costs(pretrained_costs, grid_size=eta_grid_size)
            else:
                if prompt is None:
                    raise ValueError("auto_solve_cvar_eta=True with no pretrained_costs requires prompt, not only prompt_embeds.")
                self.solve_cvar_eta_from_pretrained(
                    prompt=prompt,
                    num_pretrained_samples=num_pretrained_samples,
                    pretrained_batch_size=pretrained_batch_size,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    grid_size=eta_grid_size,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        if self.cvar_eta is None:
            raise RuntimeError(
                "cvar_eta is not set. Call solve_cvar_eta_from_pretrained(...), "
                "solve_cvar_eta_from_costs(...), set_cvar_eta(...), or pass cvar_eta=... ."
            )

        print("CVaR_eta:", self.cvar_eta)

        # 2. Define call parameters.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt.
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables.
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        num_particles = latents.shape[0]

        # Keep the call for compatibility with the original pipeline even though
        # ddim_step_KL consumes eta directly below.
        _ = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop.
        kl_loss = 0
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    old_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    old_noise_pred = noise_pred

                noise_pred = old_noise_pred

                if step_index < len(timesteps) - 1:
                    log_weights_list: List[np.ndarray] = []
                    latents_list: List[np.ndarray] = []
                    latent_dtype = latents.dtype

                    for _dup_index in range(int(self.duplicate)):
                        latents_duplicate, kl_terms = ddim_step_KL(
                            self.scheduler,
                            noise_pred,
                            old_noise_pred,
                            t,
                            latents,
                            eta=eta,
                        )
                        kl_loss += torch.mean(kl_terms)

                        duplicate_model_input = (
                            torch.cat([latents_duplicate] * 2)
                            if do_classifier_free_guidance
                            else latents_duplicate
                        )
                        duplicate_model_input = self.scheduler.scale_model_input(
                            duplicate_model_input,
                            timesteps[step_index + 1],
                        )

                        noise_pred_duplicate = self.unet(
                            duplicate_model_input,
                            timesteps[step_index + 1],
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred_duplicate.chunk(2)
                            new_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        else:
                            new_noise_pred = noise_pred_duplicate

                        log_weights = self.calculate_cvar_log_weight(
                            latents=latents_duplicate,
                            new_noise_pred=new_noise_pred,
                            t=timesteps[step_index + 1],
                            cvar_eta=self.cvar_eta,
                        )

                        log_weights_list.append(log_weights)
                        latents_list.append(latents_duplicate.detach().cpu().numpy())

                    # Shapes:
                    #   log_weights_array: [duplicate, num_particles]
                    #   latents_array:     [duplicate, num_particles, C, H, W]
                    log_weights_array = np.asarray(log_weights_list, dtype=np.float32)
                    latents_array = np.asarray(latents_list)

                    index_chosen: List[int] = []
                    for particle_index in range(num_particles):
                        logw = log_weights_array[:, particle_index]
                        probs = self.log_weights_to_probs(logw)
                        chosen_dup = int(np.random.choice(int(self.duplicate), p=probs))
                        index_chosen.append(chosen_dup)

                    selected_latents = np.stack(
                        [
                            latents_array[index_chosen[particle_index], particle_index, :, :, :]
                            for particle_index in range(num_particles)
                        ],
                        axis=0,
                    )
                    latents = torch.from_numpy(selected_latents).to(device=device, dtype=latent_dtype)

                else:
                    latents, kl_terms = ddim_step_KL(
                        self.scheduler,
                        noise_pred,
                        old_noise_pred,
                        t,
                        latents,
                        eta=eta,
                    )
                    kl_loss += torch.mean(kl_terms)

                if step_index == len(timesteps) - 1 or (
                    (step_index + 1) > num_warmup_steps
                    and (step_index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and step_index % callback_steps == 0:
                        callback(step_index, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
            has_nsfw_concept = False
        else:
            image = self.decode_latents(latents)
            has_nsfw_concept = False

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return image, has_nsfw_concept

        return image, kl_loss

    @torch.no_grad()
    def sample_max(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        cvar_eta: Optional[float] = None,
        auto_solve_cvar_eta: bool = False,
        pretrained_costs: Optional[ArrayLike] = None,
        num_pretrained_samples: int = 64,
        pretrained_batch_size: Optional[int] = None,
        eta_grid_size: int = 2001,
    ):
        """
        Generate images using CVaR/tail-controlled non-batched decoding.

        Args added relative to Decoding_nonbatch_SDPipeline:
            cvar_eta: CVaR threshold eta. If None, uses self.cvar_eta.
            auto_solve_cvar_eta: If True, solve eta before decoding.  If
                pretrained_costs is provided, solve from those; otherwise sample
                from the pre-trained StableDiffusionPipeline.
            pretrained_costs: Optional precomputed costs from pre-trained samples.
            num_pretrained_samples: Number of pre-trained samples for eta solving.
            pretrained_batch_size: Batch size for pre-trained eta samples.
            eta_grid_size: Grid size for empirical eta search.

        Note: the existing argument `eta` remains DDIM eta, not CVaR eta.
        """
        self._ensure_cvar_defaults()
        self._validate_alpha_beta(float(self.alpha), float(self.beta))

        # 0. Default height and width to unet.
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs.
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # Optional time-0 eta solving before the controlled rollout.
        print("Obtaining CVaR_eta...")

        if cvar_eta is not None:
            self.cvar_eta = float(cvar_eta)
        elif auto_solve_cvar_eta:
            if pretrained_costs is not None:
                self.solve_cvar_eta_from_costs(pretrained_costs, grid_size=eta_grid_size)
            else:
                if prompt is None:
                    raise ValueError("auto_solve_cvar_eta=True with no pretrained_costs requires prompt, not only prompt_embeds.")
                self.solve_cvar_eta_from_pretrained(
                    prompt=prompt,
                    num_pretrained_samples=num_pretrained_samples,
                    pretrained_batch_size=pretrained_batch_size,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    grid_size=eta_grid_size,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        if self.cvar_eta is None:
            raise RuntimeError(
                "cvar_eta is not set. Call solve_cvar_eta_from_pretrained(...), "
                "solve_cvar_eta_from_costs(...), set_cvar_eta(...), or pass cvar_eta=... ."
            )

        print("CVaR_eta:", self.cvar_eta)

        # 2. Define call parameters.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt.
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables.
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        num_particles = latents.shape[0]

        # Keep the call for compatibility with the original pipeline even though
        # ddim_step_KL consumes eta directly below.
        _ = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop.
        kl_loss = 0
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    old_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    old_noise_pred = noise_pred

                noise_pred = old_noise_pred

                if step_index < len(timesteps) - 1:
                    log_weights_list: List[np.ndarray] = []
                    latents_list: List[np.ndarray] = []
                    latent_dtype = latents.dtype

                    for _dup_index in range(int(self.duplicate)):
                        latents_duplicate, kl_terms = ddim_step_KL(
                            self.scheduler,
                            noise_pred,
                            old_noise_pred,
                            t,
                            latents,
                            eta=eta,
                        )
                        kl_loss += torch.mean(kl_terms)

                        duplicate_model_input = (
                            torch.cat([latents_duplicate] * 2)
                            if do_classifier_free_guidance
                            else latents_duplicate
                        )
                        duplicate_model_input = self.scheduler.scale_model_input(
                            duplicate_model_input,
                            timesteps[step_index + 1],
                        )

                        noise_pred_duplicate = self.unet(
                            duplicate_model_input,
                            timesteps[step_index + 1],
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred_duplicate.chunk(2)
                            new_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        else:
                            new_noise_pred = noise_pred_duplicate

                        values = self.calculate_cvar_value(
                                latents=latents_duplicate,
                                new_noise_pred=new_noise_pred,
                                t=timesteps[step_index + 1],
                                cvar_eta=self.cvar_eta,
                            )

                        values_list.append(values)
                        latents_list.append(latents_duplicate.detach().cpu().numpy())

                    # Shapes:
                    #   log_weights_array: [duplicate, num_particles]
                    #   latents_array:     [duplicate, num_particles, C, H, W]
                    values_array = np.asarray(values_list, dtype=np.float32)
                    latents_array = np.asarray(latents_list)

                    index_chosen: List[int] = []
                    for particle_index in range(num_particles):
                        values_b = values_array[:, particle_index]

                        # Cost-side version of reward argmax:
                        # reward max <=> cost/CVaR-value min.
                        chosen_dup = int(np.argmin(values_b))

                        index_chosen.append(chosen_dup)

                    selected_latents = np.stack(
                        [
                            latents_array[index_chosen[particle_index], particle_index, :, :, :]
                            for particle_index in range(num_particles)
                        ],
                        axis=0,
                    )
                    latents = torch.from_numpy(selected_latents).to(device=device, dtype=latent_dtype)

                else:
                    latents, kl_terms = ddim_step_KL(
                        self.scheduler,
                        noise_pred,
                        old_noise_pred,
                        t,
                        latents,
                        eta=eta,
                    )
                    kl_loss += torch.mean(kl_terms)

                if step_index == len(timesteps) - 1 or (
                    (step_index + 1) > num_warmup_steps
                    and (step_index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and step_index % callback_steps == 0:
                        callback(step_index, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
            has_nsfw_concept = False
        else:
            image = self.decode_latents(latents)
            has_nsfw_concept = False

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return image, has_nsfw_concept

        return image, kl_loss
