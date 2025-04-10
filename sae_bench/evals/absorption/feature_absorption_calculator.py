from dataclasses import dataclass

import numpy as np
import torch
from sae_lens import SAE
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_bench.evals.absorption.prompting import (
    Formatter,
    SpellingPrompt,
    create_icl_prompt,
    first_letter_formatter,
)
from sae_bench.evals.absorption.util import batchify

EPS = 1e-8


@dataclass
class FeatureScore:
    feature_id: int
    activation: float
    probe_cos_sim: float

    @property
    def probe_projection(self) -> float:
        return self.activation * self.probe_cos_sim


@dataclass
class WordAbsorptionResult:
    word: str
    prompt: str
    probe_projection: float
    main_feature_scores: list[FeatureScore]
    top_projection_feature_scores: list[FeatureScore]
    absorption_fraction: float
    is_full_absorption: bool


@dataclass
class AbsorptionResults:
    main_feature_ids: list[int]
    word_results: list[WordAbsorptionResult]


@dataclass
class FeatureAbsorptionCalculator:
    """
    Feature absorption calculator for spelling tasks.

    Absorption is defined by the following criteria:
    - The main features for a concept do not fire
    - The top feature is aligned with a probe trained on that concept
    - The top feature contributes a significant portion of the total activation probe projection
    """

    model: HookedTransformer
    icl_word_list: list[str]
    max_icl_examples: int | None = None
    base_template: str = "{word}:"
    answer_formatter: Formatter = first_letter_formatter()
    example_separator: str = "\n"
    shuffle_examples: bool = True
    # Mistral tokenizer handles the first token incorrectly if we don't do this
    prepend_separator_to_first_example: bool = True
    # the position to read activations from (depends on the template)
    word_token_pos: int = -2
    batch_size: int = 10
    topk_feats: int = 10

    # the cosine similarity between the top projecting feature and the probe must be at least this high to count as absorption (full absorption only)
    full_absorption_probe_cos_sim_threshold: float = 0.025
    # the cosine similarity between each potential absorbing latent and the probe must be at least this high to count as absorption (absorption fraction only)
    absorption_fraction_probe_cos_sim_threshold: float = 0.1
    # the total probe projection of the potential absorbing latents must contribute at least this much to the probe projection to count as absorption (both absorption metrics)
    probe_projection_proportion_threshold: float = 0.4
    # the maximum number of latents that can be considered to collectively compensate for the reduced activation of a potentially absorbed latent (absorption fraction only)
    absorption_fraction_max_absorbing_latents: int = 3

    def _build_prompts(self, words: list[str]) -> list[SpellingPrompt]:
        return [
            create_icl_prompt(
                word,
                examples=self.icl_word_list,
                base_template=self.base_template,
                answer_formatter=self.answer_formatter,
                example_separator=self.example_separator,
                max_icl_examples=self.max_icl_examples,
                shuffle_examples=self.shuffle_examples,
                prepend_separator_to_first_example=self.prepend_separator_to_first_example,
            )
            for word in words
        ]

    def _is_full_absorption(
        self,
        probe_projection: float,
        main_feature_scores: list[FeatureScore],
        top_projection_feature_scores: list[FeatureScore],
    ) -> bool:
        # if any of the main features fired, this isn't absorption
        if not all(score.activation < EPS for score in main_feature_scores):
            return False
        # If the top firing feature isn't aligned with the probe, this isn't absorption
        if (
            top_projection_feature_scores[0].probe_cos_sim
            < self.full_absorption_probe_cos_sim_threshold
        ):
            return False
        # If the probe isn't even activated, this can't be absorption
        if probe_projection < 0:
            return False
        # If the top firing feature doesn't contribute much to the total probe projection, this isn't absorption
        proj_proportion = (
            top_projection_feature_scores[0].probe_projection / probe_projection
        )
        if proj_proportion < self.probe_projection_proportion_threshold:
            return False
        return True

    @torch.inference_mode()
    def calculate_absorption(
        self,
        sae: SAE,
        words: list[str],
        probe_direction: torch.Tensor,
        main_feature_ids: list[int],
        layer: int,
        show_progress: bool = True,
    ) -> AbsorptionResults:
        """
        This method calculates the absorption for each word in the list of words
        """
        if probe_direction.ndim != 1:
            raise ValueError("probe_direction must be 1D")
        # make sure the probe direction is a unit vector
        probe_direction = probe_direction / probe_direction.norm()
        prompts = self._build_prompts(words)
        self._validate_prompts_are_same_length(prompts)
        results: list[WordAbsorptionResult] = []
        cos_sims = (
            torch.nn.functional.cosine_similarity(
                probe_direction.to(sae.device), sae.W_dec, dim=-1
            )
            .float()
            .cpu()
        )
        hook_point = f"blocks.{layer}.hook_resid_post"
        for batch_prompts in batchify(prompts, batch_size=self.batch_size):
            batch_acts = self.model.run_with_cache(
                [p.base for p in batch_prompts],
                names_filter=[hook_point],
            )[1][hook_point][:, self.word_token_pos, :]
            batch_sae_acts = sae.encode(batch_acts)
            batch_sae_probe_projections = batch_sae_acts * cos_sims.to(
                batch_sae_acts.device
            )
            batch_probe_projections = batch_acts @ probe_direction.to(
                device=batch_sae_acts.device, dtype=batch_sae_acts.dtype
            )
            for i, prompt in enumerate(tqdm(batch_prompts, disable=not show_progress)):
                sae_acts = batch_sae_acts[i]
                act_probe_proj = batch_probe_projections[i].cpu().item()
                sae_act_probe_proj = batch_sae_probe_projections[i].cpu()

                ### calculate absorption_fraction ###

                # GT probe proj of main feats
                main_feats_probe_proj = (
                    torch.sum(sae_act_probe_proj[main_feature_ids]).cpu().item()
                )

                # GT probe proj of other feats
                potential_absorbers_mask = torch.ones(
                    sae_act_probe_proj.size(0), dtype=torch.bool
                )
                potential_absorbers_mask[main_feature_ids] = False
                potential_absorbers_mask &= (
                    cos_sims >= self.absorption_fraction_probe_cos_sim_threshold
                )
                potential_absorbers_mask &= sae_act_probe_proj > 0
                potential_absorbers_probe_proj = sae_act_probe_proj[
                    potential_absorbers_mask
                ]
                top_potential_absorbers_probe_proj = (
                    potential_absorbers_probe_proj.topk(
                        k=min(
                            self.absorption_fraction_max_absorbing_latents,
                            potential_absorbers_probe_proj.numel(),
                        )
                    ).values
                )
                top_potential_absorbers_total_probe_proj = (
                    torch.sum(top_potential_absorbers_probe_proj).cpu().item()
                )

                # final absorption_fraction calculation
                top_potential_absorbers_probe_proj_proportion = (
                    top_potential_absorbers_total_probe_proj / act_probe_proj
                )
                if (
                    main_feats_probe_proj >= act_probe_proj
                    or top_potential_absorbers_probe_proj_proportion
                    < self.probe_projection_proportion_threshold
                ):
                    absorption_fraction = 0.0
                elif main_feats_probe_proj <= 0.0:
                    absorption_fraction = 1.0
                else:
                    unaccounted_probe_proj = act_probe_proj - main_feats_probe_proj
                    absorption_probe_proj = min(
                        top_potential_absorbers_total_probe_proj, unaccounted_probe_proj
                    )
                    absorption_fraction = absorption_probe_proj / (
                        absorption_probe_proj + main_feats_probe_proj
                    )
                    absorption_fraction = np.clip(absorption_fraction, 0.0, 1.0)

                ### determine whether this is full absorption with a single absorbing latent ###

                with torch.inference_mode():
                    # sort by negative ig score
                    top_proj_feats = sae_act_probe_proj.topk(
                        self.topk_feats
                    ).indices.tolist()
                    main_feature_scores = _get_feature_scores(
                        main_feature_ids,
                        probe_cos_sims=cos_sims,
                        sae_acts=sae_acts,
                    )
                    top_projection_feature_scores = _get_feature_scores(
                        top_proj_feats,
                        probe_cos_sims=cos_sims,
                        sae_acts=sae_acts,
                    )
                    is_full_absorption = self._is_full_absorption(
                        probe_projection=act_probe_proj,
                        top_projection_feature_scores=top_projection_feature_scores,
                        main_feature_scores=main_feature_scores,
                    )

                results.append(
                    WordAbsorptionResult(
                        word=prompt.word,
                        prompt=prompt.base,
                        probe_projection=act_probe_proj,
                        main_feature_scores=main_feature_scores,
                        top_projection_feature_scores=top_projection_feature_scores,
                        absorption_fraction=absorption_fraction,
                        is_full_absorption=is_full_absorption,
                    )
                )
        return AbsorptionResults(
            main_feature_ids=main_feature_ids,
            word_results=results,
        )

    def _validate_prompts_are_same_length(self, prompts: list[SpellingPrompt]):
        "Validate that all prompts have the same token length"
        token_lens = {len(self.model.to_tokens(p.base)[0]) for p in prompts}
        if len(token_lens) > 1:
            raise ValueError(
                "All prompts must have the same token length! Variable-length prompts are not yet supported."
            )


def _get_feature_scores(
    feature_ids: list[int],
    probe_cos_sims: torch.Tensor,
    sae_acts: torch.Tensor,
) -> list[FeatureScore]:
    return [
        FeatureScore(
            feature_id=feature_id,
            probe_cos_sim=probe_cos_sims[feature_id].item(),
            activation=sae_acts[feature_id].item(),
        )
        for feature_id in feature_ids
    ]
