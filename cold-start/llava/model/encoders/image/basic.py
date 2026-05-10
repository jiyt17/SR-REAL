from functools import partial
from typing import Any, Dict, List, Optional

import torch

from llava.model.encoders.base import BaseEncoder

__all__ = ["BasicImageEncoder"]


class BasicImageEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
    ) -> None:
        super().__init__(parent)
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def embed_tokens(self, tokens: Optional[str]) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        token_ids = self.parent.tokenizer(tokens).input_ids
        token_ids = torch.tensor(token_ids, device=self.parent.device)
        return self.parent.llm.model.embed_tokens(token_ids)

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(
        self, images: List[torch.Tensor], xyzs: List[torch.Tensor], masks: List[torch.Tensor], config: Dict[str, Any]
    ) -> List[torch.Tensor]:
        images = torch.stack(images, dim=0)
        if xyzs is not None:
            xyzs = torch.stack(xyzs, dim=0)
        image_features, mask_features, dummy_mask_features = self.parent.encode_images(
            images,
            masks,
            xyzs=xyzs,
            block_sizes=config.get("block_sizes"),
            block_lengths=config.get("block_lengths"),
        )
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        if mask_features is not None:
            return (
                [process_features(f) for f in image_features],
                [m[None, ...] for m in mask_features],
                [d[None, ...] for d in dummy_mask_features],
            )
        else:
            return [process_features(f) for f in image_features], [], [d[None, ...] for d in dummy_mask_features]
