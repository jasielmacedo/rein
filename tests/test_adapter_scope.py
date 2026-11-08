import copy

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import Qwen3_5TextModel

from rein.model.backbone import LORA_TARGETS, LoraSettings, attach_lora
from rein.model.batch import collate
from rein.model.encode import RequestEncoder
from rein.model.prefix import shared_state_forward
from rein.schema import DecisionRequest
from tests.devices import DEVICE, HIDDEN_ATOL

RANK = 4
ALPHA = 8
LORA_B_STD = 0.5
WEIGHT_SEED = 11
PEFT_REFERENCE_ATOL = max(HIDDEN_ATOL, 1e-4)
PEFT_REFERENCE_RTOL = 1e-4


def _wake(peft_model: torch.nn.Module) -> None:
    """LoRA starts at zero output; the B matrices get weight so the adapter changes the hidden states."""
    generator = torch.Generator().manual_seed(WEIGHT_SEED)
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            param.data.copy_(torch.randn(param.shape, generator=generator) * LORA_B_STD)


def test_rows_only_adapter_matches_an_activated_lora_on_full_sequences(
    tiny_backbone: Qwen3_5TextModel,
    encoder: RequestEncoder,
    roleplay_request: DecisionRequest,
    tool_call_request: DecisionRequest,
) -> None:
    reference_backbone = copy.deepcopy(tiny_backbone)
    rows_model = attach_lora(tiny_backbone, LoraSettings(RANK, ALPHA, 0.0)).eval()
    _wake(rows_model)
    reference = get_peft_model(
        reference_backbone,
        LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            lora_dropout=0.0,
            target_modules=list(LORA_TARGETS),
            bias="none",
            alora_invocation_tokens=[encoder.ids.question_open],
        ),
    ).eval()
    set_peft_model_state_dict(reference, get_peft_model_state_dict(rows_model))

    encoded = [encoder.encode(request) for request in (roleplay_request, tool_call_request)]
    batch = collate(encoded, encoder.pad_id).to(DEVICE)
    with torch.no_grad():
        rows_only = shared_state_forward(tiny_backbone, batch, adapter_on_state=False)
        everywhere = shared_state_forward(tiny_backbone, batch)
        index = 0
        for request in encoded:
            state = list(request.state_ids)
            for question in request.questions:
                for row in question.rows:
                    ids = torch.tensor([[*state, *row]], device=DEVICE)
                    full = reference(input_ids=ids, use_cache=False, alora_offsets=[len(row)]).last_hidden_state
                    torch.testing.assert_close(
                        rows_only[index, : len(row)],
                        full[0, len(state) :],
                        atol=PEFT_REFERENCE_ATOL,
                        rtol=PEFT_REFERENCE_RTOL,
                    )
                    index += 1
    assert not torch.allclose(rows_only, everywhere, atol=HIDDEN_ATOL)
