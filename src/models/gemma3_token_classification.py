"""Gemma-3 token-classification head, shared by training and inference.

Single definition. The training and inference modules previously each carried
their own copy and had already drifted: the training copy omitted forward(),
so SoftLabelTokenClassificationTrainer.compute_loss -> model(**inputs) hit
nn.Module._forward_unimplemented and raised NotImplementedError for any
gemma-3 checkpoint.
"""

import torch.nn as nn
from transformers import Gemma3PreTrainedModel, Gemma3TextModel
from transformers.modeling_outputs import TokenClassifierOutput


class Gemma3ForTokenClassification(Gemma3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"vision_tower", r"multi_modal_projector"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma3TextModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)
        self._register_load_state_dict_pre_hook(self._merge_multimodal_keys_hook)
        self.post_init()

    def _merge_multimodal_keys_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Rewrite checkpoint keys at from_pretrained() time.

        Gemma-3 multimodal checkpoints nest the text backbone under
        `text_model.`; this class names it `model`, so strip that segment
        in-place before the state dict is loaded.
        """
        keys_to_rename = [k for k in state_dict.keys() if "text_model" in k]
        for key in keys_to_rename:
            value = state_dict.pop(key)
            new_key = key.replace("text_model.", "")
            state_dict[new_key] = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.score(self.dropout(outputs[0]))
        return TokenClassifierOutput(logits=logits)
