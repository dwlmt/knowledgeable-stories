import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, LanguageModelHead, Seq2SeqEncoder
from allennlp.data import Vocabulary

from typing import Iterator, List, Dict, Optional, Any

from allennlp.nn import RegularizerApplicator, InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, logger
from allennlp.training.metrics import CategoricalAccuracy, Perplexity, BLEU
from pytorch_transformers import GPT2LMHeadModel
from torch.nn import CrossEntropyLoss, Parameter
from transformers.modeling_auto import AutoModelWithLMHead

from knowledgeablestories.dataset_readers.special_tokens import token_tags

EOS_TOKEN_IDS = [50256, 0]


@Model.register("knowledgeable_stories")
class KnowStoryModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 lm_name: str = "gpt2",
                 embedder_vocab_size: int = None,
                 dropout: float = 0.0,
                 dataset_config={"atomic": {"generate_text" : 10, "bleu" : True},"roc_lm": {}},
                 generation_config = {"temperature": 1.0, "top_k": 50, "max_length": 100, "do_sample": False, "num_beams": 5},
                 metric_config={"training_metrics": False, "lm_accuracy_top_k": [1, 5, 20]},
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = None,
                 ) -> None:
        super().__init__(vocab, regularizer)

        self._tokenizer = PretrainedTransformerTokenizer(model_name="gpt2", do_lowercase=False)

        # Add the relations as new tokens.
        self._tokenizer._tokenizer.add_tokens(token_tags)

        self._dataset_config = dataset_config
        self._generation_config = generation_config
        self._metric_config = metric_config

        self._lm_model = AutoModelWithLMHead.from_pretrained(lm_name)

        # If additional characters have been added then the model needs updated for the additional tokens.
        self._embedder_vocab_size = embedder_vocab_size
        if self._embedder_vocab_size:
            self._lm_model.resize_token_embeddings(self._embedder_vocab_size)

        self._dropout = torch.nn.Dropout(dropout)

        self._metrics = {}
        for key, values in self._dataset_config.items():

            for k in self._metric_config["lm_accuracy_top_k"]:
                self._metrics[f"{key}_lm_accuracy_{k}"] = CategoricalAccuracy(top_k=k)

            self._metrics[f"{key}_lm_perplexity"] = Perplexity()
            self._metrics[f"{key}_lm_bleu"] = BLEU(exclude_indices=set(EOS_TOKEN_IDS))
            self._metrics[f"{key}_lm_bleu_2"] = BLEU(ngram_weights=(0.0, 1.0, 0.0, 0.0), exclude_indices=set(EOS_TOKEN_IDS))

        if initializer is not None:
            initializer(self)

    def forward(self,
                passages: Dict[str, torch.Tensor] = None,
                premises: Dict[str, torch.Tensor] = None,
                conclusions: Dict[str, torch.Tensor] = None,
                negative_conclusions: Dict[str, torch.Tensor] = None,
                arguments: Dict[str, torch.Tensor] = None,
                negative_arguments: Dict[str, torch.Tensor] = None,
                metadata: List[Dict[str, Any]] = None
                ) -> Dict[str, torch.Tensor]:

        output = {}
        dataset_name = metadata[0]["dataset"]

        loss = torch.tensor(0.0)

        # Argument based training is for training specific relations just on the text without hierarchichal structure.
        if arguments != None:

            tokens = arguments["tokens"]
            lm_loss, lm_logits, presents, = self._lm_model(tokens, labels=tokens)

            with torch.no_grad():
                if not self.training or self._metric_config["training_metrics"]:
                    for k in self._metric_config["lm_accuracy_top_k"]:
                        self._metrics[f"{dataset_name}_lm_accuracy_{k}"](lm_logits, tokens)

                    self._metrics[f"{dataset_name}_lm_perplexity"](lm_loss)

                    if "generate_text" in self._dataset_config[dataset_name] and self._dataset_config[dataset_name]["true"]:

                        generated_text = self._generate_text(dataset_name, premises)

                        prem_tokens = premises["tokens"]

                        self._bleu_score_if_required(dataset_name, prem_tokens, conclusions, generated_text)

            loss = loss.to(lm_loss.device)
            loss += lm_loss

        output["loss"] = loss

        return output

    def _generate_text(self, dataset_name, premises):
        num_of_sequences = self._dataset_config[dataset_name]["generate_text"]
        generated_text = self._lm_model.generate(
            input_ids=premises["tokens"],
            max_length=self._generation_config["max_length"],
            temperature=self._generation_config["temperature"],
            top_k=self._generation_config["top_k"],
            do_sample=self._generation_config["do_sample"],
            num_beams=self._generation_config["num_beams"],
            eos_token_ids=EOS_TOKEN_IDS,
            num_return_sequences=num_of_sequences,
        )
        return generated_text

    def _bleu_score_if_required(self, dataset_name, prem, conclusions, generated_text):
        if "bleu" in self._dataset_config[dataset_name] and self._dataset_config[dataset_name]["bleu"] == True:
            conc = conclusions["tokens"]

            for i in range(conc.size(0)):

                text_hyp = generated_text[i, ..., len([p for p in prem[i].tolist() if p not in EOS_TOKEN_IDS]):]
                text_conc = conc[i]

                for h in text_hyp:
                    for c in text_conc:

                        if len([x for x in h.tolist() if x not in EOS_TOKEN_IDS]) > 0 and  len(h.tolist()) > 1 \
                                and len([x for x in c.tolist() if x not in EOS_TOKEN_IDS]) > 0 and len(c.tolist()) > 1 :

                            h_unsqueezed = h.unsqueeze(dim=0).long()
                            c_unsqueezed = c.unsqueeze(dim=0).long()

                            self._metrics[f"{dataset_name}_lm_bleu"](h_unsqueezed, c_unsqueezed)
                            self._metrics[f"{dataset_name}_lm_bleu_2"](h_unsqueezed, c_unsqueezed)

                            for h in text_hyp:
                                print(
                                    f"Hypothesis: {self._tokenizer._tokenizer.decode(h.tolist(), skip_special_tokens=False)}")

                            for h in text_conc:
                                print(
                                    f"Gold: {self._tokenizer._tokenizer.decode(c.tolist(), skip_special_tokens=False)}")

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self._metrics.items()}

        for k, v in metrics.items():

            if isinstance(v,Dict):
                if "BLEU" in v:
                    metrics[k] = v["BLEU"]

        return metrics

