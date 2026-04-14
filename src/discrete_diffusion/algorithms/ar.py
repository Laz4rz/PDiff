"""AR algorithm implementation extracted from :mod:`algorithms.algo`."""

import torch
import omegaconf

from . import base as trainer_base


class AR(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        self.mask_id, vocab_size = trainer_base.ensure_mask_token(tokenizer)
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()
        self._val_gen_acc_batches_seen = 0
        self._val_completion_table_rows = []
        self._val_completion_table_columns = None
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert not self.config.algo.time_conditioning
        assert self.config.prior.type == "none"

    def _process_model_input(self, x0, valid_tokens):
        valid_tokens = valid_tokens[:, 1:]
        return x0, valid_tokens

    def nll(self, x0, current_accumulation_step=None, train_mode=False):
        del current_accumulation_step, train_mode
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        output = self.backbone(input_tokens, None)
        output[:, :, self.mask_id] = self.neg_infinity
        output = output.log_softmax(-1)
        return -output.gather(-1, output_tokens[:, :, None])[:, :, 0]

    def _process_sigma(self, sigma):
        return None

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_gen_acc_batches_seen = 0

    def validation_step(self, batch, batch_idx):
        del batch_idx
        valid_tokens = batch.get("loss_mask", batch["attention_mask"])
        losses = self._loss(batch["input_ids"], valid_tokens)
        self.metrics.update_valid(losses.nlls, losses.num_tokens)

        accuracy_tokens = batch.get("accuracy_mask", None)
        if self._should_compute_generation_accuracy(accuracy_tokens):
            gen_metrics = self._compute_generation_accuracy(
                batch["input_ids"],
                accuracy_tokens,
                batch.get("attention_mask", None),
            )
            if gen_metrics is not None:
                self.metrics.update_valid_gen(*gen_metrics)
                self._val_gen_acc_batches_seen += 1
        return losses.loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if getattr(self.metrics.valid_gen_acc_token, "weight", 0) > 0:
            self.log(
                name="val/gen_acc_token",
                value=self.metrics.valid_gen_acc_token.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if getattr(self.metrics.valid_gen_acc_sample, "weight", 0) > 0:
            self.log(
                name="val/gen_acc_sample",
                value=self.metrics.valid_gen_acc_sample.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def _should_compute_generation_accuracy(self, accuracy_tokens):
        if accuracy_tokens is None:
            return False
        if self.trainer.sanity_checking:
            return False
        if not bool(getattr(self.config.eval, "compute_generation_accuracy", True)):
            return False
        raw_max_batches = getattr(self.config.eval, "generation_accuracy_max_batches", 1)
        max_batches = 1 if raw_max_batches is None else int(raw_max_batches)
        if max_batches <= 0:
            return False
        if self._val_gen_acc_batches_seen >= max_batches:
            return False
        return True

    @torch.no_grad()
    def _compute_generation_accuracy(self, input_tokens, accuracy_tokens, attention_mask=None):
        method = str(
            omegaconf.OmegaConf.select(
                self.config, "eval.generation_accuracy_method", default="exact"
            )
        )

        accuracy_mask = accuracy_tokens.to(device=input_tokens.device, dtype=torch.bool)
        if self.ignore_bos:
            accuracy_mask = accuracy_mask.clone()
            accuracy_mask[:, 0] = False
        has_accuracy = accuracy_mask.any(dim=-1)
        if not has_accuracy.any():
            return None

        pred = input_tokens.clone()
        for pos in range(1, pred.shape[1]):
            logits = self.backbone(pred[:, :pos], None)
            logits = logits.clone()
            logits[:, :, self.mask_id] = self.neg_infinity
            next_token = logits[:, -1, :].argmax(dim=-1)
            pred[:, pos] = torch.where(accuracy_mask[:, pos], next_token, input_tokens[:, pos])

        self._maybe_log_validation_completions(
            input_tokens=input_tokens,
            pred_tokens=pred,
            accuracy_mask=accuracy_mask,
            has_accuracy=has_accuracy,
            attention_mask=attention_mask,
            method=method,
        )

        if method == "brevo_topo":
            from ..data.generators.brevo import TopoSortDepthStats

            dataset_cfg = getattr(self.config.data, "dataset_config", None)
            if dataset_cfg is None:
                multi = False
            elif hasattr(dataset_cfg, "get"):
                multi = bool(dataset_cfg.get("multi_token", False))
            else:
                multi = bool(getattr(dataset_cfg, "multi_token", False))
            parser = (
                TopoSortDepthStats.parse_tokens_multi
                if multi
                else TopoSortDepthStats.parse_tokens
            )

            correct_samples = 0
            num_samples = int(has_accuracy.sum().item())
            valid_idx = torch.nonzero(has_accuracy, as_tuple=False).squeeze(-1)
            for idx in valid_idx.tolist():
                sample_pred = pred[idx].detach().cpu().tolist()
                ok, _, _ = parser(sample_pred)
                if ok:
                    correct_samples += 1

            return (
                None,
                None,
                torch.tensor(float(correct_samples), device=self.device),
                torch.tensor(float(num_samples), device=self.device),
            )

        if method != "exact":
            raise ValueError(
                "Unsupported eval.generation_accuracy_method="
                f"{method!r}. Expected one of: exact, brevo_topo."
            )

        token_correct = (pred == input_tokens) & accuracy_mask
        sample_correct = ((~accuracy_mask) | (pred == input_tokens)).all(dim=-1) & has_accuracy
        return (
            token_correct.sum().to(dtype=torch.float32),
            accuracy_mask.sum().to(dtype=torch.float32),
            sample_correct.sum().to(dtype=torch.float32),
            has_accuracy.sum().to(dtype=torch.float32),
        )

    def _split_prefix_completion(self, seq_tokens, completion_mask, attention_mask=None):
        if attention_mask is None:
            valid_mask = torch.ones_like(completion_mask, dtype=torch.bool)
        else:
            valid_mask = attention_mask.to(dtype=torch.bool)

        prefix_mask = (~completion_mask) & valid_mask
        completion_mask = completion_mask & valid_mask

        prefix_ids = seq_tokens[prefix_mask]
        completion_ids = seq_tokens[completion_mask]

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            prefix_ids = prefix_ids[prefix_ids != int(pad_id)]
            completion_ids = completion_ids[completion_ids != int(pad_id)]

        return prefix_ids.tolist(), completion_ids.tolist()

    def _render_token_ids(self, token_ids):
        ids = [int(tok) for tok in token_ids]
        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            converted = convert(ids)
            if isinstance(converted, list):
                return " ".join(str(tok) for tok in converted)
        return " ".join(str(tok) for tok in ids)

    def _maybe_log_validation_completions(
        self,
        input_tokens,
        pred_tokens,
        accuracy_mask,
        has_accuracy,
        attention_mask,
        method,
    ):
        if not bool(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions", default=False
            )
        ):
            return
        if self._val_gen_acc_batches_seen != 0:
            return
        if self.trainer is None or self.trainer.sanity_checking:
            return
        if not self.trainer.is_global_zero:
            return
        logger = self.trainer.logger
        if logger is None or not hasattr(logger, "log_table"):
            return

        max_samples = int(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions_max_samples", default=8
            )
        )
        if max_samples <= 0:
            return

        parser = None
        if method == "brevo_topo":
            from ..data.generators.brevo import TopoSortDepthStats

            dataset_cfg = getattr(self.config.data, "dataset_config", None)
            if dataset_cfg is None:
                multi = False
            elif hasattr(dataset_cfg, "get"):
                multi = bool(dataset_cfg.get("multi_token", False))
            else:
                multi = bool(getattr(dataset_cfg, "multi_token", False))
            parser = (
                TopoSortDepthStats.parse_tokens_multi
                if multi
                else TopoSortDepthStats.parse_tokens
            )

        valid_idx = torch.nonzero(has_accuracy, as_tuple=False).squeeze(-1).tolist()
        rows = []
        for idx in valid_idx[:max_samples]:
            true_seq = input_tokens[idx].detach().cpu()
            pred_seq = pred_tokens[idx].detach().cpu()
            comp_mask = accuracy_mask[idx].detach().cpu()
            attn = attention_mask[idx].detach().cpu() if attention_mask is not None else None

            prefix_ids, target_completion = self._split_prefix_completion(
                true_seq, comp_mask, attn
            )
            _, predicted_completion = self._split_prefix_completion(pred_seq, comp_mask, attn)

            if parser is not None:
                ok, _, _ = parser(pred_seq.tolist())
                rows.append(
                    [
                        self._render_token_ids(prefix_ids),
                        self._render_token_ids(target_completion),
                        self._render_token_ids(predicted_completion),
                        str(bool(ok)),
                    ]
                )
            else:
                rows.append(
                    [
                        self._render_token_ids(prefix_ids),
                        self._render_token_ids(target_completion),
                        self._render_token_ids(predicted_completion),
                    ]
                )

        if not rows:
            return

        if parser is not None:
            columns = ["prefix", "target_completion", "predicted_completion", "topo_ok"]
        else:
            columns = ["prefix", "target_completion", "predicted_completion"]
        include_step = bool(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions_include_step", default=False
            )
        )
        if include_step:
            columns = ["global_step"] + columns
            rows = [[int(self.global_step)] + row for row in rows]

        aggregate = bool(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions_aggregate", default=True
            )
        )
        if aggregate:
            if self._val_completion_table_columns != columns:
                self._val_completion_table_columns = columns
                self._val_completion_table_rows = []
            self._val_completion_table_rows.extend(rows)
            max_rows = int(
                omegaconf.OmegaConf.select(
                    self.config, "eval.log_validation_completions_max_rows", default=256
                )
            )
            if max_rows > 0 and len(self._val_completion_table_rows) > max_rows:
                self._val_completion_table_rows = self._val_completion_table_rows[-max_rows:]
            data = self._val_completion_table_rows
        else:
            data = rows

        table_key = str(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions_table_key", default="val/completions"
            )
        )
        logger.log_table(
            key=table_key,
            columns=columns,
            data=data,
        )
        self._maybe_update_wandb_summary_table(logger, table_key, columns, data)

    def _maybe_update_wandb_summary_table(self, logger, table_key, columns, data):
        experiment = getattr(logger, "experiment", None)
        if experiment is None:
            return
        summary = getattr(experiment, "summary", None)
        if summary is None:
            return
        try:
            import wandb
        except Exception:
            return

        summary_key = str(
            omegaconf.OmegaConf.select(
                self.config,
                "eval.log_validation_completions_summary_key",
                default=f"{table_key}_latest",
            )
        )
        try:
            summary[summary_key] = wandb.Table(columns=columns, data=data)
        except Exception:
            return
