import zipfile
from contextlib import nullcontext
from pathlib import Path

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from smaller3d.models.metrics import IoU
from torch import nn
from torch.utils.data import DataLoader


class SemanticSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        # model
        self.teacher_model = hydra.utils.instantiate(config.teacher_model)

        # freeze model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.student_model = hydra.utils.instantiate(config.student_model)
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label
        self.criterion = hydra.utils.instantiate(config.loss)
        # metrics
        self.confusion = hydra.utils.instantiate(config.metrics)
        self.iou = IoU()
        # misc
        self.labels_info = dict()

    def forward(self, x):
        # Teacher network
        with self.optional_freeze():
            sout = self.student_model(x)
        sout = self.student_model.final(sout)
        return sout
    def forward_teacher(self,x):
        with torch.no_grad():
            tout = self.teacher_model(x)
            tout = self.teacher_model.final(tout)
        return tout
        
    def training_step(self, batch, batch_idx):
        data, target = batch
        data = ME.SparseTensor(coords=data.coordinates, feats=data.features)
        data.to(self.device)


        print("Training Step!!!")
        tout1 = self.teacher_model(data)
        print(f"{tout1.shape}")
        tout2 = self.teacher_model.final(tout1)
        print(f"{tout2.shape}")

        sout1 = self.student_model(data)
        print(f"{sout1.shape}")

        sout2 = self.student_model.final(sout1)
        print(f"{sout2.shape}")

        loss = self.criterion(sout2.F, tout2.F).unsqueeze(0)
        if self.config.student_model.last_feature_map_included:
            sout1 = self.student_model.ExpandSparseLayer(sout1)
            loss += self.criterion(sout1.F, tout1.F).unsqueeze(0)

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        data, target = batch
        inverse_maps = data.inverse_maps
        original_labels = data.original_labels
        data = ME.SparseTensor(coords=data.coordinates, feats=data.features)
        data.to(self.device)


        print("Validation Step!!!")
        tout1 = self.teacher_model(data)
        print(f"{tout1.shape}")
        tout2 = self.teacher_model.final(tout1)
        print(f"{tout2.shape}")

        sout1 = self.student_model(data)
        print(f"{sout1.shape}")

        sout2 = self.student_model.final(sout1)
        print(f"{sout2.shape}")

        loss = self.criterion(sout2.F, tout2.F).unsqueeze(0)
        if self.config.student_model.last_feature_map_included:
            sout1 = self.student_model.ExpandSparseLayer(sout1)
            loss += self.criterion(sout1.F, tout1.F).unsqueeze(0)


        # getting original labels
        ordered_output = []
        for i in range(len(inverse_maps)):
            # https://github.com/NVIDIA/MinkowskiEngine/issues/119
            ordered_output.append(sout2.F[sout2.C[:, 0] == i])
        sout2 = ordered_output
        for i, (out, inverse_map) in enumerate(zip(sout2, inverse_maps)):
            out = out.max(1)[1].view(-1).detach().cpu()
            sout2[i] = out[inverse_map].numpy()

        self.confusion.add(np.hstack(sout2), np.hstack(original_labels))
        return {
            "val_loss": loss,
        }

    def training_epoch_end(self, outputs):
        train_loss = torch.cat([out["loss"] for out in outputs], dim=0).mean()
        results = {"train_loss": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        val_loss = torch.cat([out["val_loss"] for out in outputs], dim=0).mean()
        results = {"val_loss": val_loss}

        confusion_matrix = self.confusion.value()
        results_iou = self.iou.value(confusion_matrix)
        for i, k in enumerate(self.labels_info.keys()):
            metric_name = self.labels_info[k]["name"]
            results["val_IoU_" + metric_name] = results_iou[i]
        results["val_IoU"] = results_iou.mean()
        self.log_dict(results)
        self.confusion.reset()

    def test_step(self, batch, batch_idx):
        data, target = batch
        inverse_maps = data.inverse_maps
        original_labels = data.original_labels
        data = ME.SparseTensor(coords=data.coordinates, feats=data.features)
        data.to(self.device)
        tout = self.forward_teacher(data)
        sout = self.forward(data)
        loss = 0
        if original_labels[0].size > 0:
            loss = self.criterion(sout.F, tout.F).unsqueeze(0)
            target = target.detach().cpu()
        original_predicted = []
        for i in range(len(inverse_maps)):
            # https://github.com/NVIDIA/MinkowskiEngine/issues/119
            out = sout.F[sout.C[:, 0] == i]
            out = out.max(1)[1].view(-1).detach().cpu()
            original_predicted.append(out[inverse_maps[i]].numpy())

        return {
            "test_loss": loss,
            "metrics": {
                "original_output": original_predicted,
                "original_label": original_labels,
            },
        }

    def test_epoch_end(self, outputs):
        outs = []
        labels = []
        results = {}

        results = dict()
        for out in outputs:
            outs.extend(out["metrics"]["original_output"])
            labels.extend(out["metrics"]["original_label"])
        if labels[0].size > 0:
            self.confusion.reset()
            self.confusion.add(np.hstack(outs), np.hstack(labels))
            confusion_matrix = self.confusion.value()
            results_iou = self.iou.value(confusion_matrix)
            for i, k in enumerate(self.labels_info.keys()):
                metric_name = self.labels_info[k]["name"]
                results["test_IoU_" + metric_name] = results_iou[i]
            results["test_IoU"] = results_iou.mean()

        if self.test_dataset.data[0].get("scene"):
            results = scannet_submission(
                self.trainer.weights_save_path,
                outs,
                results,
                self.test_dataset.data,
                self.test_dataset._remap_model_output,
            )
        self.log_dict(results)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)
        self.labels_info = self.train_dataset.label_info

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader, self.train_dataset, collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader, self.test_dataset, collate_fn=c_fn,
        )


def scannet_submission(path, outputs, results, test_filebase, remap_function):
    # checking if we are submitting scannet
    filepaths = []
    for out, info in zip(outputs, test_filebase):
        save_path = (
            Path(path).parent
            / "submission"
            / f"scene{info['scene']:04}_{info['sub_scene']:02}.txt"
        )
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        out = remap_function(out)
        np.savetxt(save_path, out, fmt="%d")
        filepaths.append(save_path)

    zip_path = save_path.parent.parent / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as myzip:
        for file in filepaths:
            myzip.write(file)
    results["submission_path"] = str(zip_path)
    return results
