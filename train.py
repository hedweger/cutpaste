import torch, argparse, warnings, importlib
from datasets.cutpaste import CutPasteDataset as Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument("-t", "--dataset_type")
    parser.add_argument("-c", "--dataset_class")
    parser.add_argument("-e", "--encoder", default="resnet18")
    parser.add_argument("-n", "--max_epochs", default=1, type=int)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("-i", "--input", default=224)
    parser.add_argument("--gradient_clip_val", default=25, type=int)
    parser.add_argument("--log_every_n_step", default=12, type=int)
    parser.add_argument("--learning_rate", default=0.003)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--weight_decay", default=0.00003)
    parser.add_argument("--train_batch_size", default=6, type=int)
    parser.add_argument("--test_batch_size", default=256, type=int)
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--log_dir", default=r"./tb_logs")
    parser.add_argument("--log_dir_name", default=r"exp1")

    args = parser.parse_args()
    return args
class CutPasteDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.args = hparams
        
    def setup(self, stage=None):
        self.train_dataset = Dataset(self.args.dataset_type, self.args.dataset_class, self.args.input, 'train')
        self.test_dataset = Dataset(self.args.dataset_type, self.args.dataset_class, self.args.input, 'test')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            pin_memory=True
        )
    
class CutPasteModule(pl.LightningModule):
    def __init__(self, hparams):
        super(CutPasteModule, self).__init__()
        self.save_hyperparameters(hparams)
        module = importlib.import_module('models.models')
        class_ = getattr(module, hparams.encoder)
        self.model = class_(pretrained = args.pretrained)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        logits, embeds = self.model(x)
        return logits, embeds

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, self.hparams.max_epochs
        )
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x = torch.cat(batch, axis=0).to(self.device)
        y = torch.arange(len(batch)).to(self.device)
        y = y.repeat_interleave(len(batch[0]))
        logits, embeds = self(x.to(self.device))
        loss = self.criterion(logits, y)
        predicted = torch.argmax(logits, axis=1)
        accuracy = torch.true_divide(torch.sum(predicted == y), predicted.size(0))
        self.log('loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits, embeds = self(x.to(self.device))
        pred_probs = torch.softmax(logits, dim=-1)
        auc_score = roc_auc_score(y.cpu().numpy(), pred_probs[:,1].cpu().numpy(), multi_class='ovo')
        self.log('hp_metric', np.float32(auc_score), on_step=False, on_epoch=True, prog_bar=True)
        self.auc_score = auc_score
        return auc_score

    def test_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits, embeds = self(x.to(self.device))
        pred_probs = torch.softmax(logits, dim=-1)
        auc_score = roc_auc_score(y.cpu().numpy(), pred_probs[:,1].cpu().numpy(), multi_class='ovo')
        self.logger.log_metrics({'hp_metric': auc_score})
        print(auc_score)
        return auc_score
    

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    args = get_args()
    logger = TensorBoardLogger(Path(args.log_dir), name=args.log_dir_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        dirpath=f'{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints',
        filename=f'{args.encoder}',
        mode='min',
        save_last=True
    )
    module = CutPasteModule(hparams=args)
    datamodule = CutPasteDataModule(hparams=args)
    trainer = pl.Trainer(
        accelerator = args.accelerator, 
        logger=logger, 
        callbacks=[checkpoint_callback], 
        max_epochs=args.max_epochs, 
        gradient_clip_val=args.gradient_clip_val,
        auto_scale_batch_size=True,
        limit_train_batches=32,
    )
    trainer.fit(module, datamodule)
    trainer.test(ckpt_path='best', verbose=True, datamodule=datamodule)
