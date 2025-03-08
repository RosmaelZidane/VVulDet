from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score, accuracy_score, precision_recall_curve, auc
import pytorch_lightning as pl
import torch as th
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torchmetrics
import torch     
import dgl
import os
from dgl.nn import GATConv, GraphConv
from torch.optim import AdamW

import utils.cvecwefeaturemanip as gpht
import utils.utills as imp 

class LitGNN(pl.LightningModule):
    """Main Trainer."""

    def __init__(
        self,
        hfeat: int = 512, # 1024#
        embtype: str = "codebert",
        embfeat: int = -1,  # Keep for legacy purposes
        num_heads: int = 4,
        lr: float = 1e-4, #1e-3, # 1e-4
        hdropout: float = 0.2,
        mlpdropout: float = 0.2,
        gatdropout: float = 0.2,
        methodlevel: bool = False,
        nsampling: bool = False,
        model: str = "gat2layer",
        loss: str = "ce", # "sce", # 
        multitask: str = "linemethod",
        stmtweight: int = 1, # 5
        gnntype: str = "gat",
        random: bool = False,
        scea: float = 0.7,
    ):
        """Initialization."""
        super().__init__()
        self.lr = lr
        self.random = random
        self.save_hyperparameters()

        self.test_step_outputs = []

        # Set params based on embedding type
        if self.hparams.embtype == "codebert":
            self.hparams.embfeat = 768
            self.EMBED = "_CODEBERT"

        # Loss
        if self.hparams.loss == "sce":
            self.loss = gpht.SCELoss(self.hparams.scea, 1 - self.hparams.scea)
            self.loss_f = th.nn.CrossEntropyLoss()
        else:
            self.loss = th.nn.CrossEntropyLoss(
                weight=th.Tensor([1, self.hparams.stmtweight]) 
            )
            self.loss_f = th.nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2, average = 'macro')
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", num_classes=2)

        # GraphConv Type
        hfeat = self.hparams.hfeat
        gatdrop = self.hparams.gatdropout
        numheads = self.hparams.num_heads
        embfeat = self.hparams.embfeat
        gnn_args = {"out_feats": hfeat}
        if self.hparams.gnntype == "gat":
            gnn = GATConv
            gat_args = {"num_heads": numheads, "feat_drop": gatdrop}
            gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}
            gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads}
        elif self.hparams.gnntype == "gcn":
            gnn = GraphConv
            gnn1_args = {"in_feats": embfeat, **gnn_args}
            gnn2_args = {"in_feats": hfeat, **gnn_args}

        # model: gat2layer
        if "gat" in self.hparams.model:
            self.gat = gnn(**gnn1_args)
            self.gat2 = gnn(**gnn2_args)
            fcin = hfeat * numheads if self.hparams.gnntype == "gat" else hfeat
            self.fc = th.nn.Linear(fcin, self.hparams.hfeat)
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: contains femb
        if "+femb" in self.hparams.model:
            self.fc_femb = th.nn.Linear(embfeat * 2, self.hparams.hfeat)

        # Transform codebert embedding
        self.codebertfc = th.nn.Linear(768, self.hparams.hfeat)

        # Hidden Layers
        self.fch = []
        for _ in range(8):
            self.fch.append(th.nn.Linear(self.hparams.hfeat, self.hparams.hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(self.hparams.hdropout)
        self.fc2 = th.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        """Forward pass."""
        if self.hparams.nsampling and not test:
            hdst = g[2][-1].dstdata[self.EMBED]
            h_func = g[2][-1].dstdata["_FUNC_EMB"]
            g2 = g[2][1]
            g = g[2][0]
            if "gat2layer" in self.hparams.model:
                h = g.srcdata[self.EMBED]
            elif "gat1layer" in self.hparams.model:
                h = g2.srcdata[self.EMBED]
        else:
            g2 = g
            h = g.ndata[self.EMBED]
            if len(feat_override) > 0:
                h = g.ndata[feat_override]
            h_func = g.ndata["_FUNC_EMB"]
            hdst = h

        if self.random:
            return th.rand((h.shape[0], 2)).to(self.device), th.rand(
                h_func.shape[0], 2
            ).to(self.device)

        # model: contains femb
        if "+femb" in self.hparams.model:
            h = th.cat([h, h_func], dim=1)
            h = F.elu(self.fc_femb(h))

        # Transform h_func if wrong size
        if self.hparams.embfeat != 768:
            h_func = self.codebertfc(h_func)

        # model: gat2layer
        if "gat" in self.hparams.model:
            if "gat2layer" in self.hparams.model:
                h = self.gat(g, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
                h = self.gat2(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            elif "gat1layer" in self.hparams.model:
                h = self.gat(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            h = self.mlpdropout(F.elu(self.fc(h)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Edge masking (for GNNExplainer)
        if test and len(e_weights) > 0:
            g.ndata["h"] = h
            g.edata["ew"] = e_weights
            g.update_all(
                dgl.function.u_mul_e("h", "ew", "m"), dgl.function.mean("m", "h")
            )
            h = g.ndata["h"]

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            h = self.mlpdropout(F.elu(self.fconly(hdst)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Hidden layers
        for idx, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h)))
            h_func = self.hdropout(F.elu(hlayer(h_func)))
        h = self.fc2(h)
        h_func = self.fc2(
            h_func
        )  # Share weights between method-level and statement-level tasks

        if self.hparams.methodlevel:
            g.ndata["h"] = h
            return dgl.mean_nodes(g, "h"), None
        else:
            return h, h_func  # Return two values for multitask training

    def shared_step(self, batch, test=False):
        """Shared step."""
        logits = self(batch, test)
        if self.hparams.methodlevel:
            if self.hparams.nsampling:
                raise ValueError("Cannot train on method level with nsampling.")
            labels = dgl.max_nodes(batch, "_VULN").long()
            labels_func = None
        else:
            if self.hparams.nsampling and not test:
                labels = batch[2][-1].dstdata["_VULN"].long()
                labels_func = batch[2][-1].dstdata["_FVULN"].long()
            else:
                labels = batch.ndata["_VULN"].long()
                labels_func = batch.ndata["_FVULN"].long()
        return logits, labels, labels_func

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits, labels, labels_func = self.shared_step(batch)
        loss1 = self.loss(logits[0], labels)
        
        logits1 = logits[0]
        
        if not self.hparams.methodlevel:
            loss2 = self.loss_f(logits[1], labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
            
        else:
            loss = loss1
            acc_func = self.accuracy(logits, labels_func)
            self.log("train_acc_func", acc_func, prog_bar=True, logger=True, batch_size=batch_idx)
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_idx)
        self.log("train_loss_func", loss2, on_epoch=True, prog_bar=True, batch_size=batch_idx)
        self.log("train_acc", self.accuracy(preds, labels), prog_bar=True, batch_size=batch_idx)
        self.log("train_mcc", self.mcc(preds, labels), prog_bar=True, batch_size=batch_idx)
        
        if not self.hparams.methodlevel:
            self.log("train_acc_func", self.accuracy(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
            self.log("train_mcc_func", self.mcc(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits, labels, labels_func = self.shared_step(batch)
        logits1 = logits[0]
        loss1 = self.loss(logits1, labels)
        if not self.hparams.methodlevel:
            logits2 = logits[1]
            loss2 = self.loss_f(logits2, labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_idx)
        self.log("val_auroc", self.auroc(preds, labels), prog_bar=True, batch_size=batch_idx)
        self.log("val_acc", self.accuracy(preds, labels), prog_bar=True, batch_size=batch_idx)
        self.log("val_mcc", self.mcc(preds, labels), prog_bar=True, batch_size=batch_idx)

        if not self.hparams.methodlevel:
            self.log("val_acc_func", self.accuracy(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
            self.log("val_auroc_func", self.auroc(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
            self.log("val_mcc_func", self.mcc(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits, labels, labels_func = self.shared_step(batch, test=True)
        logits1 = logits[0]
        loss1 = self.loss(logits1, labels)
        if not self.hparams.methodlevel:
            logits2 = logits[1]
            loss2 = self.loss_f(logits2, labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None

        metrics = {
            "test_loss": loss,
            "test_acc": self.accuracy(preds, labels),
            "test_mcc": self.mcc(preds, labels),
        }
        
        if not self.hparams.methodlevel:
            metrics["test_acc_func"] = self.accuracy(preds_func, labels_func)
            metrics["test_mcc_func"] = self.mcc(preds_func, labels_func)

        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        """Test epoch end."""
        avg_metrics = {
            key: th.mean(th.stack([x[key] for x in self.test_step_outputs]))
            for key in self.test_step_outputs[0].keys()
        }
        print(f"what is insight self.test_step_outputs {self.test_step_outputs}")
        self.test_step_outputs.clear()
        self.log_dict(avg_metrics)
        return
        
    def configure_optimizers(self):
        """Configure optimizers."""
        return AdamW(self.parameters(), lr=self.lr)


# compute metrics function
def statementcalculate_metrics(model, data):
    """
    Calculate ranking metrics: MRR, N@5, MFR,
    and classification metrics: F1-Score, Precision.
    """

    # Extract function-level predictions and true labels
    all_preds_ = []
    all_labels_ = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    for batch in data.test_dataloader():
        with torch.no_grad():
            logits, labels, labels_func = model.shared_step(batch.to(device), test=True)
            if labels is not None:  # the comented code is best
                preds_ = torch.softmax(logits[0], dim=1).cpu().numpy()
                labels_f = labels.cpu().numpy()
                all_preds_.extend(preds_)
                all_labels_.extend(labels_f)

    all_preds_ = np.array(all_preds_)
    all_labels_ = np.array(all_labels_)


    predicted_classes = np.argmax(all_preds_, axis=1)
    f1_c = f1_score(all_labels_, predicted_classes, average="macro")
    precision = precision_score(all_labels_, predicted_classes, average="macro")
    accuracy = accuracy_score(all_labels_, predicted_classes, normalize= True )
    recall = recall_score(all_labels_, predicted_classes, average= "macro") # average=None, zero_division=np.nan
    roc_ = roc_auc_score(all_labels_, predicted_classes, average= "macro")
    mcc_ = matthews_corrcoef(all_labels_, predicted_classes)
    precisionq, recallq, thresholds = precision_recall_curve(all_labels_, predicted_classes)
    pr_auc = auc(recallq, precisionq)
    prediction = pd.DataFrame({"true label": all_labels_,
                          "Predicted_label": predicted_classes})
   
    return {
        "accuracy": accuracy,
        "Precision": precision,
        "F1-Score": f1_c,
        "recall" : recall,
        "roc_auc" : roc_,
        "mcc": mcc_,
        "pr_auc": pr_auc,
    }, prediction
    
def methodcalculate_metrics(model, data):
    """
    Calculate ranking metrics: MRR, N@5, MFR,
    and classification metrics: F1-Score, Precision.
    """

    # Extract function-level predictions and true labels
    all_preds_ = []
    all_labels_ = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    for batch in data.test_dataloader():
        with torch.no_grad():
            logits, labels, labels_func = model.shared_step(batch.to(device), test=True)
            if labels_func is not None:  # the comented code is best
                preds_ = torch.softmax(logits[1], dim=1).cpu().numpy()
                labels_f = labels_func.cpu().numpy()
                all_preds_.extend(preds_)
                all_labels_.extend(labels_f)

    all_preds_ = np.array(all_preds_)
    all_labels_ = np.array(all_labels_)

    predicted_classes = np.argmax(all_preds_, axis=1)
    f1_c = f1_score(all_labels_, predicted_classes, average="macro")
    precision = precision_score(all_labels_, predicted_classes, average="macro")
    accuracy = accuracy_score(all_labels_, predicted_classes, normalize= True )
    recall = recall_score(all_labels_, predicted_classes, average= "macro") # average=None, zero_division=np.nan
    roc_ = roc_auc_score(all_labels_, predicted_classes, average= "macro")
    mcc_ = matthews_corrcoef(all_labels_, predicted_classes)
    precisionq, recallq, thresholds = precision_recall_curve(all_labels_, predicted_classes)
    
    pr_auc = auc(recallq, precisionq)
    
    prediction = pd.DataFrame({"true label": all_labels_,
                          "Predicted_label": predicted_classes})
    

    return {
        "accuracy": accuracy,
        "Precision": precision,
        "F1-Score": f1_c,
        "recall" : recall,
        "roc_auc" : roc_,
        "mcc": mcc_,
        "pr_auc": pr_auc,
    }, prediction

#  train the classifier
checkpoint_path = f"{imp.outputs_dir()}/checkpoints"
samplesz = -1

# list of epoch tried [30, 50 , 130, 200, 250], note that effective learning is achieve with hight epchs

max_epochs = 50


if not os.path.exists(path=checkpoint_path):
    print(f"[Infos ] --->> Training the model with Domain Knowledge: cwe description")
    run_id = imp.get_run_id()
    savepath = imp.get_dir(imp.processed_dir() / "codebert" / run_id)
    model = LitGNN( 
                   hfeat= 512,# List of values used  [256, #1024, #512, # 1024 two with 512 and 2 with 1024]
                   embtype= "codebert",
                   methodlevel=False,
                   nsampling=True,
                   model= "gat2layer", # gat1layer
                   loss="ce",
                   hdropout=0.5,
                   gatdropout=0.3,
                   num_heads=4,
                   multitask="linemethod", # methodlevel
                   stmtweight=1,
                   gnntype="gat",
                   scea=0.5,
                   lr=1e-4, # [1e-3, 1e-3, 1e-3]
                   )
    
    # load data
    data = gpht.CVEFixesDatasetLineVDDataModule(
        batch_size=64,
        sample=samplesz,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype= "pdg+raw",
        splits="default",
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    metrics = ["train_loss", "val_loss", "val_auroc"]
    trainer = pl.Trainer(
        accelerator= "auto",
        devices= "auto",
        default_root_dir=savepath,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback], 
        max_epochs=max_epochs,
        )
    trainer.fit(model, data)
    checkpoint_path = imp.get_dir(f"{imp.outputs_dir()}/checkpoints")
    trainer.save_checkpoint(f"{checkpoint_path}/model-cve-cwe-checkpoint.ckpt")
    # test 
    trainer.test(model, data)
    print(f"Statement level prediction")
    metrics1 = methodcalculate_metrics(model, data)[0]
    dfm = pd.DataFrame([metrics1])
    dfm.to_csv(f"{imp.outputs_dir()}/cve_cwe-statement-evaluation_metrics.csv", index=False)
    print(f"statelement {metrics1} ")
    # method level
    print(f"method level prediction")
    metrics = statementcalculate_metrics(model, data)[0]
    dfm = pd.DataFrame([metrics])
    dfm.to_csv(f"{imp.outputs_dir()}/cve_cwe-method-evaluation_metrics.csv", index=False)
    print(f"[Infos ] Metrics on test set \n{metrics}\n[Infos ] -> Done.")
else:   
    print(f"[Infos ] ---> Saved model exits.")
    print(f"[Infos ] ---> Load from pretarined")
    # load model
    model = LitGNN.load_from_checkpoint(f"{checkpoint_path}/model-cve-cwe-checkpoint.ckpt")
    # load data
    data = gpht.CVEFixesDatasetLineVDDataModule(
        batch_size=64,
        sample=samplesz,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype= "pdg+raw",
        splits="default",
        )
    # compute metrics
    # test 
    print(f"Statement level prediction")
    metrics1 = methodcalculate_metrics(model, data)[0]
    dfm = pd.DataFrame([metrics1])
    dfm.to_csv(f"{imp.outputs_dir()}/cve_cwe-statement-evaluation_metrics.csv", index=False)
    print(f"statelement {metrics1} ")
    # method level
    print(f"method level prediction")
    metrics = statementcalculate_metrics(model, data)[0]
    dfm = pd.DataFrame([metrics])
    dfm.to_csv(f"{imp.outputs_dir()}/cve_cwe-method-evaluation_metrics.csv", index=False)
    print(f"[Infos ] Metrics on test set \n{metrics}\n[Infos ] -> Done.")

    
    
