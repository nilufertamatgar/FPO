import torch
import os
import traceback
from data.all_atom_dataset import Cluster_AllAtomDataset
from data import all_atom_parse as aap
from data.residue_config import configure as configure_residues
from torch.utils.data import DataLoader
from torch.optim import AdamW
from functools import partial
from ema_pytorch import EMA
from tqdm.auto import tqdm
import yaml
import wandb
import argparse


def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any() or (tensor < 0).any()


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class Trainer(object):
    def __init__(
        self,
        config,
        generative_model,
        train_dataset,
        val_dataset,
        val_dataset1=None,
        results_folder="/dapustor/nilufer/ADFLIP/results/new_weights/",
        load_weight = False
    ):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # This insanity is brought to you by the PyTorch team (TM)
        torch.inverse(torch.ones((1, 1), device=device))

        self.model = generative_model.to(device)
        self.config = config

        self.batch_size = config.training.batch_size
        self.gradient_accumulate_every = config.training.gradient_accumulate_every
        self.wandb = config.wandb.use
        self.results_folder = results_folder
        if os.path.exists(self.results_folder) is False:
            os.makedirs(self.results_folder)

        # dataset and dataloader

        self.dataloader_kwargs = {
            "shuffle": True,
            "pin_memory": False,
            "num_workers": 12,
            "collate_fn": partial(aap.batch_structure_data_list, to_torch=True),
            "prefetch_factor": 4,
        }

        self.ds = train_dataset
        self.set_train_dataloader(batch_size=self.batch_size,**self.dataloader_kwargs)
        ngpus = torch.cuda.device_count()
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            **self.dataloader_kwargs,
        )


        self.val_loader1 = None

        if val_dataset1 is not None:
            self.val_loader1 = DataLoader(val_dataset1,batch_size=self.batch_size, **self.dataloader_kwargs)


        self.opt = AdamW(
            self.model.parameters(),
            lr=config.training.train_lr,
            weight_decay=config.training.weight_decay,
        )

        if self.config.training.ema:
            self.ema = EMA(
                self.model,
                beta=self.config.training.ema_beta,
                update_every=self.config.training.ema_update_every,
            )

        self.step = 0
        self.has_lowered_lr = False
        self.evaluate_step = config.training.evaluate_step

        print('loadweight,',load_weight)
        if load_weight:
            self.load(load_weight)

        if config.training.dataparallel:
            self.model = torch.nn.DataParallel(self.model)

    def set_train_dataloader(self,batch_size, **kwargs):
        for key in kwargs:
            self.dataloader_kwargs[key] = kwargs[key]
        dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            **self.dataloader_kwargs,
        )
        self.train_num_steps = self.config.training.max_epochs * len(dl)
        self.dl = cycle(dl)

    def save(self, milestone,best=False):
        model_state = self.model.module.state_dict() if self.config.training.dataparallel else self.model.state_dict()
        data = {"config": self.config, "step": self.step, "model": model_state, "opt": self.opt.state_dict()}
        if self.config.training.ema:
            data["ema"] = self.ema.state_dict()
        if best:
            torch.save(
                data, os.path.join(str(self.results_folder), f"_{self.config.wandb.run_name}_best.pt")
            )
        else:
            torch.save(
                data, os.path.join(str(self.results_folder), f"_{self.config.wandb.run_name}_{milestone}.pt")
            )

    def load(self,  filename=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        data = torch.load(
            str(self.results_folder) + "/" + filename, map_location=device
        )

        self.model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.config.training.ema and "ema" in data:
            self.ema.load_state_dict(data["ema"])


        print(f"loading from {filename}, in step {self.step}")

    def train(self):

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        train_loss, val_rr, val_perplexity, total_loss, val_loss_list = (
            [],
            [],
            [],
            0,
            [],
        )
        val_loss = torch.tensor([5.0])
        if self.wandb:
            if self.config.training.load_weight:
                wandb.init(
                    project=self.config.wandb.project,
                    config=self.config.to_dict(),
                    name=self.config.wandb.run_name,
                    entity=self.config.wandb.team,
                    resume="must",
                    id=f"{self.config.wandb.ID}"
                )
                print(f"resuming from {self.config.wandb.ID}")
            else:
                wandb.init(
                    project=self.config.wandb.project,
                    config=self.config.to_dict(),
                    name=self.config.wandb.run_name,
                    entity=self.config.wandb.team,
                )
            wandb.watch(self.model)
        eval_model = self.ema.ema_model if self.config.training.ema else self.model
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                valid_micro_steps = 0
                for _ in range(self.gradient_accumulate_every):
                    try:
                        data = next(self.dl)
                        if data is None:
                            print("Empty batch from dataloader, skipping micro-step")
                            continue
                        data = {k: v.to(device) for k, v in data.__dict__.items()}
                        
                        data = {
                            k: v.float() if v.dtype == torch.float64 else v
                            for k, v in data.items()
                        }
                        logits, loss = self.model(data)
                        if self.config.training.dataparallel:
                            loss = loss.mean()
                        if not torch.isfinite(loss):
                            print(f"Non-finite loss detected ({loss.item()}), skipping micro-step")
                            continue
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        if self.wandb:
                            wandb.log({"loss": loss.item()})
                        loss.backward()
                        valid_micro_steps += 1

                    except torch.cuda.OutOfMemoryError:
                        print("Out of memory error, lowering batch size")
                        new_batchsize = max(
                            self.batch_size // 2, 1
                        )
                        self.set_train_dataloader(batch_size=new_batchsize)

                    except Exception as e:
                        print(f"Error in training: {traceback.format_exc()}")

                if valid_micro_steps == 0:
                    self.opt.zero_grad(set_to_none=True)
                    continue

                grads_finite = True
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break
                if not grads_finite:
                    print("Non-finite gradients detected, skipping optimizer step")
                    self.opt.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.clip_grad_norm
                )

                all_iter = len(self.ds) // self.batch_size
                num_iter = self.step % all_iter + 1
                pbar.set_description(f"loss: {total_loss/num_iter:.4f}")

                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                if self.config.training.ema:
                    self.ema.to(device)
                    self.ema.update()

                if (
                    self.step % self.evaluate_step == 0
                ):  # finish one epoch

                    train_loss.append(total_loss / all_iter)

                    self.model.eval()
                    eval_model.eval()
                    validation_loss, validation_accuracy, validation_perplexity,val_interact_non_protein_accuracy, val_interact_non_protein_perplexity = (
                        eval_model.test(self.val_loader)
                    )

                    val_loss_list.append(validation_loss)

                    if self.val_loader1 is not None:
                        (
                            validation_loss1,
                            validation_accuracy1,
                            validation_perplexity1,
                            val_interact_non_protein_accuracy1,
                            val_interact_non_protein_perplexity1,
                        ) = eval_model.test(self.val_loader1)



                    val_rr.append(validation_accuracy)
                    val_perplexity.append(validation_perplexity)
                    if self.wandb:
                        log_dict = {
                            "train_loss": total_loss / all_iter,
                            "val_loss": validation_loss,

                            "val_rr": validation_accuracy,

                            "val_perplexity": validation_perplexity,

                            'val_interaction_accuracy':val_interact_non_protein_accuracy,
                            'val_interaction_perplexity':val_interact_non_protein_perplexity,

                        }
                        if self.val_loader1 is not None:
                            log_dict.update({

                                "val_loss1": validation_loss1,
                                "val_rr1": validation_accuracy1,
  
                                "val_perplexity1": validation_perplexity1,

                            })
                        wandb.log(log_dict)
                    total_loss = 0
                    print(
                        f"validation loss: {validation_loss:.4f}, validation accuracy: {validation_accuracy:.4f}, validation perplexity: {validation_perplexity:.4f}"
                    )

                    if validation_loss == min(val_loss_list):
                        self.save(self.step,best=True)
                if self.step % 100 == 0:
                    self.save(self.step)            

                pbar.update(1)

        print("training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/dapustor/nilufer/ADFLIP/config/train_v1.yaml",
        help="Date of experiment",
    )
    args = parser.parse_args()

    yaml_file = args.config_path

    with open(yaml_file, "r") as file:
        config = Config(yaml.safe_load(file))

    configure_residues(include_nonstd_amino_acids=config.data.include_nonstd_amino_acids)

    train_dataset = Cluster_AllAtomDataset(
        config.data, dataset_type="train", cache_length=30000
    )
    val_dataset = Cluster_AllAtomDataset(config.data, dataset_type="valid")


    val_dataset1 = Cluster_AllAtomDataset(config.data, dataset_type="valid", test_time=0.5)

    



    from model.zoidberg.zoidberg_GNN import Zoidberg_GNN

    denoiser = Zoidberg_GNN(
        hidden_dim=config.zoidberg_denoiser.hidden_dim,
        encoder_hidden_dim=config.zoidberg_denoiser.hidden_dim,
        num_blocks=config.zoidberg_denoiser.num_layers,
        num_heads=config.zoidberg_denoiser.num_heads,
        k=config.zoidberg_denoiser.k_neighbors,
        num_positional_embeddings=config.zoidberg_denoiser.num_positional_embeddings,
        num_rbf=config.zoidberg_denoiser.num_rbf,
        augment_eps=config.zoidberg_denoiser.augment_eps,
        backbone_diheral=config.zoidberg_denoiser.backbone_diheral,
        dropout=config.zoidberg_denoiser.dropout,
        update_atom = config.zoidberg_denoiser.update_atom,
        num_decoder_blocks=config.zoidberg_denoiser.num_decoder_blocks,
        num_tfmr_heads=config.zoidberg_denoiser.num_tfmr_heads,
        num_tfmr_layers=config.zoidberg_denoiser.num_tfmr_layers,
        number_ligand_atom = config.zoidberg_denoiser.number_ligand_atom,
        mpnn_cutoff=config.zoidberg_denoiser.mpnn_cutoff,
        output_dim=config.zoidberg_denoiser.output_dim,
    )


    from model.discrete_flow_aa import DiscreteFlow_AA

    model = DiscreteFlow_AA(config, denoiser, min_t=0.0)
    pytorch_total_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Total number of parameters in denoiser: {pytorch_total_params}")

    print('train_dataset:',len(train_dataset),'max num residue:',train_dataset.max_num_residues, 'max num atom:',train_dataset.max_num_atoms, 'cut off type:',train_dataset.cut_position_type)
    print('val_dataset:',len(val_dataset),'max num residue:',val_dataset.max_num_residues, 'max num atom:',val_dataset.max_num_atoms, 'cut off type:',val_dataset.cut_position_type)




    trainer = Trainer(
        config,
        model,
        train_dataset,
        val_dataset,
        val_dataset1=val_dataset1,
        load_weight=config.training.load_weight,
    )
    trainer.train()
