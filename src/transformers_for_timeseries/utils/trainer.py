import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import gc

from transformers_for_timeseries.config_and_scripts.n1_settings import config
from transformers_for_timeseries.utils.synthetic_dataset_creation import forced_pendulum_dataset
from transformers_for_timeseries.models.trans_scinet import PendulumNet
from transformers_for_timeseries.ml_tools.metrics import trans_scinet_loss
from transformers_for_timeseries.ml_tools.train_callbacks import EarlyStopping, LRScheduling, GradientClipping


def train_scinet(
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int = 150, 
        kld_beta: float = 0.001, 
        early_stopper: EarlyStopping = None, 
        gradient_clipper: GradientClipping = None, 
        lr_scheduler: LRScheduling = None,
        device: torch.device = torch.device('cpu')
        ) -> None:

    torch.cuda.empty_cache()
    model.to(device)
    print("------training on {}-------\n".format(device))
    history = {'train_loss': [], 'valid_loss': []}
    print(f"{'Epoch':<20} ||| {'Train Loss':<15} ||| {'KLD Loss':<12} {'Recon Loss':<12} ||||||| {'Valid Loss':<15}")

    # Training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        kld_loss, recon_loss = 0.0, 0.0
        for observations, questions, _ in tqdm(train_loader, desc="Training", leave=False):
            a_corr = observations.clone()
            observations = observations.to(device)
            questions = questions.to(device)
            a_corr = a_corr.to(device)

            optimizer.zero_grad()
            possible_answer, mean, logvar = model(observations, questions)
            loss, l_kld, l_recon = trans_scinet_loss(possible_answer, a_corr, mean, logvar, beta=kld_beta, truncate_input=False)
            loss.backward()
            if gradient_clipper is not None:
                gradient_clipper.on_backward_end(model)
            optimizer.step()

            train_loss += loss.item() * observations.size(0)
            kld_loss += l_kld.item() * observations.size(0)
            recon_loss += l_recon.item() * observations.size(0)
        train_loss /= len(train_loader.dataset)
        kld_loss /= len(train_loader.dataset)
        recon_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Evaluation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for observations, questions, _ in tqdm(valid_loader, desc="Validation", leave=False):
                a_corr = observations.clone()
                observations = observations.to(device)
                questions = questions.to(device)
                a_corr = a_corr.to(device)

                possible_answer, mean, logvar = model(observations, questions)
                loss = trans_scinet_loss(possible_answer, a_corr, mean, logvar, beta=kld_beta, truncate_input=False)[0]
                valid_loss += loss.item() * observations.size(0)
        
        valid_loss /= len(valid_loader.dataset)
        history['valid_loss'].append(valid_loss)

        print(f"{f'{epoch+1}/{num_epochs}':<20}  |  {train_loss:<15.6f}  |  {kld_loss:<12.6f} {recon_loss:<12.6f}    |    {valid_loss:<15.6f}")

        if early_stopper is not None:
            if early_stopper.check_stop(valid_loss, model):
                print(f"Early stopping at epoch {epoch + 1} with loss {valid_loss:.4f}")
                print(f"Restoring best weights for model.")
                early_stopper.restore_best_weights(model)
                break

        if lr_scheduler is not None:
            lr_scheduler.step(valid_loss)

        path = config.DIR_PARAMS_CHECKPOINTS / "pendulum_scinet_checkpointed.pth"
        torch.save(model.state_dict(), path)

        del observations, questions, a_corr, possible_answer, mean, logvar, loss, l_kld, l_recon
        torch.cuda.empty_cache()
        gc.collect()
    
    return history

def plot_history(history_train: list, history_valid: list) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_train, 'b-', linewidth=2, label='Train Loss')
    plt.plot(history_valid, 'r-', linewidth=2, label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = config.DIR_FIGURES / "training_validation_loss.png"
    plt.savefig(path)
    return None


if __name__ == "__main__":

    device = config.DEVICE

    path = config.DIR_PREPROCESSED_DATA
    train_dataset = torch.load(path / "train_dataset.pt", weights_only=False)
    valid_dataset = torch.load(path / "valid_dataset.pt", weights_only=False)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE_EVAL, shuffle=True)

    pendulum_net = PendulumNet(
        encoder_input_size=config.M_INPUT_SIZE,
        enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
        latent_size=config.M_LATENT_SIZE,
        question_size=config.M_QUESTION_SIZE,
        dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
        output_size=config.M_OUTPUT_SIZE
    ).bfloat16()
    print(pendulum_net)
    optimizer = torch.optim.Adam(pendulum_net.parameters(), lr=config.FIRST_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    early_stopper = EarlyStopping(patience=config.ES_PATIENCE, min_delta=config.ES_MIN_DELTA)
    gradient_clipper = GradientClipping(max_norm=config.GC_MAX_NORM)
    lr_scheduler = LRScheduling(optimizer, factor=config.LRS_FACTOR, patience=config.LRS_PATIENCE, min_lr=config.LRS_MIN_LR, min_delta=config.LRS_MIN_DELTA)


    history = train_scinet(
        train_dataloader, 
        valid_dataloader, 
        pendulum_net, 
        optimizer, 
        num_epochs=config.NUM_EPOCHS, 
        kld_beta=config.KLD_BETA, 
        early_stopper=early_stopper, 
        gradient_clipper=gradient_clipper, 
        lr_scheduler=lr_scheduler,
        device=device
    )

    path = config.DIR_MODEL_PARAMS / (config.BEST_MODEL_NAME + ".pth")
    torch.save(pendulum_net.state_dict(), path)

    plot_history(history['train_loss'][5:], history['valid_loss'][5:])