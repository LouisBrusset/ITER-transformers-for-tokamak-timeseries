import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers_for_timeseries.config_and_scripts.n1_settings import config
from transformers_for_timeseries.utils.synthetic_dataset_creation import forced_pendulum_dataset
from torch.utils.data import DataLoader
from transformers_for_timeseries.models.trans_scinet import PendulumNet
from transformers_for_timeseries.data_loading.synthetic_pendulum_creation import create_synthetic_damped_forced_pendulum, create_trapezoidal_forcing



def load_trained_model(model_path: str, device: torch.device = torch.device('cpu')) -> PendulumNet:
    """
    Function to automatically load a trained PendulumNet model from a given path.
    Args:
        model_path (str): Path to the saved model state dictionary.
        device (torch.device): Device to load the model onto.
    Returns:
        PendulumNet: The loaded PendulumNet model in evaluation mode.
    """
    model = PendulumNet(
        encoder_input_size=config.M_INPUT_SIZE,
        enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
        latent_size=config.M_LATENT_SIZE,
        question_size=config.M_QUESTION_SIZE,
        dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
        output_size=config.M_OUTPUT_SIZE
    ).bfloat16()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model


def full_inference(model: PendulumNet, data_loader: DataLoader, normalization_stats: dict, device: torch.device, save: bool = False) -> tuple:
    """
    Perform full inference on the provided data loader using the given model.
    Args:
        model (PendulumNet): The trained PendulumNet model.
        data_loader (DataLoader): DataLoader for the dataset to perform inference on.
        normalization_stats (dict): Dictionary containing normalization statistics.
        device (torch.device): Device to perform inference on.
        save (bool): Whether to save the reconstructions and latent means to disk.
    Returns:
        tuple: Tuple containing:
            - all_reconstructions (np.ndarray): Reconstructed outputs from the model.
            - all_means (np.ndarray): Latent means from the model.
            - all_logvars (np.ndarray): Latent log-variances from the model.
            - all_observations (np.ndarray): Original observations from the dataset.
            - all_questions (np.ndarray): Original questions from the dataset.
            - all_answers (np.ndarray): Original answers from the dataset.
            - all_params (np.ndarray): Parameters associated with each sample.
    """
    #obs_factor, que_factor, ans_factor = normalization_stats['obs_mean_max'], normalization_stats['que_mean_max'], normalization_stats['ans_mean_max']
    all_reconstructions, all_means, all_logvars = [], [], []
    all_observations, all_questions, all_answers = [], [], []
    all_params = []
    
    model.eval()
    with torch.no_grad():
        for observations, questions, params in data_loader:
            # observations = observations.to(device) / obs_factor
            # questions = questions.to(device) / que_factor
            # possible_answer, mean, logvar = model(observations, questions)
            # all_reconstructions.append(possible_answer.cpu().numpy() * ans_factor)
            # all_means.append(mean.cpu().numpy())
            # all_logvars.append(logvar.cpu().numpy())
            # all_observations.append(observations.cpu().numpy() * obs_factor)
            # all_questions.append(questions.cpu().numpy() * que_factor)
            # all_answers.append(answers.cpu().numpy() * ans_factor)
            answers = observations.clone()
            observations = observations.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            possible_answer, mean, logvar = model(observations, questions)
            all_reconstructions.append(possible_answer.float().cpu().numpy())
            all_means.append(mean.float().cpu().numpy())
            all_logvars.append(logvar.float().cpu().numpy())
            all_observations.append(observations.float().cpu().numpy())
            all_questions.append(questions.float().cpu().numpy())
            all_answers.append(answers.float().cpu().numpy())
            all_params.append(params.float().cpu().numpy())
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_means = np.concatenate(all_means, axis=0)
    all_logvars = np.concatenate(all_logvars, axis=0)
    all_observations = np.concatenate(all_observations, axis=0)
    all_questions = np.concatenate(all_questions, axis=0)
    all_answers = np.concatenate(all_answers, axis=0)
    all_params =  np.concatenate(all_params, axis=0)
    if save:
        path = config.DIR_PROCESSED_DATA / f"test_reconstructions.npy"
        np.save(path, all_reconstructions)
        path = config.DIR_PROCESSED_DATA / f"test_latent.npy"
        np.save(path, all_means)
    return all_reconstructions, all_means, all_logvars, all_observations, all_questions, all_answers, all_params



def plot_reconstructions_answers_observations(observations: np.ndarray, answers: np.ndarray, reconstructions: np.ndarray, questions: np.ndarray, sample_idx: int) -> None:
    """
    Plot the reconstruction vs the answer for a given sample index, along with the forcing question and observations.
    Args:
        observations (np.ndarray): Array of observations.
        answers (np.ndarray): Array of true answers.
        reconstructions (np.ndarray): Array of model reconstructions.
        questions (np.ndarray): Array of forcing questions.
        sample_idx (int): Index of the sample to plot.
    Returns:
        None
    """
    time = np.linspace(0, config.MAXTIME, config.M_INPUT_SIZE)
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    # plt.plot(time, answers[sample_idx], label='Answer', color='blue')
    # plt.plot(time, reconstructions[sample_idx], label='Reconstruction', color='orange', linestyle='--')
    plt.plot(answers[sample_idx], label='Answer', color='blue')
    plt.plot(reconstructions[sample_idx], label='Reconstruction', color='orange', linestyle='--')
    plt.title(f'Reconstruction vs Answer (Sample Index: {sample_idx})')
    plt.xlabel('Time Steps')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # plt.plot(time, questions[sample_idx], label='Forcing Amplitude', color='green')
    # plt.plot(time, observations[sample_idx], label='Observations', color='black', alpha=0.3)
    plt.plot(questions[sample_idx], label='Forcing Amplitude', color='green')
    plt.plot(observations[sample_idx], label='Observations', color='black', alpha=0.3)
    plt.title('Forcing Amplitude and Observations')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = config.DIR_FIGURES / f"reconstruction_vs_answer_sample_{sample_idx}.png"
    plt.savefig(path)
    #plt.show()
    return None



def plot_latent_variables(means: np.ndarray, n_max_cols: int = 5) -> None:
    """
    Plot the distribution of latent variables.
    Args:
        means (np.ndarray): Array of latent means.
        n_max_cols (int): Maximum number of columns in the subplot grid.
    Returns:
        None
    """
    latent_dim = means.shape[1]

    n_cols = min(latent_dim, n_max_cols)
    n_rows = (latent_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i in range(latent_dim):
        axes[i].hist(means[:, i], bins=30, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Latent Variable {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    for i in range(latent_dim, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    path = config.DIR_FIGURES / f"latent_variables_distribution.png"
    plt.savefig(path)
    #plt.show()
    return None



def plot_latent_space_3d(all_means: np.ndarray, all_params: np.ndarray) -> None:
    """
    See 3D plot of the latent space colored by physical parameters.
    Args:
        all_means (np.ndarray): Array of latent means.
        all_params (np.ndarray): Array of physical parameters.
    Returns:
        None
    """
    L_values = all_params[:, 0]
    b_values = all_params[:, 1]
    latent_1 = all_means[:, 0]
    latent_2 = all_means[:, 1]
    latent_3 = all_means[:, 2]

    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(131, projection='3d')
    p = ax.scatter(latent_1, latent_2, latent_3, c=L_values, cmap='viridis')
    fig.colorbar(p, ax=ax, label='Spring constant kapa')
    ax.set_title('Latent Space Colored by kapa')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')

    ax = fig.add_subplot(132, projection='3d')
    p = ax.scatter(latent_1, latent_2, latent_3, c=b_values, cmap='plasma')
    fig.colorbar(p, ax=ax, label='Damping beta')
    ax.set_title('Latent Space Colored by beta')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')

    plt.tight_layout()
    path = config.DIR_FIGURES / f"latent_space_3d.png"
    plt.savefig(path)
    #plt.show()
    return None





def get_one_latent_activation(
    model: torch.nn.Module, observation: np.array, device: torch.device = torch.device("cpu")
) -> np.array:
    """
    Get the latent activation for a single observation.
    Args:
        model (nn.Module): The trained model.
        observation (np.array): The input observation.
        device (torch.device, optional): The device to run the model on. Defaults to CPU.
    Returns:
        np.array: The latent activation.
    """
    torch.cuda.empty_cache()
    model.to(device).eval().bfloat16()
    observation_tensor = (
        torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device).bfloat16()
    )
    latent_activations = []
    with torch.no_grad():
        mean, _ = model.encoder(observation_tensor)
        latent_activations.append(mean.float().cpu().numpy())
    return np.array(latent_activations)

def get_latent_activations(
    model: torch.nn.Module,
    kapa_range: tuple,
    beta_range: tuple,
    pixel_by_line: int = 50,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.array]:
    """
    Get the latent activations for a grid of (kapa, beta) values.
    Args:
        model (nn.Module): The trained model.
        kapa_range (tuple): Range of kapa values (min, max).
        beta_range (tuple): Range of beta values (min, max).
        pixel_by_line (int, optional): Number of points per axis. Defaults to 50.
        device (torch.device, optional): The device to run the model on. Defaults to CPU
    Returns:
        3-tuple containing:
            np.array: The grid of kapa values
            np.array: The grid of b values
            np.array: The latent activations for the grid of (kapa, b) values
    """
    # Create a grid of (kapa, b) values
    kapa_values = np.linspace(*kapa_range, pixel_by_line)
    beta_values = np.linspace(*beta_range, pixel_by_line)
    kapa_grid, beta_grid = np.meshgrid(kapa_values, beta_values)
    # Loop over the grid and get latent activations
    t = np.linspace(0, config.MAXTIME, config.TIMESTEPS)
    theta0, omega0 = config.THETA0, config.OMEGA0
    trap_forcing_values_sampled = {
        "start_value": 1.0,
        "end_value": 5.0,
        "proportions": [1/6, 1/3, 2/3, 5/6]
    }
    A_forcing = create_trapezoidal_forcing(t=t, N_time_steps=config.TIMESTEPS, **trap_forcing_values_sampled)
    omega_forcing = np.ones(config.TIMESTEPS) * config.OMEGA_VALUE
    latent_activations = []
    for kapa in kapa_values:
        for beta in beta_values:
            # Generate observation
            observation = create_synthetic_damped_forced_pendulum(A_forcing, omega_forcing, beta, kapa, theta0, omega0, t)
            # Find latent activation with encoder only
            latent_act = get_one_latent_activation(model, observation, device=device)
            latent_activations.append(latent_act)
    latents = np.array(latent_activations).reshape(pixel_by_line, pixel_by_line, -1)
    return kapa_grid, beta_grid, latents

def plot_3d_latent_activations(
    kapa_grid: np.array,
    beta_grid: np.array,
    latent_activations: np.array,
    save_path: str,
    shared_scale: bool = False,
) -> None:
    """ "
    Plot the latent activations in 3D for each latent dimension.
    Args:
        kapa_grid (np.array): The grid of kapa values.
        b_grid (np.array): The grid of b values.
        latent_activations (np.array): The latent activations for the grid of (kapa, b) values.
        save_path (str): Path to save the plot.
        shared_scale (bool, optional): Whether to use a shared z-axis scale for all plots. Defaults to False.
    Returns:
        None
        Figures are saved to the specified path.
    """
    latent_dim = latent_activations.shape[2]
    fig = plt.figure(figsize=(6 * latent_dim, 8))
    if shared_scale:
        z_min = np.min(latent_activations)
        z_max = np.max(latent_activations)
    else:
        z_min = z_max = None
    # Plot each latent dimension
    for i in range(latent_dim):
        ax = fig.add_subplot(1, latent_dim, i + 1, projection="3d")
        surf = ax.plot_surface(
            kapa_grid,
            beta_grid,
            latent_activations[:, :, i],
            alpha=0.8,
            cmap="viridis",
            label=f"Latent {i+1}",
        )
        if shared_scale:
            ax.set_zlim(z_min, z_max)
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$b$")
        ax.set_title(f"Latent Dimension {i+1}")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return None




if __name__ == "__main__":
    device = config.DEVICE

    # Load model
    model_path = config.DIR_MODEL_PARAMS / f"{config.BEST_MODEL_NAME}.pth"
    pendulum_net = load_trained_model(model_path, device)

    # Load test dataset
    path = config.DIR_PREPROCESSED_DATA / f"test_dataset.pt"
    test_dataset = torch.load(path, weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE_EVAL, shuffle=True)

    # Load normalization stats
    # path = config.DIR_OTHERS_DATA_CHANNEL / f"{config.MODEL_NAME}_normalization_stats.pt"
    # normalization_stats = torch.load(path)
    normalization_stats = None

    # Full inference on test set
    reconstructions, means, logvars, all_observations, all_questions, all_answers, all_params = full_inference(pendulum_net, test_loader, normalization_stats, device, save=False)

    # Plot latent variables distributions
    plot_latent_variables(means, n_max_cols=5)

    # Plot latent space 3D
    plot_latent_space_3d(all_means=means, all_params=all_params)

    # Plot latent activations over true (kapa, beta) grid
    kapa_range = config.KAPA_RANGE
    beta_range = config.BETA_RANGE

    kapa_grid, beta_grid, latent_activations = get_latent_activations(
        pendulum_net, kapa_range, beta_range, device=device
    )
    path = config.DIR_FIGURES / "latent_activations_3d.png"
    plot_3d_latent_activations(
        kapa_grid, beta_grid, latent_activations, save_path=path, shared_scale=True
    )

    # Plot n random reconstruction
    n = 5
    for _ in range(n):
        sample_idx = np.random.choice(config.TEST_SIZE)
        plot_reconstructions_answers_observations(
            observations=all_observations, 
            answers=all_answers,
            reconstructions=reconstructions, 
            questions=all_questions, 
            sample_idx=sample_idx
        )




