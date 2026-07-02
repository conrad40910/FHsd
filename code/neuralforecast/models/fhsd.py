


__all__ = ['FHsd']


from typing import Optional, Tuple, Dict, Any
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class FrequencyEntropyPooling(nn.Module):
    """Frequency-dynamic pooling (FDAM) with FFT/STFT analysis and mask-driven downsampling."""

    def __init__(
        self,
        pool_kernel_size: int = 2,
        use_stft: bool = False,
        stft_window_size: int = 32,
        stft_hop_length: int = 16,
        num_freq_bands: int = 4,
        entropy_method: str = "energy_band",
        preserve_energy: bool = True,
    ):
        super().__init__()
        self.pool_kernel_size = pool_kernel_size
        self.use_stft = use_stft
        self.stft_window_size = stft_window_size
        self.stft_hop_length = stft_hop_length
        self.num_freq_bands = max(num_freq_bands, 1)
        self.entropy_method = entropy_method
        self.preserve_energy = preserve_energy

    def compute_frequency_entropy(self, fft_magnitude: torch.Tensor) -> torch.Tensor:
        """Compute frequency entropy from spectral energy."""
        if self.entropy_method == "spectral":
            prob = fft_magnitude / (fft_magnitude.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
            return entropy

        num_bins = fft_magnitude.shape[-1]
        band_size = max(num_bins // self.num_freq_bands, 1)

        bands = []
        for i in range(self.num_freq_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < self.num_freq_bands - 1 else num_bins
            bands.append(fft_magnitude[..., start:end].sum(dim=-1))
        energy = torch.stack(bands, dim=-1)

        prob = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
        return entropy

    def generate_dynamic_mask(
        self, fft_magnitude: torch.Tensor, entropy: torch.Tensor, length: int
    ) -> torch.Tensor:
        """Map frequency entropy to temporal weights in [0, 1]."""
        entropy_min = entropy.amin(dim=-1, keepdim=True)
        entropy_max = entropy.amax(dim=-1, keepdim=True)
        entropy_norm = (entropy - entropy_min) / (entropy_max - entropy_min + 1e-8)

        mask_strength = torch.sigmoid(entropy_norm * 5 - 2)
        if fft_magnitude.dim() == 4:
            high_freq = fft_magnitude[..., fft_magnitude.shape[-1] // 2 :].sum(dim=-1)
            total = fft_magnitude.sum(dim=-1) + 1e-8
            high_ratio = high_freq / total
            temporal_mask = mask_strength.unsqueeze(-1) * (0.5 + 0.5 * high_ratio)
            mask = F.interpolate(
                temporal_mask.unsqueeze(1), size=length, mode="linear", align_corners=False
            ).squeeze(1)
        else:
            mask = mask_strength.unsqueeze(-1).expand(*mask_strength.shape, length)

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)
        Returns:
            (B, C, L') mask-driven weighted downsampling output
        """
        B, C, L = x.shape

        if self.use_stft:
            x_view = x.reshape(B * C, L)
            n_fft_req = int(self.stft_window_size)
            hop_req = int(self.stft_hop_length)
            min_len = max(8, hop_req + 2, n_fft_req // 2 + 1)
            if L < min_len:
                fft_magnitude = torch.abs(torch.fft.rfft(x, dim=-1))
            else:
                n_fft = max(2, min(n_fft_req, L))
                hop = max(1, min(hop_req, n_fft - 1))
                if n_fft % 2 == 1:
                    n_fft = max(2, n_fft - 1)
                n_rows = x_view.size(0)
                chunk = 2048
                if n_rows <= chunk:
                    x_stft = torch.stft(
                        x_view,
                        n_fft=n_fft,
                        hop_length=hop,
                        return_complex=True,
                        center=True,
                        pad_mode="reflect",
                    )
                else:
                    parts = []
                    for i in range(0, n_rows, chunk):
                        sl = slice(i, min(i + chunk, n_rows))
                        parts.append(
                            torch.stft(
                                x_view[sl],
                                n_fft=n_fft,
                                hop_length=hop,
                                return_complex=True,
                                center=True,
                                pad_mode="reflect",
                            )
                        )
                    x_stft = torch.cat(parts, dim=0)
                fft_magnitude = torch.abs(x_stft).view(B, C, *x_stft.shape[-2:])
        else:
            fft_magnitude = torch.abs(torch.fft.rfft(x, dim=-1))

        entropy = self.compute_frequency_entropy(fft_magnitude)

        dynamic_mask = self.generate_dynamic_mask(fft_magnitude, entropy, L)

        L_pooled = math.ceil(L / self.pool_kernel_size)
        pad_len = L_pooled * self.pool_kernel_size - L
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
            dynamic_mask = F.pad(dynamic_mask, (0, pad_len))

        x_groups = x.view(B, C, L_pooled, self.pool_kernel_size)
        mask_groups = dynamic_mask.view(B, C, L_pooled, self.pool_kernel_size)

        mask_norm = mask_groups / (mask_groups.sum(dim=-1, keepdim=True) + 1e-8)
        mask_norm = mask_norm * self.pool_kernel_size
        out = (x_groups * mask_norm).sum(dim=-1)

        if self.preserve_energy:
            input_energy = x[..., :L].norm(dim=-1, keepdim=True)
            output_energy = out.norm(dim=-1, keepdim=True)
            out = out * (input_energy / (output_energy + 1e-8))

        return out


class FrequencyAwareInterpolation(nn.Module):
    """
    Frequency-aware Dynamic Interpolation (FDI) module.
    
    Uses frequency-domain statistics to dynamically modulate interpolation behavior.
    Implements residual-style correction: y_out = y_linear + α * residual.
    
    Initialized to zero contribution (α=0) for safe training start.
    """
    
    def __init__(
        self,
        target_size: int,
        high_freq_residual_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.target_size = target_size
        self.high_freq_residual_threshold = high_freq_residual_threshold
        
        # Small MLP to map frequency energy → correction weight α ∈ [0, 1]
        # Input: [low_freq_energy, high_freq_energy] from knots
        # Output: scalar α
        self.freq_weight_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Ensure α ∈ [0, 1]
        )
        
        # Initialize to zero contribution: α = 0 at initialization
        # Set the final layer bias to large negative value so Sigmoid outputs ~0
        with torch.no_grad():
            self.freq_weight_net[-2].bias.fill_(-10.0)  # Sigmoid(-10) ≈ 0
    
    def forward(self, y_coarse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_coarse: [B, C, T_coarse] coarse forecast tensor
        Returns:
            y_out: [B, C, T_target] interpolated tensor
        """
        B, C, T_coarse = y_coarse.shape
        
        # Check for invalid inputs
        if T_coarse < 1:
            # Fallback to simple interpolation if input is invalid
            return F.interpolate(
                y_coarse, size=self.target_size, mode='linear', align_corners=False
            )
        
        # Step 1: Apply linear interpolation to get baseline
        y_linear = F.interpolate(
            y_coarse, size=self.target_size, mode='linear', align_corners=False
        )  # [B, C, T_target]
        
        # Check if y_linear contains NaN or Inf
        if torch.isnan(y_linear).any() or torch.isinf(y_linear).any():
            # Return linear interpolation if it's already invalid
            return y_linear
        
        # Step 2: Compute FFT magnitude along time dimension of knots
        # Reshape for FFT: [B*C, T_coarse]
        y_flat = y_coarse.view(B * C, T_coarse)
        fft_mag = torch.abs(torch.fft.rfft(y_flat, dim=-1))  # [B*C, F]
        
        # Step 3: Aggregate frequency energy into low/high components
        num_freqs = fft_mag.shape[-1]
        
        # Handle edge cases when num_freqs is too small
        if num_freqs < 3:
            # For very small frequency components, use uniform distribution
            low_freq_ratio = torch.ones(B * C, 1, device=fft_mag.device, dtype=fft_mag.dtype) * 0.5
            high_freq_ratio = torch.ones(B * C, 1, device=fft_mag.device, dtype=fft_mag.dtype) * 0.5
        else:
            low_freq_end = max(1, num_freqs // 3)  # At least 1 frequency bin
            high_freq_start = min(2 * num_freqs // 3, num_freqs - 1)  # At least 1 frequency bin
            
            # Extract frequency bands
            if low_freq_end > 0:
                low_freq_energy = fft_mag[:, :low_freq_end].mean(dim=-1, keepdim=True)  # [B*C, 1]
            else:
                low_freq_energy = fft_mag[:, 0:1].mean(dim=-1, keepdim=True)
            
            if high_freq_start < num_freqs:
                high_freq_energy = fft_mag[:, high_freq_start:].mean(dim=-1, keepdim=True)  # [B*C, 1]
            else:
                high_freq_energy = fft_mag[:, -1:].mean(dim=-1, keepdim=True)
            
            # Normalize by total energy for stability
            total_energy = fft_mag.mean(dim=-1, keepdim=True) + 1e-8
            low_freq_ratio = low_freq_energy / total_energy
            high_freq_ratio = high_freq_energy / total_energy
        
        # Stack features from knots: [B*C, 2]
        knots_freq_features = torch.cat([low_freq_ratio, high_freq_ratio], dim=-1)
        
        # Step 4: Compute correction weight α
        alpha = self.freq_weight_net(knots_freq_features)  # [B*C, 1]
        alpha = alpha.view(B, C, 1)  # [B, C, 1]
        
        # Step 5: Compute bounded residual correction
        # Smooth the linear interpolation (simple moving average-like effect)
        y_smooth = F.avg_pool1d(
            F.pad(y_linear, (1, 1), mode='replicate'), 
            kernel_size=3, stride=1
        )  # [B, C, T_target]
        residual = y_linear - y_smooth

        if self.high_freq_residual_threshold is not None:
            thr = self.high_freq_residual_threshold
            gate = torch.clamp(
                thr / (high_freq_ratio.view(B, C, 1) + 1e-6), max=1.0
            )
            residual = residual * gate
        
        # Step 6: Apply residual correction
        y_out = y_linear + alpha * residual
        
        return y_out


class _IdentityBasis(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
        use_frequency_interpolation: bool = True,
        fdi_highfreq_threshold: Optional[float] = 0.3,
    ):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features
        self.use_frequency_interpolation = use_frequency_interpolation
        
        # Initialize frequency-aware interpolation if enabled
        if self.use_frequency_interpolation and interpolation_mode == "linear":
            self.freq_interp = FrequencyAwareInterpolation(
                target_size=forecast_size,
                high_freq_residual_threshold=fdi_highfreq_threshold,
            )
        else:
            self.freq_interp = None

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        # Interpolation is performed on default dim=-1 := H
        knots = knots.reshape(len(knots), self.out_features, -1)
        if self.interpolation_mode in ["nearest", "linear"]:
            # Use frequency-aware interpolation if enabled and mode is linear
            if self.use_frequency_interpolation and self.interpolation_mode == "linear" and self.freq_interp is not None:
                # knots: [B, out_features, T_coarse] -> freq_interp expects [B, C, T]
                forecast = self.freq_interp(knots)  # [B, out_features, T_target]
            else:
                # Standard linear/nearest interpolation
                forecast = F.interpolate(
                    knots, size=self.forecast_size, mode=self.interpolation_mode
                )
            # forecast = forecast[:,0,:]
        elif "cubic" in self.interpolation_mode:
            if self.out_features > 1:
                raise Exception(
                    "Cubic interpolation not available with multiple outputs."
                )
            batch_size = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros(
                (len(knots), self.forecast_size), device=knots.device
            )
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]  # [B,None,H,H] -> [B,H]
            forecast = forecast[:, None, :]  # [B,H] -> [B,None,H]

        # [B,Q,H] -> [B,H,Q]
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

POOLING = ["MaxPool1d", "AvgPool1d", "frequency_dynamic"]


class FHsdBlock(nn.Module):
    """
    FHsd block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
        freq_pooling_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        input_size = (
            pooled_hist_size
            + hist_input_size * pooled_hist_size
            + futr_input_size * pooled_futr_size
            + stat_input_size
        )

        self.dropout_prob = dropout_prob
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size
        self.pooling_mode = pooling_mode
        self.pool_kernel_size = n_pool_kernel_size

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        if pooling_mode == "frequency_dynamic":
            freq_kwargs = freq_pooling_kwargs.copy() if freq_pooling_kwargs else {}
            self.pooling_layer = FrequencyEntropyPooling(
                pool_kernel_size=n_pool_kernel_size, **freq_kwargs
            )
        else:
            self.pooling_layer = getattr(nn, pooling_mode)(
                kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
            )

        # Block MLPs
        hidden_layers = [
            nn.Linear(in_features=input_size, out_features=mlp_units[0][0])
        ]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                # raise NotImplementedError('dropout')
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batch_size = len(insample_y)
        if self.hist_input_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            hist_exog = self.pooling_layer(hist_exog)
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, hist_exog.reshape(batch_size, -1)), dim=1
            )

        if self.futr_input_size > 0:
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            futr_exog = self.pooling_layer(futr_exog)
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, futr_exog.reshape(batch_size, -1)), dim=1
            )

        if self.stat_input_size > 0:
            insample_y = torch.cat(
                (insample_y, stat_exog.reshape(batch_size, -1)), dim=1
            )

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class FHsd(BaseModel):
    """FHsd

    The Neural Hierarchical Interpolation for Time Series (NHITS), is an MLP-based deep
    neural architecture with backward and forward residual links. NHITS tackles volatility and
    memory complexity challenges, by locally specializing its sequential predictions into
    the signals frequencies with hierarchical interpolation and pooling.

    Args:
        h (int): Forecast horizon.
        input_size (int): autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].
        futr_exog_list (str list): future exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        stat_exog_list (str list): static exogenous columns.
        exclude_insample_y (bool): the model skips the autoregressive features y[t-input_size:t] if True.
        stack_types (List[str]): stacks list in the form N * ['identity'], to be deprecated in favor of `n_stacks`. Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).
        n_blocks (List[int]): Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).
        mlp_units (List[List[int]]): Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).
        n_pool_kernel_size (List[int]): list with the size of the windows to take a max/avg over. Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).
        n_freq_downsample (List[int]): list with the stack's coefficients (inverse expressivity ratios). Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).
        pooling_mode (str): input pooling module from ['MaxPool1d', 'AvgPool1d', 'frequency_dynamic']. 'frequency_dynamic' enables FDAM (Frequency-aware Dynamic Pooling).
        freq_pooling_kwargs (dict, optional): Additional arguments for FrequencyEntropyPooling when pooling_mode='frequency_dynamic', such as use_stft, num_freq_bands, etc.
        interpolation_mode (str): interpolation basis from ['linear', 'nearest', 'cubic'].
        use_frequency_interpolation (bool): If True, use frequency-aware dynamic interpolation instead of fixed linear interpolation. Default True.
        fdi_highfreq_threshold (float, optional): If set, dampens the FDI residual using high-frequency energy ratio (fixed-threshold style denoising). Default None (disabled).
        enable_self_distill (bool): whether to enable block-level self-distillation. Default True.
        self_distill_weight (float): weight for self-distillation loss. Default 0.1.
        dropout_prob_theta (float): Float between (0, 1). Dropout for NHITS basis.
        activation (str): activation from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'].
        learning_rate (float): Learning rate between (0, 1).
        num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
        val_check_steps (int): Number of training steps between every validation loss check.
        batch_size (int): number of different series in each batch.
        valid_batch_size (int): number of different series in each validation and test batch, if None uses batch_size.
        windows_batch_size (int): number of windows to sample in each training batch, default uses all.
        inference_windows_batch_size (int): number of windows to sample in each inference batch, -1 uses all.
        start_padding_enabled (bool): if True, the model will pad the time series with zeros at the beginning, by input size.
        training_data_availability_threshold (Union[float, List[float]]): minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).
        step_size (int): step size between each window of temporal data.
        scaler_type (str): type of scaler for temporal inputs normalization see [temporal scalers](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_scalers.py).
        random_seed (int): random_seed for pytorch initializer and numpy generators.
        drop_last_loader (bool): if True `TimeSeriesDataLoader` drops last non-full batch.
        alias (str): optional,  Custom name of the model.
        optimizer (Subclass of 'torch.optim.Optimizer'): optional, user specified optimizer instead of the default choice (Adam).
        optimizer_kwargs (dict): optional, list of parameters used by the user specified `optimizer`.
        lr_scheduler (Subclass of 'torch.optim.lr_scheduler.LRScheduler'): optional, user specified lr_scheduler instead of the default choice (StepLR).
        lr_scheduler_kwargs (dict): optional, list of parameters used by the user specified `lr_scheduler`.
        dataloader_kwargs (dict): optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`.
        **trainer_kwargs (int):  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

    References:
        - [Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Artur Dubrawski (2023). "NHITS: Neural Hierarchical Interpolation for Time Series Forecasting". Accepted at the Thirty-Seventh AAAI Conference on Artificial Intelligence.](https://arxiv.org/abs/2201.12886)
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h,
        input_size,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        stack_types: list = ["identity", "identity", "identity"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        freq_pooling_kwargs: Optional[Dict[str, Any]] = None,
        interpolation_mode: str = "linear",
        use_frequency_interpolation: bool = True,
        fdi_highfreq_threshold: Optional[float] = None,
        dropout_prob_theta=0.0,
        activation="ReLU",
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader=False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        enable_self_distill: bool = True,
        self_distill_weight: float = 0.1,
        **trainer_kwargs,
    ):

        # Remove use_frequency_interpolation from trainer_kwargs if present
        # (it's a model parameter, not a trainer parameter)
        trainer_kwargs.pop('use_frequency_interpolation', None)

        # Inherit BaseWindows class
        super(FHsd, self).__init__(
            h=h,
            input_size=input_size,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        # Architecture
        blocks = self.create_stack(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            futr_input_size=self.futr_exog_size,
            hist_input_size=self.hist_exog_size,
            stat_input_size=self.stat_exog_size,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            use_frequency_interpolation=use_frequency_interpolation,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
            freq_pooling_kwargs=freq_pooling_kwargs,
            fdi_highfreq_threshold=fdi_highfreq_threshold,
        )
        self.blocks = torch.nn.ModuleList(blocks)
        
        # Self-distillation parameters
        # Deeper NHITS blocks capture longer temporal dependencies and richer global patterns.
        # Self-distillation encourages earlier blocks to align with these representations,
        # improving stability and generalization without modifying the architecture.
        self.enable_self_distill = enable_self_distill
        self.self_distill_weight = self_distill_weight

    def create_stack(
        self,
        h,
        input_size,
        stack_types,
        n_blocks,
        mlp_units,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        use_frequency_interpolation,
        dropout_prob_theta,
        activation,
        futr_input_size,
        hist_input_size,
        stat_input_size,
        freq_pooling_kwargs,
        fdi_highfreq_threshold: Optional[float] = None,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                assert (
                    stack_types[i] == "identity"
                ), f"Block type {stack_types[i]} not found!"

                n_theta = input_size + self.loss.outputsize_multiplier * max(
                    h // n_freq_downsample[i], 1
                )
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=self.loss.outputsize_multiplier,
                    interpolation_mode=interpolation_mode,
                    use_frequency_interpolation=use_frequency_interpolation,
                    fdi_highfreq_threshold=fdi_highfreq_threshold,
                )

                nbeats_block = FHsdBlock(
                    h=h,
                    input_size=input_size,
                    futr_input_size=futr_input_size,
                    hist_input_size=hist_input_size,
                    stat_input_size=stat_input_size,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                    freq_pooling_kwargs=freq_pooling_kwargs,
                )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list

    def training_step(self, batch, batch_idx):
        """
        Training step with block-level self-distillation.
        
        Mathematical formulation:
        L_total = L_forecast + λ · Σ || ŷ_student^k − ŷ_teacher^k ||²
        
        where:
        - Teacher = deeper blocks
        - Student = shallower blocks
        - λ = self_distill_weight
        """
        # Set horizon to h_train in case of recurrent model to speed up training
        if self.RECURRENT:
            self.h = self.h_train

        # windows: [Ws, L + h, C, n_series] or [Ws, L + h, C]
        y_idx = batch["y_idx"]

        temporal_cols = batch["temporal_cols"]
        windows_temporal, static, static_cols = self._create_windows(
            batch, step="train"
        )
        windows = self._sample_windows(
            windows_temporal, static, static_cols, temporal_cols, step="train"
        )
        original_outsample_y = torch.clone(
            windows["temporal"][:, self.input_size :, y_idx]
        )
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L, n_series]
            insample_mask=insample_mask,  # [Ws, L, n_series]
            futr_exog=futr_exog,  # univariate: [Ws, L, F]; multivariate: [Ws, F, L, n_series]
            hist_exog=hist_exog,  # univariate: [Ws, L, X]; multivariate: [Ws, X, L, n_series]
            stat_exog=stat_exog,
        )  # univariate: [Ws, S]; multivariate: [n_series, S]

        # Model Predictions
        # Get block forecasts if self-distillation is enabled
        if self.enable_self_distill:
            output, block_forecasts = self(windows_batch, return_block_forecasts=True)
        else:
            output = self(windows_batch, return_block_forecasts=False)
        
        output = self.loss.domain_map(output)

        # Compute forecast loss (original loss)
        if self.loss.is_distribution_output:
            y_loc, y_scale = self._get_loc_scale(y_idx)
            outsample_y = original_outsample_y
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            forecast_loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            forecast_loss = self.loss(
                y=outsample_y, y_hat=output, y_insample=insample_y, mask=outsample_mask
            )

        # Compute self-distillation loss
        # L_total = L_forecast + λ · Σ || ŷ_student^k − ŷ_teacher^k ||²
        distill_loss = torch.tensor(0.0, device=forecast_loss.device, dtype=forecast_loss.dtype)
        if self.enable_self_distill and len(block_forecasts) > 1:
            # Define Teacher/Student block alignment
            # block_forecasts[0] is the initial naive forecast
            # block_forecasts[1:] are cumulative forecasts after each block
            actual_blocks = block_forecasts[1:]  # Skip initial naive forecast
            n_actual_blocks = len(actual_blocks)
            
            if n_actual_blocks > 1:
                # Split into student (first half) and teacher (second half)
                n_student = n_actual_blocks // 2
                n_teacher = n_actual_blocks - n_student
                
                student_forecasts = actual_blocks[:n_student]
                teacher_forecasts = actual_blocks[-n_teacher:]
                
                # Align student and teacher forecasts for distillation
                # Pair them: student[i] with teacher[-(i+1)] (reverse order pairing)
                num_pairs = min(len(student_forecasts), len(teacher_forecasts))
                
                for i in range(num_pairs):
                    student_pred = student_forecasts[i]
                    # Get corresponding teacher (from the end, in reverse order)
                    teacher_idx = len(teacher_forecasts) - 1 - i
                    teacher_pred = teacher_forecasts[teacher_idx].detach()  # Detach to stop gradient
                    
                    # Compute MSE loss between student and teacher cumulative forecasts
                    # Handle dimensions: [B, H, 1] or [B, H]
                    student_pred = student_pred.squeeze(-1) if student_pred.dim() > 2 else student_pred
                    teacher_pred = teacher_pred.squeeze(-1) if teacher_pred.dim() > 2 else teacher_pred
                    
                    distill_loss += F.mse_loss(student_pred, teacher_pred)
                
                distill_loss = distill_loss / num_pairs if num_pairs > 0 else distill_loss

        # Total loss: L_total = L_forecast + λ · L_distill
        total_loss = forecast_loss + self.self_distill_weight * distill_loss

        if torch.isnan(total_loss):
            print("Model Parameters", self.hparams)
            print("insample_y", torch.isnan(insample_y).sum())
            print("outsample_y", torch.isnan(outsample_y).sum())
            print("forecast_loss", forecast_loss.item())
            print("distill_loss", distill_loss.item())
            raise Exception("Loss is NaN, training stopped.")

        train_loss_log = total_loss.detach().item()
        self.log(
            "train_loss",
            train_loss_log,
            batch_size=outsample_y.size(0),
            prog_bar=True,
            on_epoch=True,
        )
        if self.enable_self_distill:
            self.log(
                "distill_loss",
                distill_loss.detach().item(),
                batch_size=outsample_y.size(0),
                prog_bar=False,
                on_epoch=True,
            )
        self.train_trajectories.append((self.global_step, train_loss_log))

        self.h = self.horizon_backup

        return total_loss

    def forward(self, windows_batch, return_block_forecasts=False):
        """
        Forward pass of FHsd model.
        
        Args:
            windows_batch: Input batch dictionary
            return_block_forecasts: If True, returns individual block forecasts for self-distillation.
                                   Only used during training when enable_self_distill=True.
        
        Returns:
            If return_block_forecasts=False: final forecast (same as original)
            If return_block_forecasts=True: (final_forecast, list of block_forecasts)
        """
        # Parse windows_batch
        insample_y = windows_batch["insample_y"].squeeze(-1).contiguous()
        insample_mask = windows_batch["insample_mask"].squeeze(-1).contiguous()
        futr_exog = windows_batch["futr_exog"]
        hist_exog = windows_batch["hist_exog"]
        stat_exog = windows_batch["stat_exog"]

        # insample
        residuals = insample_y.flip(dims=(-1,))  # backcast init
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:, None]  # Level with Naive1
        
        # Initialize block_forecasts list only if needed
        if return_block_forecasts or self.decompose_forecast:
            # For self-distillation: store cumulative forecasts (not incremental)
            # For decomposition: store incremental forecasts (original behavior)
            if return_block_forecasts:
                # Store cumulative forecasts for self-distillation
                block_forecasts = [forecast.repeat(1, self.h, 1)]
            else:
                # Store incremental forecasts for decomposition (original behavior)
                block_forecasts = [forecast.repeat(1, self.h, 1)]
        
        # Process each block
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            
            # Cache block forecasts if needed for self-distillation or decomposition
            if return_block_forecasts or self.decompose_forecast:
                if return_block_forecasts:
                    # For self-distillation: store cumulative forecast after each block
                    block_forecasts.append(forecast.clone())
                else:
                    # For decomposition: store incremental forecast (original behavior)
                    block_forecasts.append(block_forecast)

        if return_block_forecasts:
            # Return both final forecast and individual block forecasts for self-distillation
            return forecast, block_forecasts
        elif self.decompose_forecast:
            # (n_batch, n_blocks, h, output_size)
            block_forecasts = torch.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2, 3)
            block_forecasts = block_forecasts.squeeze(-1)  # univariate output
            return block_forecasts
        else:
            return forecast
