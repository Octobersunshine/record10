import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, height, width)
        c = torch.zeros(batch_size, self.hidden_dim, height, width)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_sizes: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)
        
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.layers.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_sizes[i]))
    
    def forward(self, x: torch.Tensor, hidden_states: Optional[List[Tuple]] = None) -> Tuple[torch.Tensor, List[Tuple]]:
        b, t, c, h, w = x.shape
        
        if hidden_states is None:
            hidden_states = []
            for layer in self.layers:
                hidden_states.append(layer.init_hidden(b, h, w))
        
        layer_output = None
        for t_step in range(t):
            x_t = x[:, t_step, :, :, :]
            for layer_idx, layer in enumerate(self.layers):
                h_t, c_t = hidden_states[layer_idx]
                h_next, c_next = layer(x_t, h_t, c_t)
                hidden_states[layer_idx] = (h_next, c_next)
                x_t = h_next
            layer_output = x_t if layer_output is None else torch.cat([layer_output, x_t], dim=0)
        
        return layer_output, hidden_states


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.conv_lstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_sizes=[3, 3, 3]
        )
        
    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        _, hidden_states = self.conv_lstm(x)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, output_dim: int = 1, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        self.output_dim = output_dim
        self.conv_lstm = ConvLSTM(
            input_dim=hidden_dims[0],
            hidden_dims=hidden_dims,
            kernel_sizes=[3, 3, 3]
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_dim, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, hidden_states: List[Tuple], output_length: int, 
               height: int, width: int, batch_size: int) -> torch.Tensor:
        outputs = []
        x_t = torch.zeros(batch_size, self.conv_lstm.input_dim, height, width)
        
        for t in range(output_length):
            for layer_idx, layer in enumerate(self.conv_lstm.layers):
                h_t, c_t = hidden_states[layer_idx]
                h_next, c_next = layer(x_t, h_t, c_t)
                hidden_states[layer_idx] = (h_next, c_next)
                x_t = h_next
            
            output_frame = self.output_conv(x_t)
            outputs.append(output_frame)
        
        return torch.stack(outputs, dim=1)


class Generator(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1, 
                 hidden_dims: List[int] = [64, 128, 256],
                 num_history_frames: int = 6, num_future_frames: int = 20):
        super().__init__()
        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames
        
        self.encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims)
        decoder_dims = list(reversed(hidden_dims))
        self.decoder = Decoder(output_dim=output_dim, hidden_dims=decoder_dims)
        
        self.condition_proj = nn.Conv2d(input_dim * num_history_frames, hidden_dims[0], kernel_size=3, padding=1)
        
    def forward(self, history: torch.Tensor, future_length: Optional[int] = None) -> torch.Tensor:
        b, t, c, h, w = history.shape
        
        if future_length is None:
            future_length = self.num_future_frames
        
        hidden_states = self.encoder(history)
        
        last_frame = history[:, -1, :, :, :]
        condition = history.reshape(b, t * c, h, w)
        decoder_input = self.condition_proj(condition)
        
        outputs = []
        current_hidden = hidden_states
        current_input = decoder_input
        
        for step in range(future_length):
            for layer_idx, layer in enumerate(self.decoder.conv_lstm.layers):
                h_t, c_t = current_hidden[layer_idx]
                h_next, c_next = layer(current_input, h_t, c_t)
                current_hidden[layer_idx] = (h_next, c_next)
                current_input = h_next
            
            output_frame = self.decoder.output_conv(current_input)
            outputs.append(output_frame)
        
        return torch.stack(outputs, dim=1)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 1, sequence_length: int = 26):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(input_dim, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = sequence.transpose(1, 2)
        features = self.conv_layers(x)
        output = self.fc(features)
        return output


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class RadarGAN(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dims: List[int] = [64, 128, 256],
                 num_history_frames: int = 6, num_future_frames: int = 20,
                 lambda_adv: float = 0.01, lambda_perceptual: float = 0.1):
        super().__init__()
        self.generator = Generator(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames
        )
        self.discriminator = Discriminator(
            input_dim=input_dim,
            sequence_length=num_history_frames + num_future_frames
        )
        
        self.lambda_adv = lambda_adv
        self.lambda_perceptual = lambda_perceptual
        
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.bce_loss = nn.BCELoss()
        
    def generator_loss(self, history: torch.Tensor, future: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_future = self.generator(history)
        
        mse_loss = self.mse_loss(pred_future, future)
        
        full_sequence = torch.cat([history, pred_future], dim=1)
        fake_preds = self.discriminator(full_sequence)
        adv_loss = self.bce_loss(fake_preds, torch.ones_like(fake_preds))
        
        perceptual_loss = self.perceptual_loss(pred_future, future)
        
        total_loss = mse_loss + self.lambda_adv * adv_loss + self.lambda_perceptual * perceptual_loss
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'adversarial': adv_loss,
            'perceptual': perceptual_loss,
            'predictions': pred_future
        }
    
    def discriminator_loss(self, history: torch.Tensor, future: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            pred_future = self.generator(history)
        
        real_sequence = torch.cat([history, future], dim=1)
        fake_sequence = torch.cat([history, pred_future], dim=1)
        
        real_preds = self.discriminator(real_sequence)
        fake_preds = self.discriminator(fake_sequence)
        
        real_loss = self.bce_loss(real_preds, torch.ones_like(real_preds))
        fake_loss = self.bce_loss(fake_preds, torch.zeros_like(fake_preds))
        
        total_loss = (real_loss + fake_loss) / 2
        
        return {
            'total': total_loss,
            'real': real_loss,
            'fake': fake_loss
        }


class HybridExtrapolator:
    def __init__(self, optical_flow_model, gan_model: Optional[RadarGAN] = None,
                 flow_weight: float = 0.5, gan_weight: float = 0.5,
                 switch_lead_time: int = 60):
        self.optical_flow_model = optical_flow_model
        self.gan_model = gan_model
        self.flow_weight = flow_weight
        self.gan_weight = gan_weight
        self.switch_lead_time = switch_lead_time
        
    def predict(self, history_frames: List[np.ndarray], lead_times: List[int],
                time_interval: int = 6) -> Dict[int, np.ndarray]:
        num_frames = len(history_frames)
        h, w = history_frames[0].shape
        
        flow_predictions = {}
        if self.optical_flow_model is not None:
            flow_extrapolator = self.optical_flow_model
            max_lead = max(lead_times)
            num_steps = max_lead // time_interval
            flow_results = flow_extrapolator.extrapolate_enhanced(
                history_frames, steps=num_steps, time_interval=time_interval
            )
            
            for lt in lead_times:
                step_idx = (lt // time_interval) - 1
                if step_idx < len(flow_results):
                    flow_predictions[lt] = flow_results[step_idx]
                else:
                    flow_predictions[lt] = flow_results[-1]
        
        gan_predictions = {}
        if self.gan_model is not None:
            self.gan_model.eval()
            with torch.no_grad():
                history_tensor = torch.tensor(np.array(history_frames), dtype=torch.float32)
                history_tensor = history_tensor.unsqueeze(1).unsqueeze(0)
                
                max_lead = max(lead_times)
                num_steps = max_lead // time_interval
                gan_results = self.gan_model.generator(history_tensor, future_length=num_steps)
                gan_results = gan_results.squeeze(0).squeeze(1).numpy()
                
                for lt in lead_times:
                    step_idx = (lt // time_interval) - 1
                    if step_idx < len(gan_results):
                        gan_predictions[lt] = gan_results[step_idx]
                    else:
                        gan_predictions[lt] = gan_results[-1]
        
        final_predictions = {}
        for lt in lead_times:
            if lt <= self.switch_lead_time:
                if lt in flow_predictions and lt in gan_predictions:
                    final_predictions[lt] = (
                        self.flow_weight * flow_predictions[lt] +
                        self.gan_weight * gan_predictions[lt]
                    )
                elif lt in flow_predictions:
                    final_predictions[lt] = flow_predictions[lt]
                else:
                    final_predictions[lt] = gan_predictions.get(lt, np.zeros((h, w)))
            else:
                if lt in gan_predictions:
                    final_predictions[lt] = gan_predictions[lt]
                elif lt in flow_predictions:
                    final_predictions[lt] = flow_predictions[lt]
                else:
                    final_predictions[lt] = np.zeros((h, w))
        
        return final_predictions


def train_step(model: RadarGAN, history: torch.Tensor, future: torch.Tensor,
               g_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, float]:
    model.train()
    
    d_optimizer.zero_grad()
    d_losses = model.discriminator_loss(history, future)
    d_losses['total'].backward()
    d_optimizer.step()
    
    g_optimizer.zero_grad()
    g_losses = model.generator_loss(history, future)
    g_losses['total'].backward()
    g_optimizer.step()
    
    return {
        'g_total': g_losses['total'].item(),
        'g_mse': g_losses['mse'].item(),
        'g_adv': g_losses['adversarial'].item(),
        'g_perceptual': g_losses['perceptual'].item(),
        'd_total': d_losses['total'].item(),
        'd_real': d_losses['real'].item(),
        'd_fake': d_losses['fake'].item()
    }
