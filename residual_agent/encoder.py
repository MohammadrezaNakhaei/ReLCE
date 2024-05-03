import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import MutableMapping, Optional, Tuple
import numpy as np


class ConvEncoder(nn.Module):
    def __init__(
        self, 
        state_dim:int, 
        action_dim:int, 
        seq_len:int, 
        latent_dim:int = 16,
        hidden_dim:int = 16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.state_emb = nn.Linear(state_dim, hidden_dim, bias=False)
        self.action_emb = nn.Linear(action_dim, hidden_dim, bias=False)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, bias=False),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, bias=False),
            nn.ReLU(), 
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(), 
            nn.Flatten(),
        ) 
        tst_sample = torch.ones((1, hidden_dim, 2*seq_len)) 
        out_conv = self._get_shape(tst_sample)
        self.output = nn.Linear(out_conv, latent_dim, bias=False)

    def _get_shape(self, sample):
        return np.prod(self.conv(sample).shape)

    def forward(self, states, actions, *kargs, **kwargs):
        assert states.ndim == 3, 'state should have 3 dimenstions: batch, time, states'
        assert actions.ndim == 3, 'action should have 3 dimenstions: batch, time, actions'
        # B, T, dim
        batch_size, seq_len, _ = states.shape
        assert seq_len == self.seq_len
        state_emb = self.state_emb(states)
        action_emb = self.action_emb(actions)
        # [batch_size, seq_len * 2, emb_dim], ( s_0, a_0, s_1, a_1, ...)-> batch_size, seq_len, emb_dim
        sequence = torch.stack([state_emb, action_emb], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_len, self.hidden_dim)
        sequence = sequence.permute(0, 2, 1) # batch_size, emb_dim, seq_len
        out_conv = self.conv(sequence)
        return self.output(out_conv)
    

class Deocdeer(nn.Module):
    def __init__(self, 
                 state_dim:int, 
                 action_dim:int, 
                 latent_dim:int, 
                 hidden_dims:Optional[Tuple[int], List[int]]=(256, 256)
    ):
        super().__init__()
        n_layers = [state_dim+action_dim+latent_dim, *hidden_dims, state_dim]
        layers = []
        for l1, l2 in zip(n_layers[:-1], n_layers[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU())
        layers.pop()
        self.fc = nn.Sequential(*layers)

    def forward(self, states:torch.Tensor, actions:torch.Tensor, latents:torch.Tensor):
        total_input = torch.cat([states, actions, latents], dim=-1)
        return self.fc(total_input)


class EncoderModule():
    def __init__(self, 
        state_dim:int, 
        action_dim:int, 
        seq_len:int, 
        latent_dim:int = 16,
        encoder_hidden:int = 16, 
        decoder_hidden_dims:Optional[Tuple[int], List[int]]=(256, 256), 
        k_steps:int = 5,
        learning_rate: float = 0.0001, 
        omega_consistency: float = 0.1, # hyper-parameter for weighting consistency objective with respect to predictions
        mu: torch.Tensor = 0, # normalizing the output
        std: torch.Tensor = 1, 
        device:str = 'cpu', 
    ):
        self.device = torch.device(device)
        self.encoder = ConvEncoder(state_dim, action_dim, seq_len, latent_dim, encoder_hidden).to(self.device)
        self.decoder = Deocdeer(state_dim, action_dim, latent_dim, decoder_hidden_dims).to(self.device)
        self.optmizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), learning_rate)
        self.k = k_steps
        self.seq_len = self.encoder.seq_len
        self.mu = mu
        self.std = std
        self.omega = omega_consistency

    def learn_batch(self, batch:MutableMapping[str, torch.Tensor], ):
        # currently no contrastive loss, only consider the mean of latent space
        seq_states = batch['seq_states']
        seq_actions = batch['seq_actions']
        seq_masks = batch['seq_masks']
        state = batch['state']
        action = batch['action']
        next_state = batch['next_state']
        B, N, T, _ = seq_states.shape

        # assert self.k + self.seq_len == T
        idx = np.random.randint(N)
        latents = self.encode_multiple(seq_states[:, :, :self.seq_len, :], seq_actions[:, :, :self.seq_len, :], seq_masks[:, :, :self.seq_len])

        predicted_state = self.decoder(state, action, latents[:,idx])
        target = self._normalize(next_state-state)

        self.optimizer.zero_grad()
        loss_sim = similarity_loss(latents) # similar to N points
        loss_pred = F.mse_loss(predicted_state, target)
        
        # k-step prediction loss
        start_ind = self.seq_len-1
        state = seq_states[:, :, start_ind,]
        for j in range(self.k):
            pred_diff = self.decoder(state, seq_actions[:, :, start_ind+j], latents)
            target = self._normalize(seq_states[:,:,start_ind+j+1]- seq_states[:,:,start_ind+j])
            loss_pred += F.mse_loss(pred_diff, target)
            state = self._unnormalize(pred_diff)+state
        loss = loss_pred + self.alpha_sim*loss_sim 
        loss.backward()
        self.optimizer.step()
        return {
            'loss/encoder_prediction': loss_pred.item(), 
            'loss/encoder_similarity':loss_sim.item(),
            }
    
    @torch.no_grad()
    def encode(self, seq_state:torch.Tensor, seq_action:torch.Tensor, time_step:torch.Tensor):
        assert seq_state.ndim==3 # batch, seq, state
        return self.encoder(seq_state, seq_action, time_step)    

    def encode_multiple(self, seq_states:torch.Tensor, seq_actions:torch.Tensor, seq_masks:torch.Tensor):
        assert seq_states.ndim==4
        seq_states = seq_states[:,:,:self.seq_len]
        seq_actions = seq_actions[:,:,:self.seq_len]
        seq_masks = seq_masks[:,:,:self.seq_len]
        B, N, T, _ = seq_states.shape
        device = seq_states.device
        timesteps = torch.arange(0, T, device=device).repeat(N*B, 1)
        latents = self.encoder(
            seq_states.view(B*N, T, -1), 
            seq_actions.view(B*N, T, -1),
            timesteps,
            seq_masks.view(B*N, T).bool(),
            )
        latents = latents.view(B, N, -1)
        return latents  
    
    def __call__(self, seq_state:torch.Tensor, seq_action:torch.Tensor, time_step:torch.Tensor):
        return self.encode(seq_state, seq_action, time_step)

    @torch.no_grad()
    def encode(self, seq_state:torch.Tensor, seq_action:torch.Tensor, time_step:torch.Tensor):
        assert seq_state.ndim==3 # batch, seq, state
        return self.encoder(seq_state, seq_action, time_step)
    
    def _normalize(self, tensor):
        return (tensor-self.mu)/self.std
    
    def _unnormalize(self, tensor):
        return self.std*tensor+self.mu
    
    
def cosine(pred, target, reduce=False):
    x = F.normalize(pred, dim=-1, p=2)
    y = F.normalize(target, dim=-1, p=2)
    return 2 - 2*(x * y).sum(dim=-1, keepdim=(not reduce))


def similarity_loss(latents:torch.Tensor):
    assert latents.ndim == 3
    B, N, _ = latents.shape
    assert N>1
    device = latents.device
    latents = F.normalize(latents, dim=-1, p=2)
    masks = torch.eye(N, device = device).unsqueeze(0).repeat(B, 1, 1)
    sum_similarity = latents@latents.permute(0, 2, 1) # B, N, N
    sum_similarity = (1-masks)*sum_similarity
    loss = -1/(N-1)*sum_similarity.sum(dim=(1,2)).mean()
    return loss


