""" Adapted from https://github.com/ankitkv/TD-VAE"""
import numpy
import torch
from torch import nn
from torch.nn import functional as F


class DBlock(nn.Module):
    """ A basic building block for computing parameters of a normal distribution.
    Corresponds to D in the appendix."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        t = torch.tanh(self.fc1(input_))
        t = t * torch.sigmoid(self.fc2(input_))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    """

    def __init__(self, z_size, hidden_size, x_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class TDVAE(nn.Module):
    """ The full TD-VAE model with jumpy prediction.
    """

    def __init__(self, x_size,
                 input_size: int = 1024,
                 belief_size: int = 1024,
                 z_posterior_size: int = 1024,
                 num_layers: int = 2,
                 samples_per_seq: int = 200,
                 t_diff_min: int = 1,
                 t_diff_max: int = 5):
        super().__init__()
        self.num_layers = num_layers
        self.samples_per_seq = samples_per_seq
        self.t_diff_min = t_diff_min
        self.t_diff_max = t_diff_max

        # Multilayer LSTM for aggregating belief states
        self.b_belief_rnn = nn.LSTM(input_size=input_size, hidden_size=belief_size, num_layers=num_layers)

        # Multilayer state model is used. Sampling is done by sampling higher layers first.
        self.z_posterior_belief = nn.ModuleList(
            [DBlock(belief_size + (z_posterior_size if layer < num_layers - 1 else 0), 50, z_posterior_size)
             for layer in range(num_layers)])

        # Given belief and state at time t2, infer the state at time t1
        self.z1_z2_b1_inference = nn.ModuleList([DBlock(
            belief_size + num_layers * z_posterior_size + (z_posterior_size if layer < num_layers - 1 else 0), 50,
            z_posterior_size) for layer in range(num_layers)])

        # Given the state at time t1, model state at time t2 through state transition
        self.z2_z1_prediction = nn.ModuleList([DBlock(
            num_layers * z_posterior_size + (z_posterior_size if layer < num_layers - 1 else 0), 50, z_posterior_size)
            for layer in range(num_layers)])

        # state to observation
        self.x_z_decoder = Decoder(num_layers * z_posterior_size, 200, x_size)

    def forward(self, x, mask=None):
        # TODO mask so does not go beyond the length of the batch.

        # Sample the current and future time points.
        t1 = torch.randint(0, x.size(1) - self.t_diff_max, (self.samples_per_seq, x.size(0)), device=x.device)
        t2 = t1 + torch.randint(self.t_diff_min, self.t_diff_max + 1, (self.samples_per_seq, x.size(0)),
                                device=x.device)

        # Truncate the sequence if not required to the end.
        x = x[:, :t2.max() + 1]

        # Run LSTM to get belief states.
        b1, b2 = self._beliefs(x, t1, t2)

        # Sample the z posteriors (or states of the world for b1 and b2)
        qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2, qb_z2_b2_mus, qb_z2_b2_logvars, qb_z2_b2s = self.sample_posterior_z(b2)
        pb_z1_b1_mu, pb_z1_b1_logvar, pb_z1_b1, pb_z1_b1_mus, pb_z1_b1_logvars, pb_z1_b1s = self.sample_posterior_z(b1)

        # Infer the smoothing distibution, i.e. the state of the world at b1.
        qs_z1_z2_b1, qs_z1_z2_b1_logvar, qs_z1_z2_b1_mu, qs_z1_z2_b1s = self._q_inference(b1, qb_z2_b2)

        # Predict the future z from the smoothing distribution and lower level b state.
        pt_z2_z1_logvar, pt_z2_z1_mu = self._z_prediction(qb_z2_b2s, qs_z1_z2_b1)

        # This decoder is grounding the prediction in the data x2 from t2.
        pd_x2_z2 = self.x_z_decoder(qb_z2_b2)

        return (x, t2, qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar, qb_z2_b2_mu, qb_z2_b2_logvar,
                qb_z2_b2, pt_z2_z1_mu, pt_z2_z1_logvar, pd_x2_z2)

    def _z_prediction(self, qb_z2_b2s, qs_z1_z2_b1):
        ''' Predicts forwards to z2 from z1.
            p_T(z2 | z1), also conditions on q_B(z2) from higher layer
        '''
        pt_z2_z1_mus, pt_z2_z1_logvars = [], []
        for layer in range(self.num_layers - 1, -1, -1):
            if layer == self.num_layers - 1:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1_prediction[layer](qs_z1_z2_b1)
            else:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1_prediction[layer](
                    torch.cat([qs_z1_z2_b1, qb_z2_b2s[layer + 1]], dim=1))
            pt_z2_z1_mus.insert(0, pt_z2_z1_mu)
            pt_z2_z1_logvars.insert(0, pt_z2_z1_logvar)
        pt_z2_z1_mu = torch.cat(pt_z2_z1_mus, dim=1)
        pt_z2_z1_logvar = torch.cat(pt_z2_z1_logvars, dim=1)
        return pt_z2_z1_logvar, pt_z2_z1_mu

    def _q_inference(self, b1, qb_z2_b2):
        ''' This is the backwards q inference from the paper.
        The projection from t2 back to the state of the world at t1.
          q_S(z1 | z2, b1, b2) ~= q_S(z1 | z2, b1)
        '''
        qs_z1_z2_b1_mus, qs_z1_z2_b1_logvars, qs_z1_z2_b1s = [], [], []
        for layer in range(self.num_layers - 1, -1, -1):
            if layer == self.num_layers - 1:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z1_z2_b1_inference[layer](
                    torch.cat([qb_z2_b2, b1[:, layer]], dim=1))
            else:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z1_z2_b1_inference[layer](torch.cat([qb_z2_b2, b1[:, layer],
                                                                                               qs_z1_z2_b1], dim=1))
            qs_z1_z2_b1_mus.insert(0, qs_z1_z2_b1_mu)
            qs_z1_z2_b1_logvars.insert(0, qs_z1_z2_b1_logvar)

            qs_z1_z2_b1 = reparameterize_gaussian(qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, self.training)
            qs_z1_z2_b1s.insert(0, qs_z1_z2_b1)
        qs_z1_z2_b1_mu = torch.cat(qs_z1_z2_b1_mus, dim=1)
        qs_z1_z2_b1_logvar = torch.cat(qs_z1_z2_b1_logvars, dim=1)
        qs_z1_z2_b1 = torch.cat(qs_z1_z2_b1s, dim=1)
        return qs_z1_z2_b1, qs_z1_z2_b1_logvar, qs_z1_z2_b1_mu, qs_z1_z2_b1s

    def sample_posterior_z(self, b):
        ''' Samples the posterior for a belief.
        '''
        z_b_mus, z_b_logvars, b_z_bs = [], [], []
        for layer in range(self.num_layers - 1, -1, -1):
            if layer == self.num_layers - 1:
                z_b_mu, z_b_logvar = self.z_posterior_belief[layer](b[:, layer])
            else:
                z_b_mu, z_b_logvar = self.z_posterior_belief[layer](
                    torch.cat([b[:, layer], z_b], dim=1))
            z_b_mus.insert(0, z_b_mu)
            z_b_logvars.insert(0, z_b_logvar)

            z_b = reparameterize_gaussian(z_b_mu, z_b_logvar, self.training)
            b_z_bs.insert(0, z_b)

        z_b_mu = torch.cat(z_b_mus, dim=1)
        z_b_logvar = torch.cat(z_b_logvars, dim=1)
        b_z_b = torch.cat(b_z_bs, dim=1)

        return z_b_mu, z_b_logvar, b_z_b, z_b_mus, z_b_logvars, b_z_bs

    def _beliefs(self, x, t1, t2):
        ''' Runs the LSTMS to obtain beliefs at t1 and t2
        '''
        # aggregate the belief b
        b = self.b_belief_rnn(x)  # size: bs, time, layers, dim
        # replicate b multiple times
        b = b[None, ...].expand(self.samples_per_seq, -1, -1, -1, -1)  # size: copy, bs, time, layers, dim
        # Element-wise indexing. sizes: bs, layers, dim
        b1 = torch.gather(b, 2, t1[..., None, None, None].expand(-1, -1, -1, b.size(3), b.size(4))).view(
            -1, b.size(3), b.size(4))
        b2 = torch.gather(b, 2, t2[..., None, None, None].expand(-1, -1, -1, b.size(3), b.size(4))).view(
            -1, b.size(3), b.size(4))
        return b1, b2

    def rollout_posteriors(self, x, t, n):

        # Run belief network
        b = self.b_belief_rnn(x)[:, t]  # size: bs, time, layers, dim

        # Compute posterior, state of the world from belief.
        _, _, z, _, _, _ = self.sample_posterior_z(b)

        # Rollout for n timesteps predicting the future zs at n.
        rollout_x = []
        for _ in range(n):
            next_z = []
            for layer in range(self.num_layers - 1, -1, -1):
                if layer == self.num_layers - 1:
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1_prediction[layer](z)
                else:
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1_prediction[layer](torch.cat([z, pt_z2_z1], dim=1))
                pt_z2_z1 = reparameterize_gaussian(pt_z2_z1_mu, pt_z2_z1_logvar, True)
                next_z.insert(0, pt_z2_z1)

            z = torch.cat(next_z, dim=1)
            rollout_x.append(self.x_z_decoder(z))

        return torch.stack(rollout_x, dim=1)

    def loss_function(self, forward_ret, labels=None):
        ''' Takes the output from the main forward.
        '''
        (x, t2, qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar, qb_z2_b2_mu, qb_z2_b2_logvar,
         qb_z2_b2, pt_z2_z1_mu, pt_z2_z1_logvar, pd_x2_z2) = forward_ret

        # Copy and expand x.
        x = x[None, ...].expand(self.samples_per_seq, -1, -1, -1)  # size: copy, bs, time, dim
        x2 = torch.gather(x, 2, t2[..., None, None].expand(-1, -1, -1, x.size(3))).view(-1, x.size(3))
        batch_size = x2.size(0)

        # Minimize the loss between smoothed q and current beliefs.
        kl_div_qs_pb = kl_div_gaussian(qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar).mean()

        # Predict Z2 from Z1 as if it has all the future information by known at t2.
        kl_shift_qb_pt = (gaussian_log_prob(qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2) -
                          gaussian_log_prob(pt_z2_z1_mu, pt_z2_z1_logvar, qb_z2_b2)).mean()

        # Ground the t2 state of the world in the data, reconstruction loss.
        bce = F.binary_cross_entropy(pd_x2_z2, x2, reduction='sum') / batch_size
        bce_optimal = F.binary_cross_entropy(x2, x2, reduction='sum').detach() / batch_size
        bce_diff = bce - bce_optimal

        loss = bce_diff + kl_div_qs_pb + kl_shift_qb_pt

        return loss, bce_diff, kl_div_qs_pb, kl_shift_qb_pt, bce_optimal


def reparameterize_gaussian(mu, logvar, sample, return_eps=False):
    std = torch.exp(0.5 * logvar)
    if sample:
        eps = torch.randn_like(std)
    else:
        eps = torch.zeros_like(std)
    ret = eps.mul(std).add_(mu)
    if return_eps:
        return ret, eps
    else:
        return ret


def gaussian_log_prob(mu, logvar, x):
    '''Batched log probability log p(x) computation.'''
    logprob = -0.5 * (torch.log(2.0 * numpy.pi) + logvar + ((x - mu) ** 2 / logvar.exp()))
    return logprob.sum(dim=-1)


def kl_div_gaussian(q_mu, q_logvar, p_mu=None, p_logvar=None):
    '''Batched KL divergence D(q||p) computation.'''
    if p_mu is None or p_logvar is None:
        zero = q_mu.new_zeros(1)
        p_mu = p_mu or zero
        p_logvar = p_logvar or zero
    logvar_diff = q_logvar - p_logvar
    kl_div = -0.5 * (1.0 + logvar_diff - logvar_diff.exp() - ((q_mu - p_mu) ** 2 / p_logvar.exp()))
    return kl_div.sum(dim=-1)
