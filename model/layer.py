# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
import torch.nn.functional as F



class ReadOut(nn.Module):
    def __init__(self, model='avg'):
        super(ReadOut, self).__init__()

    def forward(self, spike):
        if self.step_mode == 's':
            return spike
        else:
            output = spike.reshape(self.time_step, -1, spike.shape[1])
            avg_fr = output.mean(dim=0)
            return avg_fr


class LIFLayer(neuron.LIFNode):

    def __init__(self, **cell_args):
        super(LIFLayer, self).__init__()
        tau = 1.0 / (1.0 - torch.sigmoid(cell_args['decay'])).item()
        super().__init__(tau=tau, decay_input=False, v_threshold=cell_args['thresh'], v_reset=cell_args['v_reset'],
                         detach_reset=cell_args['detach_reset'], step_mode='s')
        self.register_memory('elig', 0.)
        self.register_memory('elig_factor', 1.0)
        self.register_memory('out_spikes_mean', 0.)
        # self.register_memory('curr_time_step', 0)

    @staticmethod
    # @torch.jit.script
    def calcu_sg_and_elig(current_t: int, v: torch.Tensor, elig: torch.Tensor, elig_factor: float, v_threshold: float,
                          sigmoid_alpha: float = 4.0):
        sgax = ((v - v_threshold) * sigmoid_alpha).sigmoid()
        sg = (1. - sgax) * sgax * sigmoid_alpha
        elig = 1. / (current_t + 1) * (current_t * elig + elig_factor * sg)
        return sg, elig

    def calcu_elig_factor(self, elig_factor, lam, sg, spike):
        if self.v_reset is not None:  # hard-reset
            elig_factor = self.calcu_elig_factor_hard_reset(elig_factor, lam, spike, self.v, sg)
        else:  # soft-reset
            if not self.detach_reset:  # soft-reset w/ reset_detach==False
                elig_factor = self.calcu_elig_factor_soft_reset_not_detach_reset(elig_factor, lam, sg)
            else:  # soft-reset w/ reset_detach==True
                elig_factor = self.calcu_elig_factor_soft_reset_detach_reset(elig_factor, lam)
        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_hard_reset(elig_factor: torch.Tensor, lam: float, spike: torch.Tensor, v: torch.Tensor,
                                     sg: torch.Tensor):
        elig_factor = 1. + elig_factor * (lam * (1. - spike) - lam * v * sg)
        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_soft_reset_not_detach_reset(elig_factor: torch.Tensor, lam: float, sg: torch.Tensor):
        elig_factor = 1. + elig_factor * (lam - lam * sg)
        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_soft_reset_detach_reset(elig_factor: float, lam: float):
        elig_factor = 1. + elig_factor * lam
        return elig_factor

    def elig_init(self, x: torch.Tensor):
        self.elig = torch.zeros_like(x.data)
        self.elig_factor = 1.0

    def reset_state(self):
        self.reset()
        self.curr_time_step = 0

    def forward(self, x, **kwargs):
        if self.step_mode == 's':
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            return spike
        else:
            assert len(x.shape) in (2, 4)
            x = x.view(self.time_step, -1, *x.shape[1:])

            self.reset()
            # self.v = torch.zeros_like(x[0])
            spikes = []
            for t in range(self.time_step):
                self.v_float_to_tensor(x[t])
                self.neuronal_charge(x[t])
                spike = self.neuronal_fire()
                spikes.append(spike)
                self.neuronal_reset(spike)

            # out = torch.stack(spikes, dim=0) if not self.train_mode_multi else torch.cat(spikes, dim=0)
            out = torch.cat(spikes, dim=0)

            return out


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2

    return loss_kd


def cal_loss(outputs, labels, criterion):
    T = outputs.size(0)
    Loss_es = 0
    Loss_mmd = 0
    for t in range(T):
        Loss_es += criterion(outputs[t, :, ...], labels)
    Loss_es = Loss_es / T
    return Loss_es


def make_teacher(avg_fr, labels):
    predictions = avg_fr.argmax(dim=2)

    correct_mask = (predictions == labels.unsqueeze(0))

    correct_avg_fr = avg_fr * correct_mask.unsqueeze(2)


    correct_count = correct_mask.sum(dim=0).unsqueeze(1)

    epsilon = 1e-8
    correct_count = correct_count + epsilon

    teacher_labels = correct_avg_fr.sum(dim=0) / correct_count
    return teacher_labels
