import torch 
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.init as init
from tacotron.utils import melspectrogram2wav
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os
import librosa.display as lds

def save_samples(datas, path, iteration):
    save_path_all = os.path.join(path, 'sample', str(iteration+1))
    os.makedirs(save_path_all, exist_ok=True)
    for i, data in enumerate(datas):
        save_path = os.path.join(save_path_all, str(i))
        os.makedirs(save_path, exist_ok=True)
        write_pic_wav(data[0], os.path.join(save_path, 'a'))
        write_pic_wav(data[1], os.path.join(save_path, 'a_r'))
        write_pic_wav(data[2], os.path.join(save_path, 'ab1'))
        write_pic_wav(data[3], os.path.join(save_path, 'ab2'))
        write_pic_wav(data[4], os.path.join(save_path, 'b'))
        write_pic_wav(data[5], os.path.join(save_path, 'b_r'))
        write_pic_wav(data[6], os.path.join(save_path, 'ba1'))
        write_pic_wav(data[7], os.path.join(save_path, 'ba2'))

def write_pic_wav(data, path):
    lds.specshow(data.T, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
    plt.savefig(path+'.png')
    plt.clf()
    plt.cla()
    plt.close()
    melspectrogram2wav(data, path+'.wav')
    return

class NoiseAdder(object):
    def __init__(self, mean, std, decay_steps):
        self.mean = mean
        self.std = std
        self.decay_steps = decay_steps
        self.n_steps = 0

    def __call__(self, tensor):
        if self.n_steps < self.decay_steps: 
            self.n_steps += 1
            std = self.std * (1 - self.n_steps / self.decay_steps)
            noise = tensor.new(*tensor.size()).normal_(self.mean, std)
            ret = tensor + noise
        else:
            ret = tensor
        return ret

def sample_gumbel(size, eps=1e-20):
    u = torch.rand(size)
    sample = -torch.log(-torch.log(u + eps) + eps)
    return sample

def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + sample_gumbel(logits.size()).type(logits.type())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1., hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        _, max_ind = torch.max(y, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, max_ind, 1.0)
        y = (y_hard - y).detach() + y
    return y

def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)


class EMA(object):
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def _inflate(tensor, times, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)

def _inflate_np(np_array, times, dim):
    repeat_dims = [1] * np_array.ndim
    repeat_dims[dim] = times
    return np_array.repeat(repeat_dims)

def adjust_learning_rate(optimizer, lr):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return lr

def onehot(input_x, encode_dim=None):
    if encode_dim is None:
        encode_dim = torch.max(input_x) + 1
    input_x = input_x.int().unsqueeze(-1)
    return input_x.new_zeros(*input_x.size()[:-1], encode_dim).float().scatter_(-1, input_x, 1)

class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, tag, dictionary, step):
        self.writer.add_scalars(tag, dictionary, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

    def audio_summary(self, tag, value, step, sr):
        writer.add_audio(tag, value, step, sample_rate=sr)

def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)
