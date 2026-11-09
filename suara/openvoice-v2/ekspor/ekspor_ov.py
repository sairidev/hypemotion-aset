# Ekspor OpenVoice V2 tone color converter ke dua graf ONNX untuk HP (spektrogram ada di dalam graf):
#   ov_se.onnx : audio (1, N) 22050 Hz            -> se (1, 256, 1)
#   ov_vc.onnx : audio (1, N), se_src, se_tgt, tau -> audio (1, 1, M)
# lalu bandingkan dengan PyTorch (tau = 0 supaya deterministik).
# python -I ekspor_ov.py
import os, sys, time
import numpy as np
import torch
from torch import nn

BASIS = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, BASIS)
from uji_ov import muat, baca, spektrum  # memuat model (kode MIT dari folder asli)

OUT = os.path.join(BASIS, 'onnx')
os.makedirs(OUT, exist_ok=True)


def spek_t(audio):
    p = (1024 - 256) // 2
    y = torch.nn.functional.pad(audio[:, None], (p, p), mode='reflect')[:, 0]
    s = torch.stft(y, 1024, hop_length=256, win_length=1024, window=torch.hann_window(1024), center=False, return_complex=False)
    return torch.sqrt(s.pow(2).sum(-1) + 1e-6)


class SE(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, audio):
        return self.m.ref_enc(spek_t(audio).transpose(1, 2)).unsqueeze(-1)


class VC(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, audio, se_src, se_tgt, tau):
        sp = spek_t(audio)
        mask = torch.ones_like(sp[:, :1, :])
        e = self.m.enc_q
        x = e.pre(sp) * mask
        x = e.enc(x, mask, g=torch.zeros_like(se_src))  # zero_g=True seperti voice_conversion
        stats = e.proj(x) * mask
        mm, logs = torch.split(stats, e.out_channels, dim=1)
        z = (mm + torch.randn_like(mm) * tau * torch.exp(logs)) * mask
        z_p = self.m.flow(z, mask, g=se_src)
        z_hat = self.m.flow(z_p, mask, g=se_tgt, reverse=True)
        return self.m.dec(z_hat * mask, g=torch.zeros_like(se_tgt))


m = muat()
ys = torch.from_numpy(baca(os.path.join(BASIS, 'audio', 'sumber.wav')))[None]
yt = torch.from_numpy(baca(os.path.join(BASIS, 'audio', 'target-denia.wav')))[None]
se_m, vc_m = SE(m).eval(), VC(m).eval()
with torch.no_grad():
    # cek spektrogram sendiri = versi uji_ov (dipakai saat uji kualitas)
    print('spek beda maks', float((spek_t(ys) - spektrum(ys[0].numpy())).abs().max()))
    se_s, se_t = se_m(ys), se_m(yt)
    tau0 = torch.tensor(0.0)
    acuan = vc_m(ys, se_s, se_t, tau0)
    # bandingkan dengan jalur asli model.voice_conversion pada tau=0
    asli = m.voice_conversion(spektrum(ys[0].numpy()), torch.LongTensor([spektrum(ys[0].numpy()).size(-1)]), se_s, se_t, tau=0.0)[0]
    print('VC pembungkus vs voice_conversion asli, beda maks', float((acuan - asli).abs().max()))

kw = dict(opset_version=17, do_constant_folding=True, dynamo=False)
t = time.time()
torch.onnx.export(se_m, (ys,), os.path.join(OUT, 'ov_se.onnx'), input_names=['audio'], output_names=['se'],
                  dynamic_axes={'audio': {1: 'n'}}, **kw)
torch.onnx.export(vc_m, (ys, se_s, se_t, torch.tensor(0.3)), os.path.join(OUT, 'ov_vc.onnx'),
                  input_names=['audio', 'se_src', 'se_tgt', 'tau'], output_names=['wav'],
                  dynamic_axes={'audio': {1: 'n'}, 'wav': {2: 'm'}}, **kw)
print('ekspor', round(time.time() - t, 1), 's')
for f in os.listdir(OUT):
    print(' ', f, round(os.path.getsize(os.path.join(OUT, f)) / 1048576, 1), 'MB')
np.save(os.path.join(OUT, 'acuan_wav.npy'), acuan.numpy())
np.save(os.path.join(OUT, 'acuan_se_s.npy'), se_s.numpy())
np.save(os.path.join(OUT, 'acuan_se_t.npy'), se_t.numpy())
np.save(os.path.join(OUT, 'acuan_audio.npy'), ys.numpy())
