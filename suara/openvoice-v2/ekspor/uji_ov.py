# Uji OpenVoice V2 tone color converter (MIT) di CPU: sumber TTS Indonesia -> warna suara target.
# python -I uji_ov.py <sumber.wav> <target.wav> <keluar.wav> [tau]
import sys, os, time, wave, json
import numpy as np
import torch
import torch.nn.functional as F

BASIS = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, os.path.join(BASIS, 'asli'))  # kode model OpenVoice (MIT), hanya models/modules/attentions/commons/transforms/utils
from openvoice.models import SynthesizerTrn
from openvoice import utils

torch.set_num_threads(4)
SR = 22050


def baca(p):
    with wave.open(p) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, p
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def tulis(p, y):
    y = np.clip(y, -1, 1)
    with wave.open(p, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def spektrum(y):
    # sama dengan mel_processing.spectrogram_torch(center=False) tanpa librosa
    t = torch.from_numpy(y)[None]
    p = (1024 - 256) // 2
    t = F.pad(t[:, None], (p, p), mode='reflect')[:, 0]
    s = torch.stft(t, 1024, hop_length=256, win_length=1024, window=torch.hann_window(1024), center=False, return_complex=True)
    return torch.sqrt(s.real ** 2 + s.imag ** 2 + 1e-6)


def muat():
    hps = utils.get_hparams_from_file(os.path.join(BASIS, 'model', 'config.json'))
    m = SynthesizerTrn(0, hps.data.filter_length // 2 + 1, n_speakers=hps.data.n_speakers, **hps.model).eval()
    ck = torch.load(os.path.join(BASIS, 'model', 'checkpoint.pth'), map_location='cpu', weights_only=True)
    hilang, lebih = m.load_state_dict(ck['model'], strict=False)
    print('kunci hilang', len(hilang), 'lebih', len(lebih))
    return m


@torch.no_grad()
def se(m, y):
    return m.ref_enc(spektrum(y).transpose(1, 2)).unsqueeze(-1)


if __name__ == '__main__':
    src, tgt, out = sys.argv[1:4]
    tau = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    t0 = time.time()
    m = muat()
    print('muat', round(time.time() - t0, 1), 's; parameter', sum(p.numel() for p in m.parameters()) // 1000, 'ribu')
    ys, yt = baca(src), baca(tgt)
    t1 = time.time()
    se_s, se_t = se(m, ys), se(m, yt)
    t2 = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sp = spektrum(ys)
        o = m.voice_conversion(sp, torch.LongTensor([sp.size(-1)]), sid_src=se_s, sid_tgt=se_t, tau=tau)[0][0, 0].numpy()
    t3 = time.time()
    tulis(out, o)
    se_o = se(m, o.astype(np.float32))
    cos = lambda a, b: float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    print(f'se {t2 - t1:.2f} s, konversi {t3 - t2:.2f} s untuk {len(ys) / SR:.2f} s audio (RTF {(t3 - t2) / (len(ys) / SR):.3f})')
    print(f'kemiripan (encoder OpenVoice): sumber~target {cos(se_s, se_t):.3f} | hasil~target {cos(se_o, se_t):.3f} | hasil~sumber {cos(se_o, se_s):.3f}')
    np.save(out + '.se_t.npy', se_t.numpy()); np.save(out + '.se_s.npy', se_s.numpy())
