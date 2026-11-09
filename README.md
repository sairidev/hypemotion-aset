# hypemotion-aset

Repo ini menyimpan berkas model untuk HypeMotion, aplikasi edit video di Android. Aplikasi mengunduh berkas ini satu kali saat fiturnya pertama dipakai, lalu menjalankannya langsung di HP. Audio pengguna tidak dikirim ke server.

## Isi

### `suara/openvoice-v2`

Model untuk fitur **Suara saya**, yang mengubah warna suara sebuah ucapan menjadi suara pengguna. Kata dan iramanya tetap.

| Berkas | Fungsi | Ukuran | SHA-256 |
|---|---|---|---|
| `ov_se16.onnx` | Mengenali warna suara dari rekaman | 1,6 MB | `0d4f5a6abb5e6f3941e742a8d2e982b84ab06badc9aa862cfcd06b2565c3e917` |
| `ov_vc16.onnx` | Mengubah ucapan ke warna suara tersebut | 61,3 MB | `ee415c1051ce368677b8fe9f9e00f06dd629a0b10c3f9757ea898b2bee1a88c1` |

Kedua model menerima audio mono 22.050 Hz. Folder `ekspor` berisi skrip untuk membuat ulang berkas ONNX ini dari model aslinya.

## Sumber dan lisensi

Berkas di atas adalah hasil konversi ke format ONNX dari tone color converter OpenVoice V2 buatan MyShell.ai ([kode](https://github.com/myshell-ai/OpenVoice), [model](https://huggingface.co/myshell-ai/OpenVoiceV2)). Lisensinya MIT, Copyright 2024 MyShell.ai. Teks lengkapnya ada di `suara/openvoice-v2/LICENSE-OpenVoice.txt`.

## Pemakaian yang bertanggung jawab

Gunakan hanya suara yang berhak kamu pakai. Di HypeMotion, profil suara dibuat dari rekaman mikrofon atau berkas audio, dan pengguna harus menyatakan dulu bahwa suara itu miliknya atau pemiliknya sudah mengizinkan.
