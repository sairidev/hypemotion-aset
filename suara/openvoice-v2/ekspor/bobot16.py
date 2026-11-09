# Simpan bobot float32 sebagai float16 + node Cast ke float32 (hitungan tetap fp32; ORT melipat Cast saat memuat).
# Ukuran unduhan jadi kira-kira setengah tanpa mengandalkan kernel fp16 yang belum ada di WASM.
# python -I bobot16.py <masuk.onnx> <keluar.onnx>
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
masukan = {i.name for i in g.input}
baru_init, cast, n = [], [], 0
for t in g.initializer:
    if t.data_type == TensorProto.FLOAT and t.name not in masukan and np.prod(t.dims) >= 512:
        a = numpy_helper.to_array(t)
        if np.abs(a).max() < 65000:  # aman untuk fp16
            nama16 = t.name + '__f16'
            baru_init.append(numpy_helper.from_array(a.astype(np.float16), nama16))
            cast.append(helper.make_node('Cast', [nama16], [t.name], to=TensorProto.FLOAT, name=t.name + '__cast'))
            n += 1
            continue
    baru_init.append(t)
del g.initializer[:]
g.initializer.extend(baru_init)
simpul = list(g.node)
del g.node[:]
g.node.extend(cast + simpul)
onnx.save(m, dst)
print('bobot diubah ke fp16:', n)
