import knowhere
import json
import numpy as np
import pandas as pd
import tlsh

def hex_string_to_int_array(hex_string, little_endian=True):
    # 补充字符，使其长度为8的倍数
    while len(hex_string) % 8 != 0:
        hex_string += '0'

    # 每8个字符为一组，转换为int并存储在数组中
    if little_endian:
        int_array = [int(hex_string[i+6:i+8]+hex_string[i+4:i+6]+hex_string[i+2:i+4]+hex_string[i:i+2], 16) for i in range(0, len(hex_string), 8)]
    else:
        int_array = [int(hex_string[i:i+8], 16) for i in range(0, len(hex_string), 8)]

    return np.array(int_array).astype(np.int32)

# 示例：
hex_string = "2c03f70d5e3efb76e070447929592e1c5b8a2d36fa3e3232957031d60f35ce7a81938a"
int_array = hex_string_to_int_array(hex_string)
print(int_array)

config = {
    "dim": 35*8,
    "k": 100,
    "metric_type": "TLSH",
    "M": 16,
    "efConstruction": 100,
    "ef": 100,
}

idx = knowhere.CreateIndex("HNSW")

df = pd.read_csv(r"/go/src/github.com/milvus-io/knowhere/tlsh_demo.csv")
print(len(df))
tlshs = df['tlsh'].to_list()

xq = np.array([hex_string_to_int_array(x) for x in tlshs])
print(xq)

xb = [
    [
        "f103087113036c19d6ab93bfa643c27ce7967ee1e726785192333e7b393906114938d2",
    ]
]
xb = np.array([hex_string_to_int_array(x) for x in xb[0]])

idx.Build(
    knowhere.ArrayToDataSet(xq),
    json.dumps(config),
)
ans, _ = idx.Search(
    knowhere.ArrayToDataSet(xb),
    json.dumps(config),
    knowhere.GetNullBitSetView()
)
k_dis, k_ids = knowhere.DataSetToArray(ans)
print(k_dis, k_ids)
for dis, id in zip(k_dis[0], k_ids[0]):
    print(dis, id, tlshs[id], tlsh.diff("f103087113036c19d6ab93bfa643c27ce7967ee1e726785192333e7b393906114938d2", tlshs[id]))