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

# 读取列表
records = pd.read_csv(r"/go/src/github.com/milvus-io/knowhere/cluster_sample_single.csv", index_col=0).to_dict(orient='records')
print(len(records))


# 加载索引
idx_file = f"all_drop_duplicates_ssdeep_50w_mars.csv._m16_ef_128.idx"

config = {
    "dim": 35*8,
    "k": 1,
    "metric_type": "TLSH",
    "M": 16,
    "efConstruction": 128,
    "ef": 100,
}
idx = knowhere.CreateIndex("HNSW")

# df = pd.read_csv(r"/go/src/github.com/milvus-io/knowhere/all_drop_duplicates_ssdeep_50w_mars.csv", usecols=['tlsh'])
# tlsh_hashs = df['tlsh'].to_list()

# df = pd.read_csv(r"/go/src/github.com/milvus-io/knowhere/all_drop_duplicates_ssdeep.csv", usecols=['tlsh'])
# tlsh_hashs = df['tlsh'].to_list()
# print(len(tlsh_hashs))
# tlshs = df['tlsh'].to_list()
# data_set = np.array([hex_string_to_int_array(x) for x in tlshs])
# idx.Build(
#     knowhere.ArrayToDataSet(data_set),
#     json.dumps(config),
# )
# binset = knowhere.GetBinarySet()
# idx.Serialize(binset)

# 加载
# build_status = idx.Build(
#     knowhere.GetNullDataSet(),
#     json.dumps(config),
# )
# print(build_status)
# assert knowhere.Status(build_status) == knowhere.Status.success
# idx.Deserialize(knowhere.GetBinarySet(), json.dumps(config))

for record in records:
    hit_count = 0
    for n in range(0, 10):
        try:
            xb = [[
                    record['tlsh'],
                ]]
            xb = np.array([hex_string_to_int_array(x) for x in xb[0]])

            ans, _ = idx.Search(
                knowhere.ArrayToDataSet(xb),
                json.dumps(config),
                knowhere.GetNullBitSetView()
            )
            k_dis, k_ids = knowhere.DataSetToArray(ans)
            # print(k_dis, k_ids)
            for dis, id in zip(k_dis[0], k_ids[0]):
                # print(dis, id, tlshs[id], tlsh.diff("f103087113036c19d6ab93bfa643c27ce7967ee1e726785192333e7b393906114938d2", tlshs[id]))
                if abs(dis) < 1e-9:
                    # print(tlsh_hashs[idx], record['tlsh'], dist)
                    hit_count += 1
                    break
                # if record['tlsh'] in tlshs:
                #     print(dis, id, tlshs[id], record['tlsh'], tlsh.diff(record['tlsh'], tlshs[id]))
        except Exception as e:
            print('hnsw_search exception:', str(e), record['tlsh'])
            continue
    recall = hit_count/10
    print(f"recall: {record['tlsh']} {record['size']} {recall:.3f}")