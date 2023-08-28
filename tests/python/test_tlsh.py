import knowhere
import json
import numpy as np

config = {
    "dim": 72,
    "k": 100,
    "metric_type": "TLSH",
    "M": 16,
    "efConstruction": 100,
    "ef": 100,
}

idx = knowhere.CreateIndex("HNSW")

# def gen_data(xb_rows, xq_rows, dim):
#     return (
#         np.random.randn(xb_rows, dim).astype(np.float32),
#         np.random.randn(xq_rows, dim).astype(np.float32),
#     )
    
# xb, xq = gen_data(10000, 100, 256)
# print(xb)
# print(xq)

# def hex_str_2_int_ary(hex_string):
#     hex_string = "".join([format(ord(char), "02x") for char in hex_string])
#     print(hex_string)
#     while len(hex_string) % 8 != 0:
#         hex_string += '0'
#     hex_chunks = [hex_string[i:i + 8] for i in range(0, len(hex_string), 8)]
#     # print(hex_chunks)
#     def little_endian(chunk):
#         return ''.join([chunk[i:i + 2] for i in range(0, len(chunk), 2)][::-1])
#     int_array = [int(little_endian(chunk), 16) for chunk in hex_chunks]
#     return np.array(int_array).astype(np.float32)

# xb = [
#     hex_str_2_int_ary("5E24CF523BE1D0F0E06702751AE49B726A7EFD728B61D8D7B7850B6E19302C0DA3A753"),
#     hex_str_2_int_ary("B7B2AF5AE12849CCE27F583D6544EC694208A4E51A02E18F31ACF6862B691C39FCF37F"),
#     hex_str_2_int_ary("8715F172F786F97AD05672744C9EA69A3A26FC509E2442877340BB0EAE337E15D33311"),
#     hex_str_2_int_ary("BFB44AC6A19643BBEE8766FF358AC55DBC13D91C1B4DB4FBC789AA020A31B05ED12350"),
#     hex_str_2_int_ary("7F258D0273918025FFAE92735B55B24156BCE8253123CD3F12BA9F79AB701B11E2D26F"),
#     hex_str_2_int_ary("9F440B597A1CA800D5800AF3EC8751DE75BAEC38FE10EA6319EB78679DF203548195FE"),
#     hex_str_2_int_ary("9894F296F359DE8BDA819A348ED352C61323FC504B66430B319DF31BBA7E1544E8B2C9"),
#     hex_str_2_int_ary("2EB2BE9A91A7E898C72B187E810558E1150FEC225442E65F36F5FB9D46CB2C30FCF18E"),
#     hex_str_2_int_ary("F1E4AF22E2E17433C1273A7D9CDB576C982DFE503D2869462BE51C4F6F3D691382A293"),
#     hex_str_2_int_ary("2404700D61F2ED16FD20F2B01BD9F6E59D58AC9F9B0B207626C777541E36681BC832A0"),
#     hex_str_2_int_ary("39153347E9784F97447A40D2D86719A5B4CA6052AC4C69A3D5CD3FFAB2BF0783232F09"),
#     hex_str_2_int_ary("93B3E00BC651D741C13802F95D870F6ABE1B2B1DB9C19ACF19A50E5FBE7632265CD0AC"),
#     hex_str_2_int_ary("0DA2BF5E51B5B42FC623603DD2009DB1B90E67819042F14F31F6BB9A26A62C713CE29E"),
#     hex_str_2_int_ary("19B2AF5EE128898CE27B583D5545DC69520860E51B02F18F31ACF6862B7A1C39BDF37F"),
#     hex_str_2_int_ary("6825CF2163445FAAF07E5739E439541487F3FA07E36ACA4D7C8861EE0A63B468272F47"),
#     hex_str_2_int_ary("777463018B3254B1DE99347026D7717D419B4A185B372CDEABE8C36B3E3B4A38147BE9"),
#     hex_str_2_int_ary("C9C4E7AD96506B9AF13C21321359C0FEE6A50C7573195AEB83C72EAF2D1A1D1DC30B1B"),
#     hex_str_2_int_ary("075353A6F36DFD45DA8567394E9B86D21323FC604E21434B32D8B71B7EB90508F0E58A"),
#     hex_str_2_int_ary("ADD3913463D4E826F71334358AE5E95AE161AC96692B82FF324C367CBF3C1D549722C2"),
#     hex_str_2_int_ary("F1043BC0209E6FD2CD1E203254FB8EDE93842E1818E67515371C777CBFB1AB2B669A15"),
#     hex_str_2_int_ary("A704199830EABE16E9422D3105AACD78B8A4BF861874C5673558FFEE3F333587463116"),
#     hex_str_2_int_ary("B4F4BF22F6E04433D122197D5C1B97789836FF10392959476BEB5F4CAF3A381386AE93"),
#     hex_str_2_int_ary("5C735A07AD089A21D6B046B11C63C76D2F16BC0C89861E4F759FBE57BF327A16C5E21C"),
#     hex_str_2_int_ary("1F14E64FE180FD26D7528631087EFEB86E765C6B7715CC762112FF2F58BA224C9482A1"),
#     hex_str_2_int_ary("36E4E050BA91DA1FCB6A46750DD6DBFF2774FC229E6187873204B72F2DB1A509B02324"),
#     hex_str_2_int_ary("21346D1271DFEF3BDC6242B06956DB74246CBC55AD88814B30CD77AD2F3A92E268D04D"),
#     hex_str_2_int_ary("47B2BE5991A7E898C72B187E800948E1150FFC619482E61F36F9FB9D56CA2C31FCF18E"),
#     hex_str_2_int_ary("68F4F189F321B961DE26737405878EC1BA94D04BA42253AB9011F357BC17BEE3E7E1E4"),
#     hex_str_2_int_ary("ED84F18D3690B2EFC86BC976DD641D24DA21746B970BD303E08756AF8A0E9A7CF144F1"),
#     hex_str_2_int_ary("CC044BC0209EAFD6CD1E203254FB8DEE53842E1418E6B416371C377CAFB56B2F669A15"),
#     hex_str_2_int_ary("C9E384C42981EC6AF79D003DDA8EAEB96D147C506A8A4D72340D371C5BF3C53A98ADF1"),
#     hex_str_2_int_ary("4394F296F359DE8BDA819A348ED352C61323FC504B66430B319DF31BBA7E1544E8B2C9"),
#     hex_str_2_int_ary("D7347CC2518F2FE6CD1F103250B69BEA42D49D182DE57002325D3BBC7FB8A77B65A846"),
#     hex_str_2_int_ary("7AB2AF5AE12849CCE27F583D6544EC694208A4E51A02E18F31ACF6866B691C39FDF37F"),
#     hex_str_2_int_ary("9BA2BE5E51B5B86FC623603DD2009DF1B80E67419442F14F21F6BBAA26A62D723CE14F"),
#     hex_str_2_int_ary("AE158C1276C2C073C262357649EAA37962ABE5300F7877C7AA960B3D5E346D25D3834F")
# ]

# xb = np.random.randn(1, 35).astype(np.float32)
# xq = np.random.randn(3, 35).astype(np.float32)

# xq = [
#     list(chr(b) for b in bytes.fromhex("39153347E9784F97447A40D2D86719A5B4CA6052AC4C69A3D5CD3FFAB2BF0783232F09")),
#     list(chr(b) for b in bytes.fromhex("AE158C1276C2C073C262357649EAA37962ABE5300F7877C7AA960B3D5E346D25D3834F")),
#     list(chr(b) for b in bytes.fromhex("5E24CF523BE1D0F0E06702751AE49B726A7EFD728B61D8D7B7850B6E19302C0DA3A753")),
# ]

# xq = [[
#     "39153347E9784F97447A40D2D86719A5B4CA6052AC4C69A3D5CD3FFAB2BF0783232F09",
#     "AE158C1276C2C073C262357649EAA37962ABE5300F7877C7AA960B3D5E346D25D3834F",
#     "5E24CF523BE1D0F0E06702751AE49B726A7EFD728B61D8D7B7850B6E19302C0DA3A753",
# ]]

import pandas as pd
df = pd.read_csv(r"/mnt/k/threat_intelligence_code/knowhere/tlsh_demo.csv")
print(len(df))

xq = [df['tlsh'].to_list()]

import binascii
xq = np.array([np.frombuffer(binascii.unhexlify(x), dtype=np.uint8) for x in xq[0]])

# byte_arr = xq.astype(np.string_).tobytes()
#xq = np.array(xq)
# print(xq)

xb = [
    [
        "2c03f70d5e3efb76e070447929592e1c5b8a2d36fa3e3232957031d60f35ce7a81938a",
    ]
]

xb = np.array([np.frombuffer(binascii.unhexlify(x), dtype=np.uint8) for x in xb[0]])

idx.Build(
    knowhere.ArrayToDataSet(xq),
    json.dumps(config),
)
ans, _ = idx.Search(
    knowhere.ArrayToDataSet(np.array(xb)),
    json.dumps(config),
    knowhere.GetNullBitSetView()
)
k_dis, k_ids = knowhere.DataSetToArray(ans)
print(k_dis, k_ids)
