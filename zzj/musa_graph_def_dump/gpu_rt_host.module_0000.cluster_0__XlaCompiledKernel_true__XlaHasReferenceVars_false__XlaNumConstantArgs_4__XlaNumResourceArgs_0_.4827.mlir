module @cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_4__XlaNumResourceArgs_0_.4827 attributes {gpu.container_module, hlo.unique_id = 0 : i32, mhlo.unique_id = 0 : i64, rt.requires_blas = true} {
  func.func @cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_4__XlaNumResourceArgs_0_.4827(%arg0: memref<262144xi8> {lmhlo.params = 0 : index}, %arg1: memref<8192xi8> {lmhlo.params = 1 : index}, %arg2: memref<1048576xi8> {lmhlo.params = 2 : index}, %arg3: memref<1048576xi8> {lmhlo.params = 3 : index}, %arg4: memref<524288xi8> {lmhlo.params = 4 : index}, %arg5: memref<524288xi8> {lmhlo.params = 5 : index}, %arg6: memref<1056768xi8> {lmhlo.params = 6 : index}, %arg7: memref<532480xi8> {lmhlo.params = 7 : index}, %arg8: memref<9437184xi8> {lmhlo.params = 8 : index}, %arg9: memref<1835008xi8> {lmhlo.params = 9 : index}, %arg10: memref<524288xi8> {lmhlo.params = 10 : index}, %arg11: memref<262144xi8> {lmhlo.params = 11 : index}, %arg12: memref<131072xi8> {lmhlo.params = 12 : index}, %arg13: memref<4718592xi8> {lmhlo.params = 13 : index}, %arg14: memref<4718592xi8> {lmhlo.params = 14 : index}, %arg15: memref<69730304xi8> {lmhlo.params = 15 : index}, %arg16: memref<20971520xi8> {lmhlo.params = 16 : index}, %arg17: memref<10176896xi8> {lmhlo.constant_name = "buffer_for_constant_581"}, %arg18: memref<10176896xi8> {lmhlo.constant_name = "buffer_for_constant_681"}, %arg19: memref<10176896xi8> {lmhlo.constant_name = "buffer_for_constant_715"}, %arg20: memref<10176896xi8> {lmhlo.constant_name = "buffer_for_constant_785"}, %arg21: memref<1572864xi8> {lmhlo.constant_name = "buffer_for_constant_359"}, %arg22: memref<786432xi8> {lmhlo.constant_name = "buffer_for_constant_435"}, %arg23: memref<786432xi8> {lmhlo.constant_name = "buffer_for_constant_1122"}, %arg24: memref<524288xi8> {lmhlo.constant_name = "buffer_for_constant_418"}, %arg25: memref<294912xi8> {lmhlo.constant_name = "buffer_for_constant_125"}, %arg26: memref<262144xi8> {lmhlo.constant_name = "buffer_for_constant_475"}, %arg27: memref<262144xi8> {lmhlo.constant_name = "buffer_for_constant_999"}, %arg28: memref<262144xi8> {lmhlo.constant_name = "buffer_for_constant_1058"}, %arg29: memref<262144xi8> {lmhlo.constant_name = "buffer_for_constant_1181"}, %arg30: memref<229376xi8> {lmhlo.constant_name = "buffer_for_constant_1570"}, %arg31: memref<196608xi8> {lmhlo.constant_name = "buffer_for_constant_253"}, %arg32: memref<196608xi8> {lmhlo.constant_name = "buffer_for_constant_256"}, %arg33: memref<196608xi8> {lmhlo.constant_name = "buffer_for_constant_931"}, %arg34: memref<196608xi8> {lmhlo.constant_name = "buffer_for_constant_1408"}, %arg35: memref<196608xi8> {lmhlo.constant_name = "buffer_for_constant_1284"}, %arg36: memref<131072xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}, %arg37: memref<131072xi8> {lmhlo.constant_name = "buffer_for_constant_244"}, %arg38: memref<131072xi8> {lmhlo.constant_name = "buffer_for_constant_64"}, %arg39: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4522"}, %arg40: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4629"}, %arg41: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4368"}, %arg42: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4475"}, %arg43: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4214"}, %arg44: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4321"}, %arg45: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4060"}, %arg46: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4167"}, %arg47: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_3906"}, %arg48: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_4013"}, %arg49: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_3752"}, %arg50: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_3859"}, %arg51: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_3598"}, %arg52: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_3705"}, %arg53: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_3444"}, %arg54: memref<102400xi8> {lmhlo.constant_name = "buffer_for_constant_3551"}, %arg55: memref<98304xi8> {lmhlo.constant_name = "buffer_for_constant_874"}, %arg56: memref<98304xi8> {lmhlo.constant_name = "buffer_for_constant_877"}, %arg57: memref<98304xi8> {lmhlo.constant_name = "buffer_for_constant_1344"}, %arg58: memref<65536xi8> {lmhlo.constant_name = "buffer_for_constant_1475"}, %arg59: memref<65536xi8> {lmhlo.constant_name = "buffer_for_constant_1214"}, %arg60: memref<65536xi8> {lmhlo.constant_name = "buffer_for_constant_1225"}, %arg61: memref<65536xi8> {lmhlo.constant_name = "buffer_for_constant_1465"}, %arg62: memref<59168xi8> {lmhlo.constant_name = "buffer_for_constant_585"}, %arg63: memref<59168xi8> {lmhlo.constant_name = "buffer_for_constant_719"}, %arg64: memref<44032xi8> {lmhlo.constant_name = "buffer_for_constant_547"}, %arg65: memref<44032xi8> {lmhlo.constant_name = "buffer_for_constant_801"}, %arg66: memref<40960xi8> {lmhlo.constant_name = "buffer_for_constant_4694"}, %arg67: memref<32768xi8> {lmhlo.constant_name = "buffer_for_constant_1534"}, %arg68: memref<32768xi8> {lmhlo.constant_name = "buffer_for_constant_153"}, %arg69: memref<32768xi8> {lmhlo.constant_name = "buffer_for_constant_937"}, %arg70: memref<32768xi8> {lmhlo.constant_name = "buffer_for_constant_940"}, %arg71: memref<29584xi8> {lmhlo.constant_name = "buffer_for_constant_685"}, %arg72: memref<29584xi8> {lmhlo.constant_name = "buffer_for_constant_789"}, %arg73: memref<19456xi8> {lmhlo.constant_name = "buffer_for_constant_1725"}, %arg74: memref<16384xi8> {lmhlo.constant_name = "buffer_for_constant_3344"}, %arg75: memref<16384xi8> {lmhlo.constant_name = "buffer_for_constant_857"}, %arg76: memref<12288xi8> {lmhlo.constant_name = "buffer_for_constant_3159"}, %arg77: memref<12288xi8> {lmhlo.constant_name = "buffer_for_constant_1874"}, %arg78: memref<12288xi8> {lmhlo.constant_name = "buffer_for_constant_2388"}, %arg79: memref<12288xi8> {lmhlo.constant_name = "buffer_for_constant_2131"}, %arg80: memref<12288xi8> {lmhlo.constant_name = "buffer_for_constant_2645"}, %arg81: memref<12288xi8> {lmhlo.constant_name = "buffer_for_constant_2902"}, %arg82: memref<6912xi8> {lmhlo.constant_name = "buffer_for_constant_4769"}, %arg83: memref<5184xi8> {lmhlo.constant_name = "buffer_for_constant_3057"}, %arg84: memref<3072xi8> {lmhlo.constant_name = "buffer_for_constant_304"}, %arg85: memref<3072xi8> {lmhlo.constant_name = "buffer_for_constant_120"}, %arg86: memref<3072xi8> {lmhlo.constant_name = "buffer_for_constant_118"}, %arg87: memref<3072xi8> {lmhlo.constant_name = "buffer_for_constant_110"}, %arg88: memref<3072xi8> {lmhlo.constant_name = "buffer_for_constant_109"}, %arg89: memref<3072xi8> {lmhlo.constant_name = "buffer_for_constant_105"}, %arg90: memref<3072xi8> {lmhlo.constant_name = "buffer_for_constant_104"}, %arg91: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_362"}, %arg92: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_107"}, %arg93: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_106"}, %arg94: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_1002"}, %arg95: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_85"}, %arg96: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_84"}, %arg97: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_1125"}, %arg98: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_50"}, %arg99: memref<2048xi8> {lmhlo.constant_name = "buffer_for_constant_49"}, %arg100: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_94"}, %arg101: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_93"}, %arg102: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_1193"}, %arg103: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_1204"}, %arg104: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_1069"}, %arg105: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_76"}, %arg106: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_75"}, %arg107: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_52"}, %arg108: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_51"}, %arg109: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_71"}, %arg110: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_70"}, %arg111: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_48"}, %arg112: memref<1536xi8> {lmhlo.constant_name = "buffer_for_constant_47"}, %arg113: memref<1024xi8> {lmhlo.constant_name = "buffer_for_constant_247"}, %arg114: memref<1024xi8> {lmhlo.constant_name = "buffer_for_constant_438"}, %arg115: memref<1024xi8> {lmhlo.constant_name = "buffer_for_constant_421"}, %arg116: memref<1024xi8> {lmhlo.constant_name = "buffer_for_constant_478"}, %arg117: memref<1024xi8> {lmhlo.constant_name = "buffer_for_constant_82"}, %arg118: memref<1024xi8> {lmhlo.constant_name = "buffer_for_constant_81"}, %arg119: memref<1024xi8> {lmhlo.constant_name = "buffer_for_constant_1666"}, %arg120: memref<1024xi8> {lmhlo.constant_name = "buffer_for_constant_1656"}, %arg121: memref<688xi8> {lmhlo.constant_name = "buffer_for_constant_551"}, %arg122: memref<688xi8> {lmhlo.constant_name = "buffer_for_constant_574"}, %arg123: memref<688xi8> {lmhlo.constant_name = "buffer_for_constant_577"}, %arg124: memref<688xi8> {lmhlo.constant_name = "buffer_for_constant_708"}, %arg125: memref<688xi8> {lmhlo.constant_name = "buffer_for_constant_711"}, %arg126: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3398"}, %arg127: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3401"}, %arg128: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4526"}, %arg129: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4633"}, %arg130: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4372"}, %arg131: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4479"}, %arg132: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4218"}, %arg133: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4325"}, %arg134: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4064"}, %arg135: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4171"}, %arg136: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3910"}, %arg137: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4017"}, %arg138: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3756"}, %arg139: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3863"}, %arg140: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3602"}, %arg141: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3709"}, %arg142: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3448"}, %arg143: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_3555"}, %arg144: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4679"}, %arg145: memref<640xi8> {lmhlo.constant_name = "buffer_for_constant_4682"}, %arg146: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_60"}, %arg147: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_59"}, %arg148: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_112"}, %arg149: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_111"}, %arg150: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_1478"}, %arg151: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_100"}, %arg152: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_99"}, %arg153: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_190"}, %arg154: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_156"}, %arg155: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_183"}, %arg156: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_1197"}, %arg157: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_1218"}, %arg158: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_1229"}, %arg159: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_934"}, %arg160: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_946"}, %arg161: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_87"}, %arg162: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_86"}, %arg163: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_1061"}, %arg164: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_1411"}, %arg165: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_1184"}, %arg166: memref<512xi8> {lmhlo.constant_name = "buffer_for_constant_1287"}, %arg167: memref<384xi8> {lmhlo.constant_name = "buffer_for_constant_441"}, %arg168: memref<384xi8> {lmhlo.constant_name = "buffer_for_constant_454"}, %arg169: memref<304xi8> {lmhlo.constant_name = "buffer_for_constant_41"}, %arg170: memref<304xi8> {lmhlo.constant_name = "buffer_for_constant_39"}, %arg171: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1574"}, %arg172: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3347"}, %arg173: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_4697"}, %arg174: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_148"}, %arg175: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_157"}, %arg176: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_164"}, %arg177: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_171"}, %arg178: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_179"}, %arg179: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2881"}, %arg180: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2878"}, %arg181: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2865"}, %arg182: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2862"}, %arg183: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2849"}, %arg184: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2846"}, %arg185: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2833"}, %arg186: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2830"}, %arg187: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2817"}, %arg188: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2814"}, %arg189: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2809"}, %arg190: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2806"}, %arg191: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_804"}, %arg192: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_128"}, %arg193: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_127"}, %arg194: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_860"}, %arg195: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_544"}, %arg196: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1537"}, %arg197: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_58"}, %arg198: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_54"}, %arg199: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_53"}, %arg200: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1468"}, %arg201: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1347"}, %arg202: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1728"}, %arg203: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3138"}, %arg204: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3135"}, %arg205: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3122"}, %arg206: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3119"}, %arg207: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3106"}, %arg208: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3103"}, %arg209: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3090"}, %arg210: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3087"}, %arg211: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3074"}, %arg212: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3071"}, %arg213: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3066"}, %arg214: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_3063"}, %arg215: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1853"}, %arg216: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1850"}, %arg217: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1837"}, %arg218: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1834"}, %arg219: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1821"}, %arg220: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1818"}, %arg221: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1805"}, %arg222: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1802"}, %arg223: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1789"}, %arg224: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1786"}, %arg225: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1781"}, %arg226: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_1778"}, %arg227: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2624"}, %arg228: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2621"}, %arg229: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2608"}, %arg230: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2605"}, %arg231: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2592"}, %arg232: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2589"}, %arg233: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2576"}, %arg234: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2573"}, %arg235: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2560"}, %arg236: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2557"}, %arg237: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2552"}, %arg238: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2549"}, %arg239: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2367"}, %arg240: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2364"}, %arg241: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2351"}, %arg242: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2348"}, %arg243: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2335"}, %arg244: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2332"}, %arg245: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2319"}, %arg246: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2316"}, %arg247: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2303"}, %arg248: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2300"}, %arg249: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2295"}, %arg250: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2292"}, %arg251: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2110"}, %arg252: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2107"}, %arg253: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2094"}, %arg254: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2091"}, %arg255: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2078"}, %arg256: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2075"}, %arg257: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2062"}, %arg258: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2059"}, %arg259: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2046"}, %arg260: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2043"}, %arg261: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2038"}, %arg262: memref<256xi8> {lmhlo.constant_name = "buffer_for_constant_2035"}, %arg263: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_177"}, %arg264: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3307"}, %arg265: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3282"}, %arg266: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3279"}, %arg267: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3266"}, %arg268: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3263"}, %arg269: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3250"}, %arg270: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3247"}, %arg271: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3234"}, %arg272: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3231"}, %arg273: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3218"}, %arg274: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3215"}, %arg275: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3210"}, %arg276: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3162"}, %arg277: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1877"}, %arg278: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3207"}, %arg279: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_169"}, %arg280: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2022"}, %arg281: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1997"}, %arg282: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1994"}, %arg283: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1981"}, %arg284: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1978"}, %arg285: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1965"}, %arg286: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1962"}, %arg287: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1949"}, %arg288: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1946"}, %arg289: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1933"}, %arg290: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1930"}, %arg291: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1925"}, %arg292: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_1922"}, %arg293: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_154"}, %arg294: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2536"}, %arg295: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2511"}, %arg296: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2508"}, %arg297: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2495"}, %arg298: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2492"}, %arg299: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2479"}, %arg300: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2476"}, %arg301: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2463"}, %arg302: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2460"}, %arg303: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2447"}, %arg304: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2444"}, %arg305: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2439"}, %arg306: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2436"}, %arg307: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2391"}, %arg308: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2134"}, %arg309: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_146"}, %arg310: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2279"}, %arg311: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2254"}, %arg312: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2251"}, %arg313: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2238"}, %arg314: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2235"}, %arg315: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2222"}, %arg316: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2219"}, %arg317: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2206"}, %arg318: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2203"}, %arg319: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2190"}, %arg320: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2187"}, %arg321: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2182"}, %arg322: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2179"}, %arg323: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_162"}, %arg324: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2768"}, %arg325: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2765"}, %arg326: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2752"}, %arg327: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2749"}, %arg328: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2736"}, %arg329: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2733"}, %arg330: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2720"}, %arg331: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2717"}, %arg332: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2704"}, %arg333: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2701"}, %arg334: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2696"}, %arg335: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2648"}, %arg336: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2905"}, %arg337: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2693"}, %arg338: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2793"}, %arg339: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3050"}, %arg340: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3025"}, %arg341: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3022"}, %arg342: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3009"}, %arg343: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_3006"}, %arg344: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2993"}, %arg345: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2990"}, %arg346: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2977"}, %arg347: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2974"}, %arg348: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2961"}, %arg349: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2958"}, %arg350: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2953"}, %arg351: memref<192xi8> {lmhlo.constant_name = "buffer_for_constant_2950"}, %arg352: memref<128xi8> {lmhlo.constant_name = "buffer_for_constant_182"}, %arg353: memref<128xi8> {lmhlo.constant_name = "buffer_for_constant_535"}, %arg354: memref<128xi8> {lmhlo.constant_name = "buffer_for_constant_445"}, %arg355: memref<108xi8> {lmhlo.constant_name = "buffer_for_constant_4772"}, %arg356: memref<108xi8> {lmhlo.constant_name = "buffer_for_constant_3060"}, %arg357: memref<4xi8> {lmhlo.constant_name = "buffer_for_constant_46"}, %arg358: memref<12xi8> {lmhlo.constant_name = "buffer_for_constant_1208"}, %arg359: memref<16xi8> {lmhlo.constant_name = "buffer_for_constant_1659"}, %arg360: memref<16xi8> {lmhlo.constant_name = "buffer_for_constant_1670"}, %arg361: memref<122096960xi8>) attributes {num_partitions = 1 : i64, replica_count = 1 : i64, result_xla_shape = "f32[1024,32]{1,0}"} {
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c768_i32 = arith.constant 768 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1536_i32 = arith.constant 1536 : i32
    %c3_i32 = arith.constant 3 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c5504_i32 = arith.constant 5504 : i32
    %c14792_i32 = arith.constant 14792 : i32
    %c44032_i32 = arith.constant 44032 : i32
    %c29584_i32 = arith.constant 29584 : i32
    %c1376_i32 = arith.constant 1376 : i32
    %c2304_i32 = arith.constant 2304 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %c608_i32 = arith.constant 608 : i32
    %c384_i32 = arith.constant 384 : i32
    %c2560_i32 = arith.constant 2560 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c10240_i32 = arith.constant 10240 : i32
    %c1280_i32 = arith.constant 1280 : i32
    %c256_i32 = arith.constant 256 : i32
    %c11615232 = arith.constant 11615232 : index
    %c11611136 = arith.constant 11611136 : index
    %c11607040 = arith.constant 11607040 : index
    %c11602944 = arith.constant 11602944 : index
    %c12581888 = arith.constant 12581888 : index
    %c12577792 = arith.constant 12577792 : index
    %c12573696 = arith.constant 12573696 : index
    %c12569600 = arith.constant 12569600 : index
    %c12585984 = arith.constant 12585984 : index
    %c12631040 = arith.constant 12631040 : index
    %c12598272 = arith.constant 12598272 : index
    %c8522752 = arith.constant 8522752 : index
    %c9833472 = arith.constant 9833472 : index
    %c9178112 = arith.constant 9178112 : index
    %c7867392 = arith.constant 7867392 : index
    %c6556672 = arith.constant 6556672 : index
    %c5901312 = arith.constant 5901312 : index
    %c7212032 = arith.constant 7212032 : index
    %c12565504 = arith.constant 12565504 : index
    %c10226688 = arith.constant 10226688 : index
    %c10488832 = arith.constant 10488832 : index
    %c12667904 = arith.constant 12667904 : index
    %c12663808 = arith.constant 12663808 : index
    %c12258304 = arith.constant 12258304 : index
    %c12061696 = arith.constant 12061696 : index
    %c12454912 = arith.constant 12454912 : index
    %c723968 = arith.constant 723968 : index
    %c11865088 = arith.constant 11865088 : index
    %c11668480 = arith.constant 11668480 : index
    %c920576 = arith.constant 920576 : index
    %c11406336 = arith.constant 11406336 : index
    %c11144192 = arith.constant 11144192 : index
    %c1051648 = arith.constant 1051648 : index
    %c789504 = arith.constant 789504 : index
    %c1248256 = arith.constant 1248256 : index
    %c314368 = arith.constant 314368 : index
    %c4721664 = arith.constant 4721664 : index
    %c5278720 = arith.constant 5278720 : index
    %c527360 = arith.constant 527360 : index
    %c5016576 = arith.constant 5016576 : index
    %c91061248 = arith.constant 91061248 : index
    %c90885120 = arith.constant 90885120 : index
    %c60591104 = arith.constant 60591104 : index
    %c30297088 = arith.constant 30297088 : index
    %c121179136 = arith.constant 121179136 : index
    %c4201472 = arith.constant 4201472 : index
    %c4197376 = arith.constant 4197376 : index
    %c36703232 = arith.constant 36703232 : index
    %c40897536 = arith.constant 40897536 : index
    %c41946112 = arith.constant 41946112 : index
    %c39848960 = arith.constant 39848960 : index
    %c9440256 = arith.constant 9440256 : index
    %c7347200 = arith.constant 7347200 : index
    %c7343104 = arith.constant 7343104 : index
    %c33557504 = arith.constant 33557504 : index
    %c6294528 = arith.constant 6294528 : index
    %c121834496 = arith.constant 121834496 : index
    %c69737472 = arith.constant 69737472 : index
    %c121572352 = arith.constant 121572352 : index
    %c70786048 = arith.constant 70786048 : index
    %c69209088 = arith.constant 69209088 : index
    %c69733376 = arith.constant 69733376 : index
    %c5245952 = arith.constant 5245952 : index
    %c71044096 = arith.constant 71044096 : index
    %c2100224 = arith.constant 2100224 : index
    %c3673088 = arith.constant 3673088 : index
    %c3148800 = arith.constant 3148800 : index
    %c1575936 = arith.constant 1575936 : index
    %c68684800 = arith.constant 68684800 : index
    %c67111936 = arith.constant 67111936 : index
    %c71326720 = arith.constant 71326720 : index
    %c71830528 = arith.constant 71830528 : index
    %c72879104 = arith.constant 72879104 : index
    %c70781952 = arith.constant 70781952 : index
    %c71322624 = arith.constant 71322624 : index
    %c71318528 = arith.constant 71318528 : index
    %c73403392 = arith.constant 73403392 : index
    %c71310336 = arith.constant 71310336 : index
    %c71314432 = arith.constant 71314432 : index
    %c70257664 = arith.constant 70257664 : index
    %c3072 = arith.constant 3072 : index
    %c71306240 = arith.constant 71306240 : index
    %c0_i32 = arith.constant 0 : i32
    %c265216 = arith.constant 265216 : index
    %c0 = arith.constant 0 : index
    %view = memref.view %arg357[%c0][] : memref<4xi8> to memref<f32>
    %view_0 = memref.view %arg0[%c0][] : memref<262144xi8> to memref<1024x64xf32>
    %view_1 = memref.view %arg36[%c0][] : memref<131072xi8> to memref<1024xf32>
    %view_2 = memref.view %arg361[%c265216][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_81(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_0, %view, %view_1, %view_2) {kernel = "fusion_157", rt.trace = #rt.hlo_trace<"fusion.157">, stream = 0 : i64, uid = 106 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x64xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_3 = memref.view %arg199[%c0][] : memref<256xi8> to memref<64xf32>
    %view_4 = memref.view %arg198[%c0][] : memref<256xi8> to memref<64xf32>
    %view_5 = memref.view %arg197[%c0][] : memref<256xi8> to memref<64xf32>
    %view_6 = memref.view %arg361[%c71306240][] : memref<122096960xi8> to memref<1024xf32>
    %view_7 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.func.launch_80(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_5, %view_1, %view_0, %view_2, %view_4, %view_3, %view_6, %view_7) {kernel = "fusion_155", rt.trace = #rt.hlo_trace<"fusion.155">, stream = 0 : i64, uid = 105 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<64xf32>, memref<1024xf32>, memref<1024x64xf32>, memref<1024xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024x64xf32>) -> ()
    %view_8 = memref.view %arg68[%c0][] : memref<32768xi8> to memref<64x128xf32>
    %view_9 = memref.view %arg361[%c70257664][] : memref<122096960xi8> to memref<1024x128xf32>
    call @xla.gpu.gemm_30(%view_7, %view_8, %view_9) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.3">, uid = 67 : i64} : (memref<1024x64xf32>, memref<64x128xf32>, memref<1024x128xf32>) -> ()
    %view_10 = memref.view %arg7[%c0][] : memref<532480xi8> to memref<1024x130xf32>
    %view_11 = memref.view %arg361[%c71314432][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_79(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_10, %view_11) {kernel = "fusion_110", rt.trace = #rt.hlo_trace<"fusion.110">, stream = 0 : i64, uid = 104 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x130xf32>, memref<1024xf32>) -> ()
    %view_12 = memref.view %arg5[%c0][] : memref<524288xi8> to memref<1024x128xf32>
    %view_13 = memref.view %arg361[%c71310336][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_78(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_12, %view_13) {kernel = "fusion_159", rt.trace = #rt.hlo_trace<"fusion.159">, stream = 0 : i64, uid = 103 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x128xf32>, memref<1024xf32>) -> ()
    %view_14 = memref.view %arg154[%c0][] : memref<512xi8> to memref<128xf32>
    %view_15 = memref.view %arg361[%c73403392][] : memref<122096960xi8> to memref<1024xf32>
    %view_16 = memref.view %arg361[%c71318528][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_77(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_9, %view_14, %view_6, %view_12, %view_13, %view_10, %view_11, %view_15, %view_16) {kernel = "fusion_198", rt.trace = #rt.hlo_trace<"fusion.198">, stream = 0 : i64, uid = 102 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024xf32>, memref<1024x130xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_17 = memref.view %arg10[%c0][] : memref<524288xi8> to memref<1024x128xf32>
    %view_18 = memref.view %arg361[%c71322624][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_76(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_17, %view, %view_1, %view_18) {kernel = "fusion_148", rt.trace = #rt.hlo_trace<"fusion.148">, stream = 0 : i64, uid = 101 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x128xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_19 = memref.view %arg149[%c0][] : memref<512xi8> to memref<128xf32>
    %view_20 = memref.view %arg148[%c0][] : memref<512xi8> to memref<128xf32>
    %view_21 = memref.view %arg147[%c0][] : memref<512xi8> to memref<128xf32>
    %view_22 = memref.view %arg146[%c0][] : memref<512xi8> to memref<128xf32>
    %view_23 = memref.view %arg361[%c70781952][] : memref<122096960xi8> to memref<1024x128xf32>
    %view_24 = memref.view %arg361[%c72879104][] : memref<122096960xi8> to memref<1024x128xf32>
    call @xla.gpu.func.launch_75(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_22, %view_21, %view_1, %view_17, %view_18, %view_20, %view_19, %view_23, %view_24) {kernel = "fusion_146", rt.trace = #rt.hlo_trace<"fusion.146">, stream = 0 : i64, uid = 100 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024xf32>, memref<128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<1024x128xf32>) -> ()
    %view_25 = memref.view %arg38[%c0][] : memref<131072xi8> to memref<128x256xf32>
    %view_26 = memref.view %arg361[%c71830528][] : memref<122096960xi8> to memref<1024x256xf32>
    call @xla.gpu.gemm_24(%view_23, %view_25, %view_26) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.4">, uid = 66 : i64} : (memref<1024x128xf32>, memref<128x256xf32>, memref<1024x256xf32>) -> ()
    %view_27 = memref.view %arg153[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_74(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_27, %view_26, %view_1) {kernel = "fusion_202", rt.trace = #rt.hlo_trace<"fusion.202">, stream = 0 : i64, uid = 99 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<128xf32>, memref<1024x256xf32>, memref<1024xf32>) -> ()
    %view_28 = memref.view %arg6[%c0][] : memref<1056768xi8> to memref<1024x258xf32>
    call @xla.gpu.func.launch_73(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_28, %view_18) {kernel = "fusion_181", rt.trace = #rt.hlo_trace<"fusion.181">, stream = 0 : i64, uid = 98 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x258xf32>, memref<1024xf32>) -> ()
    %view_29 = memref.view %arg361[%c71326720][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_72(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_18, %view_16, %view_27, %view_26, %view_9, %view_14, %view_6, %view_10, %view_11, %view_28, %view_29) {kernel = "fusion_106", rt.trace = #rt.hlo_trace<"fusion.106">, stream = 0 : i64, uid = 97 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<128xf32>, memref<1024x256xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x130xf32>, memref<1024xf32>, memref<1024x258xf32>, memref<1024xf32>) -> ()
    %view_30 = memref.view %arg155[%c0][] : memref<512xi8> to memref<128xf32>
    %view_31 = memref.view %arg101[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_32 = memref.view %arg100[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_33 = memref.view %arg4[%c0][] : memref<524288xi8> to memref<1024x128xf32>
    %view_34 = memref.view %arg361[%c67111936][] : memref<122096960xi8> to memref<1024x384xf32>
    %view_35 = memref.view %arg361[%c68684800][] : memref<122096960xi8> to memref<3x1024x128xf32>
    call @xla.gpu.func.launch_71(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_29, %view_32, %view_31, %view_1, %view_18, %view_16, %view_27, %view_26, %view_9, %view_14, %view_6, %view_10, %view_11, %view_28, %view_30, %view_12, %view_13, %view_33, %view_34, %view_35) {kernel = "fusion_105", rt.trace = #rt.hlo_trace<"fusion.105">, stream = 0 : i64, uid = 96 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<128xf32>, memref<1024x256xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x130xf32>, memref<1024xf32>, memref<1024x258xf32>, memref<128xf32>, memref<1024x128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024x384xf32>, memref<3x1024x128xf32>) -> ()
    %view_36 = memref.view %arg102[%c0][] : memref<1536xi8> to memref<3x128xf32>
    %view_37 = memref.view %arg361[%c68684800][] : memref<122096960xi8> to memref<3x131072xf32>
    %reinterpret_cast = memref.reinterpret_cast %view_37 to offset: [0], sizes: [131072, 3], strides: [1, 131072] : memref<3x131072xf32> to memref<131072x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 131072)>>
    %view_38 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<131072x128xf32>
    call @xla.gpu.gemm_35(%reinterpret_cast, %view_36, %view_38) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.9">, uid = 65 : i64} : (memref<131072x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 131072)>>, memref<3x128xf32>, memref<131072x128xf32>) -> ()
    %view_39 = memref.view %arg156[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_70(%c0_i32, %c32768_i32, %c1_i32, %c1_i32, %c32_i32, %c4_i32, %c1_i32, %view_38, %view_39) {kernel = "fusion_134", rt.trace = #rt.hlo_trace<"fusion.134">, stream = 0 : i64, uid = 95 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<131072x128xf32>, memref<128xf32>) -> ()
    %view_40 = memref.view %arg103[%c0][] : memref<1536xi8> to memref<128x3xf32>
    %view_41 = memref.view %arg361[%c70257664][] : memref<122096960xi8> to memref<3x131072xf32>
    %reinterpret_cast_42 = memref.reinterpret_cast %view_41 to offset: [0], sizes: [131072, 3], strides: [1, 131072] : memref<3x131072xf32> to memref<131072x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 131072)>>
    call @xla.gpu.gemm_34(%view_38, %view_40, %reinterpret_cast_42) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.10">, uid = 64 : i64} : (memref<131072x128xf32>, memref<128x3xf32>, memref<131072x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 131072)>>) -> ()
    %view_43 = memref.view %arg358[%c0][] : memref<12xi8> to memref<3xf32>
    %view_44 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x3x128xf32>
    call @xla.gpu.func.launch_69(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_35, %view_43, %view_41, %view_44) {kernel = "fusion_207", rt.trace = #rt.hlo_trace<"fusion.207">, stream = 0 : i64, uid = 94 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<3x1024x128xf32>, memref<3xf32>, memref<3x131072xf32>, memref<1024x3x128xf32>) -> ()
    %view_45 = memref.view %arg59[%c0][] : memref<65536xi8> to memref<128x128xf32>
    %view_46 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<3072x128xf32>
    %view_47 = memref.view %arg361[%c1575936][] : memref<122096960xi8> to memref<3072x128xf32>
    call @xla.gpu.gemm_33(%view_46, %view_45, %view_47) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.11">, uid = 63 : i64} : (memref<3072x128xf32>, memref<128x128xf32>, memref<3072x128xf32>) -> ()
    %view_48 = memref.view %arg157[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_68(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c32_i32, %c4_i32, %c1_i32, %view_47, %view_48) {kernel = "fusion_132", rt.trace = #rt.hlo_trace<"fusion.132">, stream = 0 : i64, uid = 93 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<3072x128xf32>, memref<128xf32>) -> ()
    %view_49 = memref.view %arg60[%c0][] : memref<65536xi8> to memref<128x128xf32>
    call @xla.gpu.gemm_33(%view_47, %view_49, %view_46) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.12">, uid = 62 : i64} : (memref<3072x128xf32>, memref<128x128xf32>, memref<3072x128xf32>) -> ()
    %view_50 = memref.view %arg361[%c1575936][] : memref<122096960xi8> to memref<1024x3x128xf32>
    call @xla.gpu.func.launch_67(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_35, %view_50) {kernel = "wrapped_transpose_219", rt.trace = #rt.hlo_trace<"wrapped_transpose.219">, stream = 0 : i64, uid = 92 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<3x1024x128xf32>, memref<1024x3x128xf32>) -> ()
    %view_51 = memref.view %arg55[%c0][] : memref<98304xi8> to memref<384x64xf32>
    %view_52 = memref.view %arg361[%c1575936][] : memref<122096960xi8> to memref<1024x384xf32>
    %view_53 = memref.view %arg361[%c3148800][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_29(%view_52, %view_51, %view_53) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.5">, uid = 61 : i64} : (memref<1024x384xf32>, memref<384x64xf32>, memref<1024x64xf32>) -> ()
    %view_54 = memref.view %arg56[%c0][] : memref<98304xi8> to memref<64x384xf32>
    %view_55 = memref.view %arg361[%c3673088][] : memref<122096960xi8> to memref<1024x384xf32>
    call @xla.gpu.gemm_32(%view_53, %view_54, %view_55) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.6">, uid = 60 : i64} : (memref<1024x64xf32>, memref<64x384xf32>, memref<1024x384xf32>) -> ()
    %view_56 = memref.view %arg361[%c1575936][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_66(%c0_i32, %c1024_i32, %c2_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_33, %view, %view_30, %view_26, %view_1, %view_56) {kernel = "fusion_222", rt.trace = #rt.hlo_trace<"fusion.222">, stream = 0 : i64, uid = 91 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x128xf32>, memref<f32>, memref<128xf32>, memref<1024x256xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_57 = memref.view %arg104[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_58 = memref.view %arg158[%c0][] : memref<512xi8> to memref<128xf32>
    %view_59 = memref.view %arg361[%c70257664][] : memref<122096960xi8> to memref<3x1024x128xf32>
    %view_60 = memref.view %arg361[%c2100224][] : memref<122096960xi8> to memref<1024x384xf32>
    call @xla.gpu.func.launch_65(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_46, %view_58, %view_35, %view_43, %view_41, %view_55, %view_57, %view_56, %view_1, %view_15, %view_60) {kernel = "fusion_131", rt.trace = #rt.hlo_trace<"fusion.131">, stream = 0 : i64, uid = 90 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<3072x128xf32>, memref<128xf32>, memref<3x1024x128xf32>, memref<3xf32>, memref<3x131072xf32>, memref<1024x384xf32>, memref<384xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x384xf32>) -> ()
    %view_61 = memref.view %arg361[%c71830528][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_64(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_59, %view, %view_1, %view_61) {kernel = "fusion_129", rt.trace = #rt.hlo_trace<"fusion.129">, stream = 0 : i64, uid = 89 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<3x1024x128xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_62 = memref.view %arg110[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_63 = memref.view %arg109[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_64 = memref.view %arg361[%c68684800][] : memref<122096960xi8> to memref<1024x3x128xf32>
    call @xla.gpu.func.launch_63(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_1, %view_63, %view_62, %view_59, %view_61, %view_64) {kernel = "fusion_128", rt.trace = #rt.hlo_trace<"fusion.128">, stream = 0 : i64, uid = 88 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<3x1024x128xf32>, memref<1024xf32>, memref<1024x3x128xf32>) -> ()
    %view_65 = memref.view %arg35[%c0][] : memref<196608xi8> to memref<384x128xf32>
    %view_66 = memref.view %arg361[%c68684800][] : memref<122096960xi8> to memref<1024x384xf32>
    call @xla.gpu.gemm_26(%view_66, %view_65, %view_9) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.13">, uid = 59 : i64} : (memref<1024x384xf32>, memref<384x128xf32>, memref<1024x128xf32>) -> ()
    %view_67 = memref.view %arg166[%c0][] : memref<512xi8> to memref<128xf32>
    %view_68 = memref.view %arg361[%c71044096][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_23(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_9, %view_67, %view_68) {kernel = "fusion_208", rt.trace = #rt.hlo_trace<"fusion.208">, stream = 0 : i64, uid = 87 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    %view_69 = memref.view %arg361[%c70781952][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_62(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_55, %view, %view_1, %view_69) {kernel = "fusion_141", rt.trace = #rt.hlo_trace<"fusion.141">, stream = 0 : i64, uid = 86 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x384xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_70 = memref.view %arg108[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_71 = memref.view %arg107[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_72 = memref.view %arg361[%c5245952][] : memref<122096960xi8> to memref<1024x384xf32>
    call @xla.gpu.func.launch_61(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_1, %view_71, %view_70, %view_55, %view_69, %view_72) {kernel = "fusion_140", rt.trace = #rt.hlo_trace<"fusion.140">, stream = 0 : i64, uid = 85 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<1024x384xf32>, memref<1024xf32>, memref<1024x384xf32>) -> ()
    %view_73 = memref.view %arg23[%c0][] : memref<786432xi8> to memref<384x512xf32>
    %view_74 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x512xf32>
    call @xla.gpu.gemm_31(%view_72, %view_73, %view_74) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.7">, uid = 58 : i64} : (memref<1024x384xf32>, memref<384x512xf32>, memref<1024x512xf32>) -> ()
    %view_75 = memref.view %arg97[%c0][] : memref<2048xi8> to memref<512xf32>
    call @xla.gpu.func.launch_44(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_74, %view_75, %view_1) {kernel = "fusion_205", rt.trace = #rt.hlo_trace<"fusion.205">, stream = 0 : i64, uid = 84 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x512xf32>, memref<512xf32>, memref<1024xf32>) -> ()
    call @xla.gpu.func.launch_43(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_74, %view_75, %view_69) {kernel = "fusion_137", rt.trace = #rt.hlo_trace<"fusion.137">, stream = 0 : i64, uid = 83 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x512xf32>, memref<512xf32>, memref<1024xf32>) -> ()
    %view_76 = memref.view %arg99[%c0][] : memref<2048xi8> to memref<512xf32>
    %view_77 = memref.view %arg98[%c0][] : memref<2048xi8> to memref<512xf32>
    call @xla.gpu.func.launch_42(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_69, %view_77, %view_76, %view_1, %view_74, %view_75) {kernel = "fusion_136", rt.trace = #rt.hlo_trace<"fusion.136">, stream = 0 : i64, uid = 82 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<512xf32>, memref<512xf32>, memref<1024xf32>, memref<1024x512xf32>, memref<512xf32>) -> ()
    %view_78 = memref.view %arg29[%c0][] : memref<262144xi8> to memref<512x128xf32>
    %view_79 = memref.view %arg361[%c68684800][] : memref<122096960xi8> to memref<1024x128xf32>
    call @xla.gpu.gemm_27(%view_74, %view_78, %view_79) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.8">, uid = 57 : i64} : (memref<1024x512xf32>, memref<512x128xf32>, memref<1024x128xf32>) -> ()
    %view_80 = memref.view %arg165[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_23(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_79, %view_80, %view_56) {kernel = "fusion_208", rt.trace = #rt.hlo_trace<"fusion.206">, stream = 0 : i64, uid = 81 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    call @xla.gpu.func.launch_60(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_60, %view_1) {kernel = "fusion_125", rt.trace = #rt.hlo_trace<"fusion.125">, stream = 0 : i64, uid = 80 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x384xf32>, memref<1024xf32>) -> ()
    %view_81 = memref.view %arg106[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_82 = memref.view %arg105[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_83 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x384xf32>
    call @xla.gpu.func.launch_59(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_60, %view_1, %view_82, %view_81, %view_83) {kernel = "fusion_124", rt.trace = #rt.hlo_trace<"fusion.124">, stream = 0 : i64, uid = 79 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x384xf32>, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<1024x384xf32>) -> ()
    %view_84 = memref.view %arg33[%c0][] : memref<196608xi8> to memref<384x128xf32>
    %view_85 = memref.view %arg361[%c69733376][] : memref<122096960xi8> to memref<1024x128xf32>
    call @xla.gpu.gemm_26(%view_83, %view_84, %view_85) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.14">, uid = 56 : i64} : (memref<1024x384xf32>, memref<384x128xf32>, memref<1024x128xf32>) -> ()
    %view_86 = memref.view %arg159[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_58(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_85, %view_86) {kernel = "fusion_123", rt.trace = #rt.hlo_trace<"fusion.123">, stream = 0 : i64, uid = 78 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x128xf32>, memref<128xf32>) -> ()
    %view_87 = memref.view %arg69[%c0][] : memref<32768xi8> to memref<128x64xf32>
    %view_88 = memref.view %arg361[%c70781952][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_9(%view_85, %view_87, %view_88) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.16">, uid = 55 : i64} : (memref<1024x128xf32>, memref<128x64xf32>, memref<1024x64xf32>) -> ()
    %view_89 = memref.view %arg70[%c0][] : memref<32768xi8> to memref<64x128xf32>
    %view_90 = memref.view %arg361[%c69209088][] : memref<122096960xi8> to memref<1024x128xf32>
    call @xla.gpu.gemm_30(%view_88, %view_89, %view_90) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.17">, uid = 54 : i64} : (memref<1024x64xf32>, memref<64x128xf32>, memref<1024x128xf32>) -> ()
    %view_91 = memref.view %arg160[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_57(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_85, %view_90, %view_91, %view_1, %view_69) {kernel = "fusion_199", rt.trace = #rt.hlo_trace<"fusion.199">, stream = 0 : i64, uid = 77 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_92 = memref.view %arg361[%c70786048][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_56(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_69, %view_56, %view_68, %view_85, %view_9, %view_67, %view_79, %view_80, %view_92) {kernel = "fusion_121", rt.trace = #rt.hlo_trace<"fusion.121">, stream = 0 : i64, uid = 76 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    %view_93 = memref.view %arg112[%c0][] : memref<1536xi8> to memref<384xf32>
    %view_94 = memref.view %arg111[%c0][] : memref<1536xi8> to memref<384xf32>
    call @xla.gpu.func.launch_55(%c0_i32, %c768_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_92, %view_94, %view_93, %view_69, %view_56, %view_68, %view_85, %view_9, %view_67, %view_79, %view_80, %view_83) {kernel = "fusion_120", rt.trace = #rt.hlo_trace<"fusion.120">, stream = 0 : i64, uid = 75 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x384xf32>) -> ()
    %view_95 = memref.view %arg57[%c0][] : memref<98304xi8> to memref<384x64xf32>
    %view_96 = memref.view %arg361[%c121572352][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_29(%view_83, %view_95, %view_96) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.15">, uid = 53 : i64} : (memref<1024x384xf32>, memref<384x64xf32>, memref<1024x64xf32>) -> ()
    %view_97 = memref.view %arg361[%c68684800][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_54(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_85, %view_90, %view_91, %view_97) {kernel = "fusion_117", rt.trace = #rt.hlo_trace<"fusion.117">, stream = 0 : i64, uid = 74 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    %view_98 = memref.view %arg162[%c0][] : memref<512xi8> to memref<128xf32>
    %view_99 = memref.view %arg161[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_53(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_97, %view_99, %view_98, %view_1, %view_85, %view_90, %view_91) {kernel = "fusion_116", rt.trace = #rt.hlo_trace<"fusion.116">, stream = 0 : i64, uid = 73 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>) -> ()
    %view_100 = memref.view %arg27[%c0][] : memref<262144xi8> to memref<128x512xf32>
    call @xla.gpu.gemm_28(%view_85, %view_100, %view_74) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.18">, uid = 52 : i64} : (memref<1024x128xf32>, memref<128x512xf32>, memref<1024x512xf32>) -> ()
    %view_101 = memref.view %arg94[%c0][] : memref<2048xi8> to memref<512xf32>
    call @xla.gpu.func.launch_44(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_74, %view_101, %view_1) {kernel = "fusion_205", rt.trace = #rt.hlo_trace<"fusion.200">, stream = 0 : i64, uid = 72 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x512xf32>, memref<512xf32>, memref<1024xf32>) -> ()
    call @xla.gpu.func.launch_43(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_74, %view_101, %view_97) {kernel = "fusion_137", rt.trace = #rt.hlo_trace<"fusion.113">, stream = 0 : i64, uid = 71 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x512xf32>, memref<512xf32>, memref<1024xf32>) -> ()
    %view_102 = memref.view %arg96[%c0][] : memref<2048xi8> to memref<512xf32>
    %view_103 = memref.view %arg95[%c0][] : memref<2048xi8> to memref<512xf32>
    call @xla.gpu.func.launch_42(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_97, %view_103, %view_102, %view_1, %view_74, %view_101) {kernel = "fusion_136", rt.trace = #rt.hlo_trace<"fusion.112">, stream = 0 : i64, uid = 70 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<512xf32>, memref<512xf32>, memref<1024xf32>, memref<1024x512xf32>, memref<512xf32>) -> ()
    %view_104 = memref.view %arg28[%c0][] : memref<262144xi8> to memref<512x128xf32>
    call @xla.gpu.gemm_27(%view_74, %view_104, %view_79) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.19">, uid = 51 : i64} : (memref<1024x512xf32>, memref<512x128xf32>, memref<1024x128xf32>) -> ()
    %view_105 = memref.view %arg163[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_23(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_79, %view_105, %view_1) {kernel = "fusion_208", rt.trace = #rt.hlo_trace<"fusion.201">, stream = 0 : i64, uid = 69 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    %view_106 = memref.view %arg34[%c0][] : memref<196608xi8> to memref<384x128xf32>
    call @xla.gpu.gemm_26(%view_34, %view_106, %view_90) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.20">, uid = 50 : i64} : (memref<1024x384xf32>, memref<384x128xf32>, memref<1024x128xf32>) -> ()
    %view_107 = memref.view %arg164[%c0][] : memref<512xi8> to memref<128xf32>
    %view_108 = memref.view %arg361[%c69733376][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_23(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_90, %view_107, %view_108) {kernel = "fusion_204", rt.trace = #rt.hlo_trace<"fusion.204">, stream = 0 : i64, uid = 68 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    %view_109 = memref.view %arg361[%c69737472][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_52(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_108, %view_90, %view_107, %view_79, %view_105, %view_109) {kernel = "fusion_102", rt.trace = #rt.hlo_trace<"fusion.102">, stream = 0 : i64, uid = 67 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    %view_110 = memref.view %arg118[%c0][] : memref<1024xi8> to memref<256xf32>
    %view_111 = memref.view %arg117[%c0][] : memref<1024xi8> to memref<256xf32>
    %view_112 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x256xf32>
    call @xla.gpu.func.launch_51(%c0_i32, %c512_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_109, %view_111, %view_110, %view_1, %view_108, %view_90, %view_107, %view_79, %view_105, %view_112) {kernel = "fusion_101", rt.trace = #rt.hlo_trace<"fusion.101">, stream = 0 : i64, uid = 66 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<256xf32>, memref<256xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x256xf32>) -> ()
    %view_113 = memref.view %arg61[%c0][] : memref<65536xi8> to memref<256x64xf32>
    %view_114 = memref.view %arg361[%c121834496][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_25(%view_112, %view_113, %view_114) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.21">, uid = 49 : i64} : (memref<1024x256xf32>, memref<256x64xf32>, memref<1024x64xf32>) -> ()
    %view_115 = memref.view %arg37[%c0][] : memref<131072xi8> to memref<128x256xf32>
    %view_116 = memref.view %arg361[%c6294528][] : memref<122096960xi8> to memref<1024x256xf32>
    call @xla.gpu.gemm_24(%view_24, %view_115, %view_116) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.24">, uid = 48 : i64} : (memref<1024x128xf32>, memref<128x256xf32>, memref<1024x256xf32>) -> ()
    %view_117 = memref.view %arg113[%c0][] : memref<1024xi8> to memref<256xf32>
    %view_118 = memref.view %arg3[%c0][] : memref<1048576xi8> to memref<1024x256xf32>
    %view_119 = memref.view %arg2[%c0][] : memref<1048576xi8> to memref<1024x256xf32>
    %view_120 = memref.view %arg361[%c33557504][] : memref<122096960xi8> to memref<3x1024x256xf32>
    call @xla.gpu.func.launch_50(%c0_i32, %c1536_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_118, %view_119, %view_116, %view_117, %view_120) {kernel = "fusion_94", rt.trace = #rt.hlo_trace<"fusion.94">, stream = 0 : i64, uid = 65 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x256xf32>, memref<1024x256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<3x1024x256xf32>) -> ()
    %view_121 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x3x256xf32>
    call @xla.gpu.func.launch_49(%c0_i32, %c1536_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_120, %view_121) {kernel = "wrapped_transpose_225", rt.trace = #rt.hlo_trace<"wrapped_transpose.225">, stream = 0 : i64, uid = 64 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<3x1024x256xf32>, memref<1024x3x256xf32>) -> ()
    %view_122 = memref.view %arg31[%c0][] : memref<196608xi8> to memref<768x64xf32>
    %view_123 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x768xf32>
    %view_124 = memref.view %arg361[%c7343104][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_23(%view_123, %view_122, %view_124) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.25">, uid = 47 : i64} : (memref<1024x768xf32>, memref<768x64xf32>, memref<1024x64xf32>) -> ()
    %view_125 = memref.view %arg32[%c0][] : memref<196608xi8> to memref<64x768xf32>
    %view_126 = memref.view %arg361[%c3148800][] : memref<122096960xi8> to memref<1024x768xf32>
    call @xla.gpu.gemm_22(%view_124, %view_125, %view_126) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.26">, uid = 46 : i64} : (memref<1024x64xf32>, memref<64x768xf32>, memref<1024x768xf32>) -> ()
    %view_127 = memref.view %arg361[%c7343104][] : memref<122096960xi8> to memref<1024xf32>
    %view_128 = memref.view %arg361[%c7347200][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_48(%c0_i32, %c1024_i32, %c3_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_119, %view, %view_118, %view_116, %view_117, %view_1, %view_127, %view_128) {kernel = "fusion_220", rt.trace = #rt.hlo_trace<"fusion.220">, stream = 0 : i64, uid = 63 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x256xf32>, memref<f32>, memref<1024x256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_129 = memref.view %arg84[%c0][] : memref<3072xi8> to memref<768xf32>
    call @xla.gpu.func.launch_47(%c0_i32, %c1536_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_126, %view_129, %view_120, %view_128, %view_1, %view_127, %view_123) {kernel = "fusion_93", rt.trace = #rt.hlo_trace<"fusion.93">, stream = 0 : i64, uid = 62 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x768xf32>, memref<768xf32>, memref<3x1024x256xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x768xf32>) -> ()
    %view_130 = memref.view %arg361[%c9440256][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_46(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c64_i32, %c1_i32, %c1_i32, %view_126, %view, %view_1, %view_130) {kernel = "fusion_91", rt.trace = #rt.hlo_trace<"fusion.91">, stream = 0 : i64, uid = 61 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x768xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_131 = memref.view %arg88[%c0][] : memref<3072xi8> to memref<768xf32>
    %view_132 = memref.view %arg87[%c0][] : memref<3072xi8> to memref<768xf32>
    %view_133 = memref.view %arg361[%c6294528][] : memref<122096960xi8> to memref<1024x768xf32>
    call @xla.gpu.func.launch_45(%c0_i32, %c1536_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_1, %view_132, %view_131, %view_126, %view_130, %view_133) {kernel = "fusion_90", rt.trace = #rt.hlo_trace<"fusion.90">, stream = 0 : i64, uid = 60 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<768xf32>, memref<768xf32>, memref<1024x768xf32>, memref<1024xf32>, memref<1024x768xf32>) -> ()
    %view_134 = memref.view %arg21[%c0][] : memref<1572864xi8> to memref<768x512xf32>
    %view_135 = memref.view %arg361[%c3148800][] : memref<122096960xi8> to memref<1024x512xf32>
    call @xla.gpu.gemm_21(%view_133, %view_134, %view_135) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.27">, uid = 45 : i64} : (memref<1024x768xf32>, memref<768x512xf32>, memref<1024x512xf32>) -> ()
    %view_136 = memref.view %arg91[%c0][] : memref<2048xi8> to memref<512xf32>
    call @xla.gpu.func.launch_44(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_135, %view_136, %view_1) {kernel = "fusion_205", rt.trace = #rt.hlo_trace<"fusion.192">, stream = 0 : i64, uid = 59 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x512xf32>, memref<512xf32>, memref<1024xf32>) -> ()
    %view_137 = memref.view %arg361[%c5245952][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_43(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_135, %view_136, %view_137) {kernel = "fusion_137", rt.trace = #rt.hlo_trace<"fusion.87">, stream = 0 : i64, uid = 58 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x512xf32>, memref<512xf32>, memref<1024xf32>) -> ()
    %view_138 = memref.view %arg93[%c0][] : memref<2048xi8> to memref<512xf32>
    %view_139 = memref.view %arg92[%c0][] : memref<2048xi8> to memref<512xf32>
    call @xla.gpu.func.launch_42(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_137, %view_139, %view_138, %view_1, %view_135, %view_136) {kernel = "fusion_136", rt.trace = #rt.hlo_trace<"fusion.86">, stream = 0 : i64, uid = 57 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<512xf32>, memref<512xf32>, memref<1024xf32>, memref<1024x512xf32>, memref<512xf32>) -> ()
    %view_140 = memref.view %arg24[%c0][] : memref<524288xi8> to memref<512x256xf32>
    %view_141 = memref.view %arg361[%c39848960][] : memref<122096960xi8> to memref<1024x256xf32>
    call @xla.gpu.gemm_20(%view_135, %view_140, %view_141) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.28">, uid = 44 : i64} : (memref<1024x512xf32>, memref<512x256xf32>, memref<1024x256xf32>) -> ()
    %view_142 = memref.view %arg115[%c0][] : memref<1024xi8> to memref<256xf32>
    %view_143 = memref.view %arg361[%c41946112][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_37(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_141, %view_142, %view_143) {kernel = "fusion_193", rt.trace = #rt.hlo_trace<"fusion.193">, stream = 0 : i64, uid = 56 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x256xf32>, memref<256xf32>, memref<1024xf32>) -> ()
    call @xla.gpu.func.launch_41(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c64_i32, %c1_i32, %c1_i32, %view_123, %view_1) {kernel = "fusion_80", rt.trace = #rt.hlo_trace<"fusion.80">, stream = 0 : i64, uid = 55 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x768xf32>, memref<1024xf32>) -> ()
    %view_144 = memref.view %arg86[%c0][] : memref<3072xi8> to memref<768xf32>
    %view_145 = memref.view %arg85[%c0][] : memref<3072xi8> to memref<768xf32>
    call @xla.gpu.func.launch_40(%c0_i32, %c1536_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_123, %view_1, %view_145, %view_144, %view_126) {kernel = "fusion_79", rt.trace = #rt.hlo_trace<"fusion.79">, stream = 0 : i64, uid = 54 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x768xf32>, memref<1024xf32>, memref<768xf32>, memref<768xf32>, memref<1024x768xf32>) -> ()
    %view_146 = memref.view %arg22[%c0][] : memref<786432xi8> to memref<768x256xf32>
    %view_147 = memref.view %arg361[%c40897536][] : memref<122096960xi8> to memref<1024x256xf32>
    call @xla.gpu.gemm_19(%view_126, %view_146, %view_147) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.32">, uid = 43 : i64} : (memref<1024x768xf32>, memref<768x256xf32>, memref<1024x256xf32>) -> ()
    %view_148 = memref.view %arg114[%c0][] : memref<1024xi8> to memref<256xf32>
    call @xla.gpu.func.launch_37(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_147, %view_148, %view_1) {kernel = "fusion_191", rt.trace = #rt.hlo_trace<"fusion.191">, stream = 0 : i64, uid = 53 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x256xf32>, memref<256xf32>, memref<1024xf32>) -> ()
    %view_149 = memref.view %arg167[%c0][] : memref<384xi8> to memref<3x32xf32>
    %view_150 = memref.view %arg361[%c33557504][] : memref<122096960xi8> to memref<3x262144xf32>
    %reinterpret_cast_151 = memref.reinterpret_cast %view_150 to offset: [0], sizes: [262144, 3], strides: [1, 262144] : memref<3x262144xf32> to memref<262144x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 262144)>>
    %view_152 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<262144x32xf32>
    call @xla.gpu.gemm_18(%reinterpret_cast_151, %view_149, %view_152) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.29">, uid = 42 : i64} : (memref<262144x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 262144)>>, memref<3x32xf32>, memref<262144x32xf32>) -> ()
    %view_153 = memref.view %arg354[%c0][] : memref<128xi8> to memref<32xf32>
    call @xla.gpu.func.launch_39(%c0_i32, %c16384_i32, %c1_i32, %c1_i32, %c8_i32, %c16_i32, %c1_i32, %view_152, %view_153) {kernel = "fusion_84", rt.trace = #rt.hlo_trace<"fusion.84">, stream = 0 : i64, uid = 52 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<262144x32xf32>, memref<32xf32>) -> ()
    %view_154 = memref.view %arg168[%c0][] : memref<384xi8> to memref<32x3xf32>
    %view_155 = memref.view %arg361[%c36703232][] : memref<122096960xi8> to memref<3x262144xf32>
    %reinterpret_cast_156 = memref.reinterpret_cast %view_155 to offset: [0], sizes: [262144, 3], strides: [1, 262144] : memref<3x262144xf32> to memref<262144x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 262144)>>
    call @xla.gpu.gemm_17(%view_152, %view_154, %reinterpret_cast_156) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.30">, uid = 41 : i64} : (memref<262144x32xf32>, memref<32x3xf32>, memref<262144x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 262144)>>) -> ()
    call @xla.gpu.func.launch_38(%c0_i32, %c512_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_120, %view_155, %view_112) {kernel = "fusion_83", rt.trace = #rt.hlo_trace<"fusion.83">, stream = 0 : i64, uid = 51 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<3x1024x256xf32>, memref<3x262144xf32>, memref<1024x256xf32>) -> ()
    %view_157 = memref.view %arg26[%c0][] : memref<262144xi8> to memref<256x256xf32>
    %view_158 = memref.view %arg361[%c3148800][] : memref<122096960xi8> to memref<1024x256xf32>
    call @xla.gpu.gemm_16(%view_112, %view_157, %view_158) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.31">, uid = 40 : i64} : (memref<1024x256xf32>, memref<256x256xf32>, memref<1024x256xf32>) -> ()
    %view_159 = memref.view %arg116[%c0][] : memref<1024xi8> to memref<256xf32>
    %view_160 = memref.view %arg361[%c4197376][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_37(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_158, %view_159, %view_160) {kernel = "fusion_191", rt.trace = #rt.hlo_trace<"fusion.194">, stream = 0 : i64, uid = 50 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x256xf32>, memref<256xf32>, memref<1024xf32>) -> ()
    %view_161 = memref.view %arg361[%c4201472][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_36(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c64_i32, %c1_i32, %c1_i32, %view_1, %view_143, %view_160, %view_147, %view_148, %view_158, %view_159, %view_141, %view_142, %view_161) {kernel = "fusion_76", rt.trace = #rt.hlo_trace<"fusion.76">, stream = 0 : i64, uid = 49 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024xf32>) -> ()
    %view_162 = memref.view %arg90[%c0][] : memref<3072xi8> to memref<768xf32>
    %view_163 = memref.view %arg89[%c0][] : memref<3072xi8> to memref<768xf32>
    call @xla.gpu.func.launch_35(%c0_i32, %c1536_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_161, %view_163, %view_162, %view_1, %view_143, %view_160, %view_147, %view_148, %view_158, %view_159, %view_141, %view_142, %view_123) {kernel = "fusion_75", rt.trace = #rt.hlo_trace<"fusion.75">, stream = 0 : i64, uid = 48 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<768xf32>, memref<768xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x768xf32>) -> ()
    %view_164 = memref.view %arg25[%c0][] : memref<294912xi8> to memref<768x96xf32>
    %view_165 = memref.view %arg361[%c121179136][] : memref<122096960xi8> to memref<1024x96xf32>
    call @xla.gpu.gemm_15(%view_123, %view_164, %view_165) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.33">, uid = 39 : i64} : (memref<1024x768xf32>, memref<768x96xf32>, memref<1024x96xf32>) -> ()
    %view_166 = memref.view %arg8[%c0][] : memref<9437184xi8> to memref<1024x2304xf32>
    %view_167 = memref.view %arg9[%c0][] : memref<1835008xi8> to memref<1024x448xf32>
    %view_168 = memref.view %arg361[%c30297088][] : memref<122096960xi8> to memref<1024x2752xf32>
    call @xla.gpu.func.launch_34(%c0_i32, %c5504_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_166, %view_167, %view_168) {kernel = "wrapped_concatenate_9", rt.trace = #rt.hlo_trace<"wrapped_concatenate.9">, stream = 0 : i64, uid = 47 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x2304xf32>, memref<1024x448xf32>, memref<1024x2752xf32>) -> ()
    %view_169 = memref.view %arg64[%c0][] : memref<44032xi8> to memref<64x172xf32>
    %view_170 = memref.view %arg361[%c30297088][] : memref<122096960xi8> to memref<44032x64xf32>
    %view_171 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<44032x172xf32>
    call @xla.gpu.gemm_14(%view_170, %view_169, %view_171) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.34">, uid = 38 : i64} : (memref<44032x64xf32>, memref<64x172xf32>, memref<44032x172xf32>) -> ()
    %view_172 = memref.view %arg121[%c0][] : memref<688xi8> to memref<172xf32>
    %view_173 = memref.view %arg361[%c60591104][] : memref<122096960xi8> to memref<1024x43x172xf32>
    call @xla.gpu.func.launch_33(%c0_i32, %c14792_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_171, %view_172, %view_173) {kernel = "fusion_74", rt.trace = #rt.hlo_trace<"fusion.74">, stream = 0 : i64, uid = 46 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<44032x172xf32>, memref<172xf32>, memref<1024x43x172xf32>) -> ()
    %view_174 = memref.view %arg361[%c90885120][] : memref<122096960xi8> to memref<44032xf32>
    %view_175 = memref.view %arg361[%c91061248][] : memref<122096960xi8> to memref<44032xf32>
    call @xla.gpu.func.launch_31(%c0_i32, %c44032_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_173, %view_174, %view_175) {kernel = "fusion_72", rt.trace = #rt.hlo_trace<"fusion.72">, stream = 0 : i64, uid = 45 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x43x172xf32>, memref<44032xf32>, memref<44032xf32>) -> ()
    %view_176 = memref.view %arg122[%c0][] : memref<688xi8> to memref<172xf32>
    %view_177 = memref.view %arg123[%c0][] : memref<688xi8> to memref<172xf32>
    call @xla.gpu.func.launch_30(%c0_i32, %c14792_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_174, %view_175, %view_176, %view_177, %view_173) {kernel = "fusion_71", rt.trace = #rt.hlo_trace<"fusion.71">, stream = 0 : i64, uid = 44 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<44032xf32>, memref<44032xf32>, memref<172xf32>, memref<172xf32>, memref<1024x43x172xf32>) -> ()
    %view_178 = memref.view %arg17[%c0][] : memref<10176896xi8> to memref<43x172x344xf32>
    %view_179 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<43x1024x344xf32>
    call @xla.gpu.gemm_13(%view_173, %view_178, %view_179) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.35">, uid = 37 : i64} : (memref<1024x43x172xf32>, memref<43x172x344xf32>, memref<43x1024x344xf32>) -> ()
    %view_180 = memref.view %arg62[%c0][] : memref<59168xi8> to memref<43x344xf32>
    %view_181 = memref.view %arg361[%c60591104][] : memref<122096960xi8> to memref<1024x43x344xf32>
    call @xla.gpu.func.launch_29(%c0_i32, %c29584_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_179, %view_180, %view_181) {kernel = "fusion_70", rt.trace = #rt.hlo_trace<"fusion.70">, stream = 0 : i64, uid = 43 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<43x1024x344xf32>, memref<43x344xf32>, memref<1024x43x344xf32>) -> ()
    %view_182 = memref.view %arg18[%c0][] : memref<10176896xi8> to memref<43x344x172xf32>
    %view_183 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<43x1024x172xf32>
    call @xla.gpu.gemm_12(%view_181, %view_182, %view_183) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.36">, uid = 36 : i64} : (memref<1024x43x344xf32>, memref<43x344x172xf32>, memref<43x1024x172xf32>) -> ()
    %view_184 = memref.view %arg71[%c0][] : memref<29584xi8> to memref<43x172xf32>
    call @xla.gpu.func.launch_32(%c0_i32, %c14792_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_183, %view_184, %view_173) {kernel = "fusion_69", rt.trace = #rt.hlo_trace<"fusion.69">, stream = 0 : i64, uid = 42 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<43x1024x172xf32>, memref<43x172xf32>, memref<1024x43x172xf32>) -> ()
    call @xla.gpu.func.launch_31(%c0_i32, %c44032_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_173, %view_174, %view_175) {kernel = "fusion_72", rt.trace = #rt.hlo_trace<"fusion.67">, stream = 0 : i64, uid = 41 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x43x172xf32>, memref<44032xf32>, memref<44032xf32>) -> ()
    %view_185 = memref.view %arg124[%c0][] : memref<688xi8> to memref<172xf32>
    %view_186 = memref.view %arg125[%c0][] : memref<688xi8> to memref<172xf32>
    call @xla.gpu.func.launch_30(%c0_i32, %c14792_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_174, %view_175, %view_185, %view_186, %view_173) {kernel = "fusion_71", rt.trace = #rt.hlo_trace<"fusion.66">, stream = 0 : i64, uid = 40 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<44032xf32>, memref<44032xf32>, memref<172xf32>, memref<172xf32>, memref<1024x43x172xf32>) -> ()
    %view_187 = memref.view %arg19[%c0][] : memref<10176896xi8> to memref<43x172x344xf32>
    call @xla.gpu.gemm_13(%view_173, %view_187, %view_179) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.37">, uid = 35 : i64} : (memref<1024x43x172xf32>, memref<43x172x344xf32>, memref<43x1024x344xf32>) -> ()
    %view_188 = memref.view %arg63[%c0][] : memref<59168xi8> to memref<43x344xf32>
    call @xla.gpu.func.launch_29(%c0_i32, %c29584_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_179, %view_188, %view_181) {kernel = "fusion_70", rt.trace = #rt.hlo_trace<"fusion.65">, stream = 0 : i64, uid = 39 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<43x1024x344xf32>, memref<43x344xf32>, memref<1024x43x344xf32>) -> ()
    %view_189 = memref.view %arg20[%c0][] : memref<10176896xi8> to memref<43x344x172xf32>
    call @xla.gpu.gemm_12(%view_181, %view_189, %view_183) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.38">, uid = 34 : i64} : (memref<1024x43x344xf32>, memref<43x344x172xf32>, memref<43x1024x172xf32>) -> ()
    %view_190 = memref.view %arg72[%c0][] : memref<29584xi8> to memref<43x172xf32>
    %view_191 = memref.view %arg361[%c30297088][] : memref<122096960xi8> to memref<1024x172xf32>
    call @xla.gpu.func.launch_28(%c0_i32, %c1376_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_183, %view_190, %view_191) {kernel = "fusion_64", rt.trace = #rt.hlo_trace<"fusion.64">, stream = 0 : i64, uid = 38 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<43x1024x172xf32>, memref<43x172xf32>, memref<1024x172xf32>) -> ()
    %view_192 = memref.view %arg65[%c0][] : memref<44032xi8> to memref<172x64xf32>
    call @xla.gpu.gemm_11(%view_191, %view_192, %view_7) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.39">, uid = 33 : i64} : (memref<1024x172xf32>, memref<172x64xf32>, memref<1024x64xf32>) -> ()
    %view_193 = memref.view %arg191[%c0][] : memref<256xi8> to memref<64xf32>
    call @xla.gpu.func.launch_27(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_7, %view_193, %view_1) {kernel = "fusion_195", rt.trace = #rt.hlo_trace<"fusion.195">, stream = 0 : i64, uid = 37 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xf32>) -> ()
    call @xla.gpu.func.launch_26(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_7, %view_193, %view_2) {kernel = "fusion_61", rt.trace = #rt.hlo_trace<"fusion.61">, stream = 0 : i64, uid = 36 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xf32>) -> ()
    %view_194 = memref.view %arg193[%c0][] : memref<256xi8> to memref<64xf32>
    %view_195 = memref.view %arg192[%c0][] : memref<256xi8> to memref<64xf32>
    call @xla.gpu.func.launch_25(%c0_i32, %c512_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_2, %view_195, %view_194, %view_1, %view_7, %view_193) {kernel = "fusion_60", rt.trace = #rt.hlo_trace<"fusion.60">, stream = 0 : i64, uid = 35 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>) -> ()
    %view_196 = memref.view %arg75[%c0][] : memref<16384xi8> to memref<64x64xf32>
    %view_197 = memref.view %arg361[%c5016576][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_3(%view_7, %view_196, %view_197) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.40">, uid = 32 : i64} : (memref<1024x64xf32>, memref<64x64xf32>, memref<1024x64xf32>) -> ()
    %view_198 = memref.view %arg361[%c527360][] : memref<122096960xi8> to memref<1024x128xf32>
    call @xla.gpu.func.launch_24(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_28, %view_198) {kernel = "wrapped_slice_8", rt.trace = #rt.hlo_trace<"wrapped_slice.8">, stream = 0 : i64, uid = 34 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x258xf32>, memref<1024x128xf32>) -> ()
    %view_199 = memref.view %arg58[%c0][] : memref<65536xi8> to memref<128x128xf32>
    %view_200 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x128xf32>
    call @xla.gpu.gemm_10(%view_198, %view_199, %view_200) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.22">, uid = 31 : i64} : (memref<1024x128xf32>, memref<128x128xf32>, memref<1024x128xf32>) -> ()
    %view_201 = memref.view %arg150[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_23(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_200, %view_201, %view_1) {kernel = "fusion_208", rt.trace = #rt.hlo_trace<"fusion.196">, stream = 0 : i64, uid = 33 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    %view_202 = memref.view %arg361[%c527360][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_22(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_200, %view_201, %view_202) {kernel = "fusion_98", rt.trace = #rt.hlo_trace<"fusion.98">, stream = 0 : i64, uid = 32 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) -> ()
    %view_203 = memref.view %arg151[%c0][] : memref<512xi8> to memref<128xf32>
    %view_204 = memref.view %arg152[%c0][] : memref<512xi8> to memref<128xf32>
    call @xla.gpu.func.launch_21(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_202, %view_203, %view_204, %view_1, %view_200, %view_201) {kernel = "fusion_97", rt.trace = #rt.hlo_trace<"fusion.97">, stream = 0 : i64, uid = 31 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<128xf32>) -> ()
    %view_205 = memref.view %arg67[%c0][] : memref<32768xi8> to memref<128x64xf32>
    %view_206 = memref.view %arg361[%c5278720][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_9(%view_200, %view_205, %view_206) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.23">, uid = 30 : i64} : (memref<1024x128xf32>, memref<128x64xf32>, memref<1024x64xf32>) -> ()
    %view_207 = memref.view %arg11[%c0][] : memref<262144xi8> to memref<1024x64xf32>
    %view_208 = memref.view %arg14[%c0][] : memref<4718592xi8> to memref<1024x1152xf32>
    %view_209 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x18x64xf32>
    call @xla.gpu.func.launch_20(%c0_i32, %c2304_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_207, %view_208, %view_209) {kernel = "fusion_164", rt.trace = #rt.hlo_trace<"fusion.164">, stream = 0 : i64, uid = 30 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x64xf32>, memref<1024x1152xf32>, memref<1024x18x64xf32>) -> ()
    %view_210 = memref.view %arg119[%c0][] : memref<1024xi8> to memref<64x4xf32>
    %view_211 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<18432x64xf32>
    %view_212 = memref.view %arg361[%c4721664][] : memref<122096960xi8> to memref<18432x4xf32>
    call @xla.gpu.gemm_8(%view_211, %view_210, %view_212) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.1">, uid = 29 : i64} : (memref<18432x64xf32>, memref<64x4xf32>, memref<18432x4xf32>) -> ()
    %view_213 = memref.view %arg16[%c0][] : memref<20971520xi8> to memref<1024x20x256xf32>
    %view_214 = memref.view %arg36[%c0][] : memref<131072xi8> to memref<1024x20xf32>
    call @xla.gpu.func.launch_19(%c0_i32, %c20480_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_213, %view_208, %view_214) {kernel = "fusion_166", rt.trace = #rt.hlo_trace<"fusion.166">, stream = 0 : i64, uid = 29 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x20x256xf32>, memref<1024x1152xf32>, memref<1024x20xf32>) -> ()
    call @xla.gpu.func.launch_18(%c0_i32, %c512_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_214, %view_213, %view_7) {kernel = "fusion_165", rt.trace = #rt.hlo_trace<"fusion.165">, stream = 0 : i64, uid = 28 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x20xf32>, memref<1024x20x256xf32>, memref<1024x64xf32>) -> ()
    %view_215 = memref.view %arg120[%c0][] : memref<1024xi8> to memref<64x4xf32>
    %view_216 = memref.view %arg36[%c0][] : memref<131072xi8> to memref<1024x4xf32>
    call @xla.gpu.gemm_7(%view_7, %view_215, %view_216) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call">, uid = 28 : i64} : (memref<1024x64xf32>, memref<64x4xf32>, memref<1024x4xf32>) -> ()
    %view_217 = memref.view %arg359[%c0][] : memref<16xi8> to memref<4xf32>
    %view_218 = memref.view %arg360[%c0][] : memref<16xi8> to memref<4xf32>
    %view_219 = memref.view %arg361[%c314368][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_17(%c0_i32, %c8_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view, %view_212, %view_218, %view_216, %view_217, %view_219) {kernel = "fusion_209", rt.trace = #rt.hlo_trace<"fusion.209">, stream = 0 : i64, uid = 27 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<18432x4xf32>, memref<4xf32>, memref<1024x4xf32>, memref<4xf32>, memref<1024xf32>) -> ()
    %view_220 = memref.view %arg170[%c0][] : memref<304xi8> to memref<76xf32>
    %view_221 = memref.view %arg169[%c0][] : memref<304xi8> to memref<76xf32>
    %view_222 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x76xf32>
    call @xla.gpu.func.launch_16(%c0_i32, %c608_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_221, %view_220, %view_219, %view_212, %view_218, %view_216, %view_217, %view_222) {kernel = "fusion_160", rt.trace = #rt.hlo_trace<"fusion.160">, stream = 0 : i64, uid = 26 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<76xf32>, memref<76xf32>, memref<1024xf32>, memref<18432x4xf32>, memref<4xf32>, memref<1024x4xf32>, memref<4xf32>, memref<1024x76xf32>) -> ()
    %view_223 = memref.view %arg73[%c0][] : memref<19456xi8> to memref<76x64xf32>
    %view_224 = memref.view %arg361[%c527360][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_6(%view_222, %view_223, %view_224) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.2">, uid = 27 : i64} : (memref<1024x76xf32>, memref<76x64xf32>, memref<1024x64xf32>) -> ()
    %view_225 = memref.view %arg201[%c0][] : memref<256xi8> to memref<64xf32>
    %view_226 = memref.view %arg200[%c0][] : memref<256xi8> to memref<64xf32>
    %view_227 = memref.view %arg196[%c0][] : memref<256xi8> to memref<64xf32>
    %view_228 = memref.view %arg202[%c0][] : memref<256xi8> to memref<64xf32>
    %view_229 = memref.view %arg195[%c0][] : memref<256xi8> to memref<64xf32>
    %view_230 = memref.view %arg194[%c0][] : memref<256xi8> to memref<64xf32>
    call @xla.gpu.func.launch_15(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view, %view_197, %view_230, %view_229, %view_165, %view_206, %view_227, %view_114, %view_226, %view_207, %view_96, %view_225, %view_224, %view_228, %view_1) {kernel = "fusion_210", rt.trace = #rt.hlo_trace<"fusion.210">, stream = 0 : i64, uid = 25 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x96xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xf32>) -> ()
    %view_231 = memref.view %arg361[%c1248256][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_14(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_1, %view_197, %view_230, %view_229, %view_165, %view_206, %view_227, %view_114, %view_226, %view_207, %view_96, %view_225, %view_224, %view_228, %view_231) {kernel = "fusion_57", rt.trace = #rt.hlo_trace<"fusion.57">, stream = 0 : i64, uid = 24 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x96xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xf32>) -> ()
    %view_232 = memref.view %arg262[%c0][] : memref<256xi8> to memref<64xf32>
    %view_233 = memref.view %arg261[%c0][] : memref<256xi8> to memref<64xf32>
    %view_234 = memref.view %arg260[%c0][] : memref<256xi8> to memref<64xf32>
    %view_235 = memref.view %arg259[%c0][] : memref<256xi8> to memref<64xf32>
    %view_236 = memref.view %arg258[%c0][] : memref<256xi8> to memref<64xf32>
    %view_237 = memref.view %arg257[%c0][] : memref<256xi8> to memref<64xf32>
    %view_238 = memref.view %arg256[%c0][] : memref<256xi8> to memref<64xf32>
    %view_239 = memref.view %arg255[%c0][] : memref<256xi8> to memref<64xf32>
    %view_240 = memref.view %arg254[%c0][] : memref<256xi8> to memref<64xf32>
    %view_241 = memref.view %arg253[%c0][] : memref<256xi8> to memref<64xf32>
    %view_242 = memref.view %arg252[%c0][] : memref<256xi8> to memref<64xf32>
    %view_243 = memref.view %arg251[%c0][] : memref<256xi8> to memref<64xf32>
    %view_244 = memref.view %arg250[%c0][] : memref<256xi8> to memref<64xf32>
    %view_245 = memref.view %arg249[%c0][] : memref<256xi8> to memref<64xf32>
    %view_246 = memref.view %arg248[%c0][] : memref<256xi8> to memref<64xf32>
    %view_247 = memref.view %arg247[%c0][] : memref<256xi8> to memref<64xf32>
    %view_248 = memref.view %arg246[%c0][] : memref<256xi8> to memref<64xf32>
    %view_249 = memref.view %arg245[%c0][] : memref<256xi8> to memref<64xf32>
    %view_250 = memref.view %arg244[%c0][] : memref<256xi8> to memref<64xf32>
    %view_251 = memref.view %arg243[%c0][] : memref<256xi8> to memref<64xf32>
    %view_252 = memref.view %arg242[%c0][] : memref<256xi8> to memref<64xf32>
    %view_253 = memref.view %arg241[%c0][] : memref<256xi8> to memref<64xf32>
    %view_254 = memref.view %arg240[%c0][] : memref<256xi8> to memref<64xf32>
    %view_255 = memref.view %arg239[%c0][] : memref<256xi8> to memref<64xf32>
    %view_256 = memref.view %arg238[%c0][] : memref<256xi8> to memref<64xf32>
    %view_257 = memref.view %arg237[%c0][] : memref<256xi8> to memref<64xf32>
    %view_258 = memref.view %arg236[%c0][] : memref<256xi8> to memref<64xf32>
    %view_259 = memref.view %arg235[%c0][] : memref<256xi8> to memref<64xf32>
    %view_260 = memref.view %arg234[%c0][] : memref<256xi8> to memref<64xf32>
    %view_261 = memref.view %arg233[%c0][] : memref<256xi8> to memref<64xf32>
    %view_262 = memref.view %arg232[%c0][] : memref<256xi8> to memref<64xf32>
    %view_263 = memref.view %arg231[%c0][] : memref<256xi8> to memref<64xf32>
    %view_264 = memref.view %arg230[%c0][] : memref<256xi8> to memref<64xf32>
    %view_265 = memref.view %arg229[%c0][] : memref<256xi8> to memref<64xf32>
    %view_266 = memref.view %arg228[%c0][] : memref<256xi8> to memref<64xf32>
    %view_267 = memref.view %arg227[%c0][] : memref<256xi8> to memref<64xf32>
    %view_268 = memref.view %arg1[%c0][] : memref<8192xi8> to memref<1024xi64>
    %view_269 = memref.view %arg361[%c789504][] : memref<122096960xi8> to memref<1024x64xf32>
    %view_270 = memref.view %arg361[%c265216][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.func.launch_13(%c0_i32, %c512_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_267, %view_266, %view_265, %view_264, %view_263, %view_262, %view_261, %view_260, %view_259, %view_258, %view_257, %view_256, %view_231, %view_1, %view_197, %view_230, %view_229, %view_165, %view_206, %view_227, %view_114, %view_226, %view_207, %view_96, %view_225, %view_224, %view_228, %view_268, %view_255, %view_254, %view_253, %view_252, %view_251, %view_250, %view_249, %view_248, %view_247, %view_246, %view_245, %view_244, %view_243, %view_242, %view_241, %view_240, %view_239, %view_238, %view_237, %view_236, %view_235, %view_234, %view_233, %view_232, %view_269, %view_7, %view_270) {kernel = "fusion_18", rt.trace = #rt.hlo_trace<"fusion.18">, stream = 0 : i64, uid = 23 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x96xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xi64>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<1024x64xf32>) -> ()
    %view_271 = memref.view %arg80[%c0][] : memref<12288xi8> to memref<64x48xf32>
    %view_272 = memref.view %arg361[%c1051648][] : memref<122096960xi8> to memref<1024x48xf32>
    call @xla.gpu.gemm_0(%view_269, %view_271, %view_272) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.65">, uid = 26 : i64} : (memref<1024x64xf32>, memref<64x48xf32>, memref<1024x48xf32>) -> ()
    %view_273 = memref.view %arg226[%c0][] : memref<256xi8> to memref<64xf32>
    %view_274 = memref.view %arg225[%c0][] : memref<256xi8> to memref<64xf32>
    %view_275 = memref.view %arg224[%c0][] : memref<256xi8> to memref<64xf32>
    %view_276 = memref.view %arg223[%c0][] : memref<256xi8> to memref<64xf32>
    %view_277 = memref.view %arg222[%c0][] : memref<256xi8> to memref<64xf32>
    %view_278 = memref.view %arg221[%c0][] : memref<256xi8> to memref<64xf32>
    %view_279 = memref.view %arg220[%c0][] : memref<256xi8> to memref<64xf32>
    %view_280 = memref.view %arg219[%c0][] : memref<256xi8> to memref<64xf32>
    %view_281 = memref.view %arg218[%c0][] : memref<256xi8> to memref<64xf32>
    %view_282 = memref.view %arg217[%c0][] : memref<256xi8> to memref<64xf32>
    %view_283 = memref.view %arg216[%c0][] : memref<256xi8> to memref<64xf32>
    %view_284 = memref.view %arg215[%c0][] : memref<256xi8> to memref<64xf32>
    %view_285 = memref.view %arg190[%c0][] : memref<256xi8> to memref<64xf32>
    %view_286 = memref.view %arg189[%c0][] : memref<256xi8> to memref<64xf32>
    %view_287 = memref.view %arg188[%c0][] : memref<256xi8> to memref<64xf32>
    %view_288 = memref.view %arg187[%c0][] : memref<256xi8> to memref<64xf32>
    %view_289 = memref.view %arg186[%c0][] : memref<256xi8> to memref<64xf32>
    %view_290 = memref.view %arg185[%c0][] : memref<256xi8> to memref<64xf32>
    %view_291 = memref.view %arg184[%c0][] : memref<256xi8> to memref<64xf32>
    %view_292 = memref.view %arg183[%c0][] : memref<256xi8> to memref<64xf32>
    %view_293 = memref.view %arg182[%c0][] : memref<256xi8> to memref<64xf32>
    %view_294 = memref.view %arg181[%c0][] : memref<256xi8> to memref<64xf32>
    %view_295 = memref.view %arg180[%c0][] : memref<256xi8> to memref<64xf32>
    %view_296 = memref.view %arg179[%c0][] : memref<256xi8> to memref<64xf32>
    %view_297 = memref.view %arg214[%c0][] : memref<256xi8> to memref<64xf32>
    %view_298 = memref.view %arg213[%c0][] : memref<256xi8> to memref<64xf32>
    %view_299 = memref.view %arg212[%c0][] : memref<256xi8> to memref<64xf32>
    %view_300 = memref.view %arg211[%c0][] : memref<256xi8> to memref<64xf32>
    %view_301 = memref.view %arg210[%c0][] : memref<256xi8> to memref<64xf32>
    %view_302 = memref.view %arg209[%c0][] : memref<256xi8> to memref<64xf32>
    %view_303 = memref.view %arg208[%c0][] : memref<256xi8> to memref<64xf32>
    %view_304 = memref.view %arg207[%c0][] : memref<256xi8> to memref<64xf32>
    %view_305 = memref.view %arg206[%c0][] : memref<256xi8> to memref<64xf32>
    %view_306 = memref.view %arg205[%c0][] : memref<256xi8> to memref<64xf32>
    %view_307 = memref.view %arg204[%c0][] : memref<256xi8> to memref<64xf32>
    %view_308 = memref.view %arg203[%c0][] : memref<256xi8> to memref<64xf32>
    %view_309 = memref.view %arg361[%c11144192][] : memref<122096960xi8> to memref<1024x64xf32>
    %view_310 = memref.view %arg361[%c11406336][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.func.launch_13(%c0_i32, %c512_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_296, %view_295, %view_294, %view_293, %view_292, %view_291, %view_290, %view_289, %view_288, %view_287, %view_286, %view_285, %view_231, %view_1, %view_197, %view_230, %view_229, %view_165, %view_206, %view_227, %view_114, %view_226, %view_207, %view_96, %view_225, %view_224, %view_228, %view_268, %view_308, %view_307, %view_306, %view_305, %view_304, %view_303, %view_302, %view_301, %view_300, %view_299, %view_298, %view_297, %view_284, %view_283, %view_282, %view_281, %view_280, %view_279, %view_278, %view_277, %view_276, %view_275, %view_274, %view_273, %view_269, %view_309, %view_310) {kernel = "fusion_18", rt.trace = #rt.hlo_trace<"fusion.55">, stream = 0 : i64, uid = 22 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x96xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xi64>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<1024x64xf32>) -> ()
    %view_311 = memref.view %arg81[%c0][] : memref<12288xi8> to memref<64x48xf32>
    %view_312 = memref.view %arg361[%c527360][] : memref<122096960xi8> to memref<1024x48xf32>
    call @xla.gpu.gemm_0(%view_269, %view_311, %view_312) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.41">, uid = 25 : i64} : (memref<1024x64xf32>, memref<64x48xf32>, memref<1024x48xf32>) -> ()
    %view_313 = memref.view %arg335[%c0][] : memref<192xi8> to memref<48xf32>
    %view_314 = memref.view %arg336[%c0][] : memref<192xi8> to memref<48xf32>
    %view_315 = memref.view %arg361[%c920576][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_11(%c0_i32, %c8_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view, %view_272, %view_313, %view_312, %view_314, %view_1, %view_315) {kernel = "fusion_218", rt.trace = #rt.hlo_trace<"fusion.218">, stream = 0 : i64, uid = 21 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x48xf32>, memref<48xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_316 = memref.view %arg337[%c0][] : memref<192xi8> to memref<48xf32>
    %view_317 = memref.view %arg334[%c0][] : memref<192xi8> to memref<48xf32>
    %view_318 = memref.view %arg333[%c0][] : memref<192xi8> to memref<48xf32>
    %view_319 = memref.view %arg332[%c0][] : memref<192xi8> to memref<48xf32>
    %view_320 = memref.view %arg331[%c0][] : memref<192xi8> to memref<48xf32>
    %view_321 = memref.view %arg330[%c0][] : memref<192xi8> to memref<48xf32>
    %view_322 = memref.view %arg329[%c0][] : memref<192xi8> to memref<48xf32>
    %view_323 = memref.view %arg328[%c0][] : memref<192xi8> to memref<48xf32>
    %view_324 = memref.view %arg327[%c0][] : memref<192xi8> to memref<48xf32>
    %view_325 = memref.view %arg326[%c0][] : memref<192xi8> to memref<48xf32>
    %view_326 = memref.view %arg325[%c0][] : memref<192xi8> to memref<48xf32>
    %view_327 = memref.view %arg324[%c0][] : memref<192xi8> to memref<48xf32>
    %view_328 = memref.view %arg338[%c0][] : memref<192xi8> to memref<48xf32>
    %view_329 = memref.view %arg351[%c0][] : memref<192xi8> to memref<48xf32>
    %view_330 = memref.view %arg350[%c0][] : memref<192xi8> to memref<48xf32>
    %view_331 = memref.view %arg349[%c0][] : memref<192xi8> to memref<48xf32>
    %view_332 = memref.view %arg348[%c0][] : memref<192xi8> to memref<48xf32>
    %view_333 = memref.view %arg347[%c0][] : memref<192xi8> to memref<48xf32>
    %view_334 = memref.view %arg346[%c0][] : memref<192xi8> to memref<48xf32>
    %view_335 = memref.view %arg345[%c0][] : memref<192xi8> to memref<48xf32>
    %view_336 = memref.view %arg344[%c0][] : memref<192xi8> to memref<48xf32>
    %view_337 = memref.view %arg343[%c0][] : memref<192xi8> to memref<48xf32>
    %view_338 = memref.view %arg342[%c0][] : memref<192xi8> to memref<48xf32>
    %view_339 = memref.view %arg341[%c0][] : memref<192xi8> to memref<48xf32>
    %view_340 = memref.view %arg340[%c0][] : memref<192xi8> to memref<48xf32>
    %view_341 = memref.view %arg339[%c0][] : memref<192xi8> to memref<48xf32>
    %view_342 = memref.view %arg361[%c11668480][] : memref<122096960xi8> to memref<1024x48xf32>
    %view_343 = memref.view %arg361[%c11865088][] : memref<122096960xi8> to memref<1024x48xf32>
    %view_344 = memref.view %arg361[%c723968][] : memref<122096960xi8> to memref<1024x48xf32>
    call @xla.gpu.func.launch_12(%c0_i32, %c384_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_327, %view_326, %view_325, %view_324, %view_323, %view_322, %view_321, %view_320, %view_319, %view_318, %view_317, %view_1, %view_272, %view_313, %view_316, %view_268, %view_328, %view_341, %view_340, %view_339, %view_338, %view_337, %view_336, %view_335, %view_334, %view_333, %view_332, %view_331, %view_330, %view_315, %view_312, %view_314, %view_329, %view_342, %view_343, %view_344) {kernel = "fusion_13", rt.trace = #rt.hlo_trace<"fusion.13">, stream = 0 : i64, uid = 20 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xi64>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<48xf32>, memref<1024x48xf32>, memref<1024x48xf32>, memref<1024x48xf32>) -> ()
    %view_345 = memref.view %arg83[%c0][] : memref<5184xi8> to memref<48x27xf32>
    %view_346 = memref.view %arg361[%c12454912][] : memref<122096960xi8> to memref<1024x27xf32>
    call @xla.gpu.gemm_5(%view_344, %view_345, %view_346) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.42">, uid = 24 : i64} : (memref<1024x48xf32>, memref<48x27xf32>, memref<1024x27xf32>) -> ()
    %view_347 = memref.view %arg79[%c0][] : memref<12288xi8> to memref<64x48xf32>
    %view_348 = memref.view %arg361[%c12061696][] : memref<122096960xi8> to memref<1024x48xf32>
    call @xla.gpu.gemm_0(%view_270, %view_347, %view_348) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.63">, uid = 23 : i64} : (memref<1024x64xf32>, memref<64x48xf32>, memref<1024x48xf32>) -> ()
    %view_349 = memref.view %arg78[%c0][] : memref<12288xi8> to memref<64x48xf32>
    %view_350 = memref.view %arg361[%c12258304][] : memref<122096960xi8> to memref<1024x48xf32>
    call @xla.gpu.gemm_0(%view_7, %view_349, %view_350) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.64">, uid = 22 : i64} : (memref<1024x64xf32>, memref<64x48xf32>, memref<1024x48xf32>) -> ()
    %view_351 = memref.view %arg308[%c0][] : memref<192xi8> to memref<48xf32>
    %view_352 = memref.view %arg307[%c0][] : memref<192xi8> to memref<48xf32>
    %view_353 = memref.view %arg361[%c12663808][] : memref<122096960xi8> to memref<1024xf32>
    %view_354 = memref.view %arg361[%c12667904][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_11(%c0_i32, %c8_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view, %view_350, %view_352, %view_348, %view_351, %view_353, %view_354) {kernel = "fusion_218", rt.trace = #rt.hlo_trace<"fusion.219">, stream = 0 : i64, uid = 19 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x48xf32>, memref<48xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_355 = memref.view %arg30[%c0][] : memref<229376xi8> to memref<896x64xf32>
    %view_356 = memref.view %arg15[%c0][] : memref<69730304xi8> to memref<19456x896xf32>
    %view_357 = memref.view %arg361[%c5245952][] : memref<122096960xi8> to memref<19456x64xf32>
    call @xla.gpu.gemm_4(%view_356, %view_355, %view_357) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.44">, uid = 21 : i64} : (memref<19456x896xf32>, memref<896x64xf32>, memref<19456x64xf32>) -> ()
    %view_358 = memref.view %arg13[%c0][] : memref<4718592xi8> to memref<1024x1152xf32>
    %view_359 = memref.view %arg361[%c10488832][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.func.launch_10(%c0_i32, %c512_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_358, %view_359) {kernel = "fusion_50", rt.trace = #rt.hlo_trace<"fusion.50">, stream = 0 : i64, uid = 18 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x1152xf32>, memref<1024x64xf32>) -> ()
    %view_360 = memref.view %arg74[%c0][] : memref<16384xi8> to memref<64x64xf32>
    %view_361 = memref.view %arg361[%c10226688][] : memref<122096960xi8> to memref<1024x64xf32>
    call @xla.gpu.gemm_3(%view_359, %view_360, %view_361) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.43">, uid = 20 : i64} : (memref<1024x64xf32>, memref<64x64xf32>, memref<1024x64xf32>) -> ()
    %view_362 = memref.view %arg171[%c0][] : memref<256xi8> to memref<64xf32>
    %view_363 = memref.view %arg172[%c0][] : memref<256xi8> to memref<64xf32>
    %view_364 = memref.view %arg361[%c3072][] : memref<122096960xi8> to memref<1024x8x160xf32>
    call @xla.gpu.func.launch_9(%c0_i32, %c2560_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_357, %view_362, %view_361, %view_363, %view_364) {kernel = "fusion_48", rt.trace = #rt.hlo_trace<"fusion.48">, stream = 0 : i64, uid = 17 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<19456x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x8x160xf32>) -> ()
    %view_365 = memref.view %arg36[%c0][] : memref<131072xi8> to memref<8192xf32>
    %view_366 = memref.view %arg361[%c12565504][] : memref<122096960xi8> to memref<8192xf32>
    call @xla.gpu.func.launch_8(%c0_i32, %c8192_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_364, %view_365, %view_366) {kernel = "fusion_46", rt.trace = #rt.hlo_trace<"fusion.46">, stream = 0 : i64, uid = 16 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x8x160xf32>, memref<8192xf32>, memref<8192xf32>) -> ()
    %view_367 = memref.view %arg126[%c0][] : memref<640xi8> to memref<160xf32>
    %view_368 = memref.view %arg127[%c0][] : memref<640xi8> to memref<160xf32>
    %view_369 = memref.view %arg361[%c7212032][] : memref<122096960xi8> to memref<1024x1x160xf32>
    %view_370 = memref.view %arg361[%c5901312][] : memref<122096960xi8> to memref<1024x1x160xf32>
    %view_371 = memref.view %arg361[%c5245952][] : memref<122096960xi8> to memref<1024x1x160xf32>
    %view_372 = memref.view %arg361[%c6556672][] : memref<122096960xi8> to memref<1024x1x160xf32>
    %view_373 = memref.view %arg361[%c7867392][] : memref<122096960xi8> to memref<1024x1x160xf32>
    %view_374 = memref.view %arg361[%c9178112][] : memref<122096960xi8> to memref<1024x1x160xf32>
    %view_375 = memref.view %arg361[%c9833472][] : memref<122096960xi8> to memref<1024x1x160xf32>
    %view_376 = memref.view %arg361[%c8522752][] : memref<122096960xi8> to memref<1024x1x160xf32>
    call @xla.gpu.func.launch_7(%c0_i32, %c10240_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_365, %view_366, %view_364, %view_367, %view_368, %view_369, %view_370, %view_371, %view_372, %view_373, %view_374, %view_375, %view_376) {kernel = "fusion_188", rt.trace = #rt.hlo_trace<"fusion.188">, stream = 0 : i64, uid = 15 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<8192xf32>, memref<8192xf32>, memref<1024x8x160xf32>, memref<160xf32>, memref<160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>) -> ()
    %view_377 = memref.view %arg53[%c0][] : memref<102400xi8> to memref<160x160xf32>
    %view_378 = memref.view %arg361[%c5245952][] : memref<122096960xi8> to memref<1024x160xf32>
    %view_379 = memref.view %arg361[%c10488832][] : memref<122096960xi8> to memref<1024x160xf32>
    call @xla.gpu.gemm_2(%view_378, %view_377, %view_379) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.45">, uid = 19 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_380 = memref.view %arg142[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_6(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_379, %view_380) {kernel = "fusion_44", rt.trace = #rt.hlo_trace<"fusion.44">, stream = 0 : i64, uid = 14 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) -> ()
    %view_381 = memref.view %arg54[%c0][] : memref<102400xi8> to memref<160x160xf32>
    call @xla.gpu.gemm_2(%view_379, %view_381, %view_378) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.46">, uid = 18 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_382 = memref.view %arg51[%c0][] : memref<102400xi8> to memref<160x160xf32>
    %view_383 = memref.view %arg361[%c6556672][] : memref<122096960xi8> to memref<1024x160xf32>
    call @xla.gpu.gemm_2(%view_383, %view_382, %view_379) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.47">, uid = 17 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_384 = memref.view %arg140[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_6(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_379, %view_384) {kernel = "fusion_44", rt.trace = #rt.hlo_trace<"fusion.43">, stream = 0 : i64, uid = 13 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) -> ()
    %view_385 = memref.view %arg52[%c0][] : memref<102400xi8> to memref<160x160xf32>
    call @xla.gpu.gemm_2(%view_379, %view_385, %view_383) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.48">, uid = 16 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_386 = memref.view %arg49[%c0][] : memref<102400xi8> to memref<160x160xf32>
    %view_387 = memref.view %arg361[%c7867392][] : memref<122096960xi8> to memref<1024x160xf32>
    call @xla.gpu.gemm_2(%view_387, %view_386, %view_379) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.49">, uid = 15 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_388 = memref.view %arg138[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_6(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_379, %view_388) {kernel = "fusion_44", rt.trace = #rt.hlo_trace<"fusion.42">, stream = 0 : i64, uid = 12 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) -> ()
    %view_389 = memref.view %arg50[%c0][] : memref<102400xi8> to memref<160x160xf32>
    call @xla.gpu.gemm_2(%view_379, %view_389, %view_387) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.50">, uid = 14 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_390 = memref.view %arg47[%c0][] : memref<102400xi8> to memref<160x160xf32>
    %view_391 = memref.view %arg361[%c9178112][] : memref<122096960xi8> to memref<1024x160xf32>
    call @xla.gpu.gemm_2(%view_391, %view_390, %view_379) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.51">, uid = 13 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_392 = memref.view %arg136[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_6(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_379, %view_392) {kernel = "fusion_44", rt.trace = #rt.hlo_trace<"fusion.41">, stream = 0 : i64, uid = 11 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) -> ()
    %view_393 = memref.view %arg48[%c0][] : memref<102400xi8> to memref<160x160xf32>
    call @xla.gpu.gemm_2(%view_379, %view_393, %view_391) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.52">, uid = 12 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_394 = memref.view %arg45[%c0][] : memref<102400xi8> to memref<160x160xf32>
    %view_395 = memref.view %arg361[%c9833472][] : memref<122096960xi8> to memref<1024x160xf32>
    call @xla.gpu.gemm_2(%view_395, %view_394, %view_379) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.53">, uid = 11 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_396 = memref.view %arg134[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_6(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_379, %view_396) {kernel = "fusion_44", rt.trace = #rt.hlo_trace<"fusion.40">, stream = 0 : i64, uid = 10 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) -> ()
    %view_397 = memref.view %arg46[%c0][] : memref<102400xi8> to memref<160x160xf32>
    call @xla.gpu.gemm_2(%view_379, %view_397, %view_395) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.54">, uid = 10 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_398 = memref.view %arg43[%c0][] : memref<102400xi8> to memref<160x160xf32>
    %view_399 = memref.view %arg361[%c8522752][] : memref<122096960xi8> to memref<1024x160xf32>
    call @xla.gpu.gemm_2(%view_399, %view_398, %view_379) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.55">, uid = 9 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_400 = memref.view %arg132[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_6(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_379, %view_400) {kernel = "fusion_44", rt.trace = #rt.hlo_trace<"fusion.39">, stream = 0 : i64, uid = 9 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) -> ()
    %view_401 = memref.view %arg44[%c0][] : memref<102400xi8> to memref<160x160xf32>
    call @xla.gpu.gemm_2(%view_379, %view_401, %view_399) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.56">, uid = 8 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_402 = memref.view %arg41[%c0][] : memref<102400xi8> to memref<160x160xf32>
    %view_403 = memref.view %arg361[%c7212032][] : memref<122096960xi8> to memref<1024x160xf32>
    call @xla.gpu.gemm_2(%view_403, %view_402, %view_379) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.57">, uid = 7 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_404 = memref.view %arg130[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_6(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_379, %view_404) {kernel = "fusion_44", rt.trace = #rt.hlo_trace<"fusion.38">, stream = 0 : i64, uid = 8 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) -> ()
    %view_405 = memref.view %arg42[%c0][] : memref<102400xi8> to memref<160x160xf32>
    call @xla.gpu.gemm_2(%view_379, %view_405, %view_403) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.58">, uid = 6 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_406 = memref.view %arg39[%c0][] : memref<102400xi8> to memref<160x160xf32>
    %view_407 = memref.view %arg361[%c5901312][] : memref<122096960xi8> to memref<1024x160xf32>
    call @xla.gpu.gemm_2(%view_407, %view_406, %view_379) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.59">, uid = 5 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_408 = memref.view %arg128[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_6(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_379, %view_408) {kernel = "fusion_44", rt.trace = #rt.hlo_trace<"fusion.37">, stream = 0 : i64, uid = 7 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) -> ()
    %view_409 = memref.view %arg40[%c0][] : memref<102400xi8> to memref<160x160xf32>
    call @xla.gpu.gemm_2(%view_379, %view_409, %view_407) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.60">, uid = 4 : i64} : (memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) -> ()
    %view_410 = memref.view %arg143[%c0][] : memref<640xi8> to memref<160xf32>
    %view_411 = memref.view %arg141[%c0][] : memref<640xi8> to memref<160xf32>
    %view_412 = memref.view %arg139[%c0][] : memref<640xi8> to memref<160xf32>
    %view_413 = memref.view %arg137[%c0][] : memref<640xi8> to memref<160xf32>
    %view_414 = memref.view %arg135[%c0][] : memref<640xi8> to memref<160xf32>
    %view_415 = memref.view %arg133[%c0][] : memref<640xi8> to memref<160xf32>
    %view_416 = memref.view %arg131[%c0][] : memref<640xi8> to memref<160xf32>
    %view_417 = memref.view %arg129[%c0][] : memref<640xi8> to memref<160xf32>
    %view_418 = memref.view %arg361[%c12598272][] : memref<122096960xi8> to memref<8192xf32>
    %view_419 = memref.view %arg361[%c12631040][] : memref<122096960xi8> to memref<8192xf32>
    call @xla.gpu.func.launch_5(%c0_i32, %c8192_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_365, %view_366, %view_364, %view_367, %view_368, %view_407, %view_417, %view_403, %view_416, %view_399, %view_415, %view_395, %view_414, %view_391, %view_413, %view_387, %view_412, %view_383, %view_411, %view_378, %view_410, %view_418, %view_419) {kernel = "fusion_34", rt.trace = #rt.hlo_trace<"fusion.34">, stream = 0 : i64, uid = 6 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<8192xf32>, memref<8192xf32>, memref<1024x8x160xf32>, memref<160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<8192xf32>, memref<8192xf32>) -> ()
    %view_420 = memref.view %arg144[%c0][] : memref<640xi8> to memref<160xf32>
    %view_421 = memref.view %arg145[%c0][] : memref<640xi8> to memref<160xf32>
    call @xla.gpu.func.launch_4(%c0_i32, %c1280_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_418, %view_419, %view_420, %view_421, %view_365, %view_366, %view_364, %view_367, %view_368, %view_407, %view_417, %view_403, %view_416, %view_399, %view_415, %view_395, %view_414, %view_391, %view_413, %view_387, %view_412, %view_383, %view_411, %view_378, %view_410, %view_379) {kernel = "fusion_33", rt.trace = #rt.hlo_trace<"fusion.33">, stream = 0 : i64, uid = 5 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<8192xf32>, memref<8192xf32>, memref<160xf32>, memref<160xf32>, memref<8192xf32>, memref<8192xf32>, memref<1024x8x160xf32>, memref<160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>) -> ()
    %view_422 = memref.view %arg66[%c0][] : memref<40960xi8> to memref<160x64xf32>
    call @xla.gpu.gemm_1(%view_379, %view_422, %view_7) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.61">, uid = 3 : i64} : (memref<1024x160xf32>, memref<160x64xf32>, memref<1024x64xf32>) -> ()
    %view_423 = memref.view %arg173[%c0][] : memref<256xi8> to memref<64xf32>
    call @xla.gpu.func.launch_3(%c0_i32, %c512_i32, %c1_i32, %c1_i32, %c64_i32, %c2_i32, %c1_i32, %view_7, %view_423) {kernel = "fusion_32", rt.trace = #rt.hlo_trace<"fusion.32">, stream = 0 : i64, uid = 4 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x64xf32>, memref<64xf32>) -> ()
    %view_424 = memref.view %arg174[%c0][] : memref<256xi8> to memref<64xf32>
    %view_425 = memref.view %arg175[%c0][] : memref<256xi8> to memref<64xf32>
    %view_426 = memref.view %arg176[%c0][] : memref<256xi8> to memref<64xf32>
    %view_427 = memref.view %arg177[%c0][] : memref<256xi8> to memref<64xf32>
    %view_428 = memref.view %arg178[%c0][] : memref<256xi8> to memref<64xf32>
    %view_429 = memref.view %arg361[%c12585984][] : memref<122096960xi8> to memref<1024xf32>
    %view_430 = memref.view %arg361[%c12565504][] : memref<122096960xi8> to memref<1024xf32>
    %view_431 = memref.view %arg361[%c12569600][] : memref<122096960xi8> to memref<1024xf32>
    %view_432 = memref.view %arg361[%c12573696][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_2(%c0_i32, %c1024_i32, %c1_i32, %c1_i32, %c32_i32, %c1_i32, %c1_i32, %view_7, %view_424, %view_425, %view_426, %view_427, %view_428, %view_1, %view_429, %view_430, %view_431, %view_432) {kernel = "fusion_26", rt.trace = #rt.hlo_trace<"fusion.26">, stream = 0 : i64, uid = 3 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_433 = memref.view %arg309[%c0][] : memref<192xi8> to memref<48xf32>
    %view_434 = memref.view %arg293[%c0][] : memref<192xi8> to memref<48xf32>
    %view_435 = memref.view %arg322[%c0][] : memref<192xi8> to memref<48xf32>
    %view_436 = memref.view %arg321[%c0][] : memref<192xi8> to memref<48xf32>
    %view_437 = memref.view %arg320[%c0][] : memref<192xi8> to memref<48xf32>
    %view_438 = memref.view %arg319[%c0][] : memref<192xi8> to memref<48xf32>
    %view_439 = memref.view %arg318[%c0][] : memref<192xi8> to memref<48xf32>
    %view_440 = memref.view %arg317[%c0][] : memref<192xi8> to memref<48xf32>
    %view_441 = memref.view %arg316[%c0][] : memref<192xi8> to memref<48xf32>
    %view_442 = memref.view %arg315[%c0][] : memref<192xi8> to memref<48xf32>
    %view_443 = memref.view %arg314[%c0][] : memref<192xi8> to memref<48xf32>
    %view_444 = memref.view %arg313[%c0][] : memref<192xi8> to memref<48xf32>
    %view_445 = memref.view %arg312[%c0][] : memref<192xi8> to memref<48xf32>
    %view_446 = memref.view %arg311[%c0][] : memref<192xi8> to memref<48xf32>
    %view_447 = memref.view %arg310[%c0][] : memref<192xi8> to memref<48xf32>
    %view_448 = memref.view %arg306[%c0][] : memref<192xi8> to memref<48xf32>
    %view_449 = memref.view %arg305[%c0][] : memref<192xi8> to memref<48xf32>
    %view_450 = memref.view %arg304[%c0][] : memref<192xi8> to memref<48xf32>
    %view_451 = memref.view %arg303[%c0][] : memref<192xi8> to memref<48xf32>
    %view_452 = memref.view %arg302[%c0][] : memref<192xi8> to memref<48xf32>
    %view_453 = memref.view %arg301[%c0][] : memref<192xi8> to memref<48xf32>
    %view_454 = memref.view %arg300[%c0][] : memref<192xi8> to memref<48xf32>
    %view_455 = memref.view %arg299[%c0][] : memref<192xi8> to memref<48xf32>
    %view_456 = memref.view %arg298[%c0][] : memref<192xi8> to memref<48xf32>
    %view_457 = memref.view %arg297[%c0][] : memref<192xi8> to memref<48xf32>
    %view_458 = memref.view %arg296[%c0][] : memref<192xi8> to memref<48xf32>
    %view_459 = memref.view %arg295[%c0][] : memref<192xi8> to memref<48xf32>
    %view_460 = memref.view %arg294[%c0][] : memref<192xi8> to memref<48xf32>
    %view_461 = memref.view %arg361[%c12577792][] : memref<122096960xi8> to memref<1024xf32>
    %view_462 = memref.view %arg361[%c12581888][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_1(%c0_i32, %c8_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_429, %view_434, %view_460, %view_459, %view_458, %view_457, %view_456, %view_455, %view_454, %view_453, %view_452, %view_451, %view_450, %view_449, %view_448, %view_268, %view_353, %view_350, %view_352, %view_1, %view_433, %view_447, %view_446, %view_445, %view_444, %view_443, %view_442, %view_441, %view_440, %view_439, %view_438, %view_437, %view_436, %view_354, %view_348, %view_351, %view_435, %view_461, %view_462) {kernel = "fusion_19", rt.trace = #rt.hlo_trace<"fusion.19">, stream = 0 : i64, uid = 2 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xi64>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_463 = memref.view %arg77[%c0][] : memref<12288xi8> to memref<64x48xf32>
    %view_464 = memref.view %arg361[%c265216][] : memref<122096960xi8> to memref<1024x48xf32>
    call @xla.gpu.gemm_0(%view_310, %view_463, %view_464) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.66">, uid = 2 : i64} : (memref<1024x64xf32>, memref<64x48xf32>, memref<1024x48xf32>) -> ()
    %view_465 = memref.view %arg76[%c0][] : memref<12288xi8> to memref<64x48xf32>
    %view_466 = memref.view %arg361[%c11406336][] : memref<122096960xi8> to memref<1024x48xf32>
    call @xla.gpu.gemm_0(%view_309, %view_465, %view_466) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.67">, uid = 1 : i64} : (memref<1024x64xf32>, memref<64x48xf32>, memref<1024x48xf32>) -> ()
    %view_467 = memref.view %arg277[%c0][] : memref<192xi8> to memref<48xf32>
    %view_468 = memref.view %arg276[%c0][] : memref<192xi8> to memref<48xf32>
    %view_469 = memref.view %arg361[%c11602944][] : memref<122096960xi8> to memref<1024xf32>
    %view_470 = memref.view %arg361[%c11607040][] : memref<122096960xi8> to memref<1024xf32>
    %view_471 = memref.view %arg361[%c11611136][] : memref<122096960xi8> to memref<1024xf32>
    %view_472 = memref.view %arg361[%c11615232][] : memref<122096960xi8> to memref<1024xf32>
    call @xla.gpu.func.launch_0(%c0_i32, %c8_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_466, %view_468, %view, %view_464, %view_467, %view_469, %view_470, %view_471, %view_472) {kernel = "fusion_217", rt.trace = #rt.hlo_trace<"fusion.217">, stream = 0 : i64, uid = 1 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<1024x48xf32>, memref<48xf32>, memref<f32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) -> ()
    %view_473 = memref.view %arg82[%c0][] : memref<6912xi8> to memref<64x27xf32>
    %view_474 = memref.view %arg361[%c12061696][] : memref<122096960xi8> to memref<1024x27xf32>
    call @xla.gpu.gemm(%view_7, %view_473, %view_474) {algorithm = -1 : i64, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision = dense<0> : tensor<2xi32>, rt.trace = #rt.hlo_trace<"custom-call.62">, uid = 0 : i64} : (memref<1024x64xf32>, memref<64x27xf32>, memref<1024x27xf32>) -> ()
    %view_475 = memref.view %arg323[%c0][] : memref<192xi8> to memref<48xf32>
    %view_476 = memref.view %arg279[%c0][] : memref<192xi8> to memref<48xf32>
    %view_477 = memref.view %arg263[%c0][] : memref<192xi8> to memref<48xf32>
    %view_478 = memref.view %arg352[%c0][] : memref<128xi8> to memref<32xf32>
    %view_479 = memref.view %arg292[%c0][] : memref<192xi8> to memref<48xf32>
    %view_480 = memref.view %arg291[%c0][] : memref<192xi8> to memref<48xf32>
    %view_481 = memref.view %arg290[%c0][] : memref<192xi8> to memref<48xf32>
    %view_482 = memref.view %arg289[%c0][] : memref<192xi8> to memref<48xf32>
    %view_483 = memref.view %arg288[%c0][] : memref<192xi8> to memref<48xf32>
    %view_484 = memref.view %arg287[%c0][] : memref<192xi8> to memref<48xf32>
    %view_485 = memref.view %arg286[%c0][] : memref<192xi8> to memref<48xf32>
    %view_486 = memref.view %arg285[%c0][] : memref<192xi8> to memref<48xf32>
    %view_487 = memref.view %arg284[%c0][] : memref<192xi8> to memref<48xf32>
    %view_488 = memref.view %arg283[%c0][] : memref<192xi8> to memref<48xf32>
    %view_489 = memref.view %arg282[%c0][] : memref<192xi8> to memref<48xf32>
    %view_490 = memref.view %arg281[%c0][] : memref<192xi8> to memref<48xf32>
    %view_491 = memref.view %arg280[%c0][] : memref<192xi8> to memref<48xf32>
    %view_492 = memref.view %arg356[%c0][] : memref<108xi8> to memref<27xf32>
    %view_493 = memref.view %arg278[%c0][] : memref<192xi8> to memref<48xf32>
    %view_494 = memref.view %arg275[%c0][] : memref<192xi8> to memref<48xf32>
    %view_495 = memref.view %arg274[%c0][] : memref<192xi8> to memref<48xf32>
    %view_496 = memref.view %arg273[%c0][] : memref<192xi8> to memref<48xf32>
    %view_497 = memref.view %arg272[%c0][] : memref<192xi8> to memref<48xf32>
    %view_498 = memref.view %arg271[%c0][] : memref<192xi8> to memref<48xf32>
    %view_499 = memref.view %arg270[%c0][] : memref<192xi8> to memref<48xf32>
    %view_500 = memref.view %arg269[%c0][] : memref<192xi8> to memref<48xf32>
    %view_501 = memref.view %arg268[%c0][] : memref<192xi8> to memref<48xf32>
    %view_502 = memref.view %arg267[%c0][] : memref<192xi8> to memref<48xf32>
    %view_503 = memref.view %arg266[%c0][] : memref<192xi8> to memref<48xf32>
    %view_504 = memref.view %arg265[%c0][] : memref<192xi8> to memref<48xf32>
    %view_505 = memref.view %arg264[%c0][] : memref<192xi8> to memref<48xf32>
    %view_506 = memref.view %arg355[%c0][] : memref<108xi8> to memref<27xf32>
    %view_507 = memref.view %arg353[%c0][] : memref<128xi8> to memref<32xf32>
    %view_508 = memref.view %arg12[%c0][] : memref<131072xi8> to memref<1024x32xf32>
    %view_509 = memref.view %arg36[%c0][] : memref<131072xi8> to memref<1024x32xf32>
    call @xla.gpu.func.launch(%c0_i32, %c256_i32, %c1_i32, %c1_i32, %c128_i32, %c1_i32, %c1_i32, %view_478, %view_507, %view_165, %view_508, %view_432, %view_477, %view_505, %view_504, %view_503, %view_502, %view_501, %view_500, %view_499, %view_498, %view_497, %view_496, %view_495, %view_494, %view_469, %view_493, %view_431, %view_476, %view_491, %view_490, %view_489, %view_488, %view_487, %view_486, %view_485, %view_484, %view_483, %view_482, %view_481, %view_480, %view_479, %view_461, %view_462, %view_471, %view_430, %view_475, %view_342, %view_343, %view_268, %view_474, %view_506, %view_346, %view_492, %view_470, %view_466, %view_468, %view_472, %view_464, %view_467, %view_509) {kernel = "fusion", rt.trace = #rt.hlo_trace<"fusion">, stream = 0 : i64, uid = 0 : i64} : (i32, i32, i32, i32, i32, i32, i32, memref<32xf32>, memref<32xf32>, memref<1024x96xf32>, memref<1024x32xf32>, memref<1024xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<48xf32>, memref<1024xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<48xf32>, memref<1024x48xf32>, memref<1024x48xf32>, memref<1024xi64>, memref<1024x27xf32>, memref<27xf32>, memref<1024x27xf32>, memref<27xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024x32xf32>) -> ()
    return
  }
  func.func private @xla.gpu.gemm(memref<1024x64xf32>, memref<64x27xf32>, memref<1024x27xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_0(memref<1024x64xf32>, memref<64x48xf32>, memref<1024x48xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_1(memref<1024x160xf32>, memref<160x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_2(memref<1024x160xf32>, memref<160x160xf32>, memref<1024x160xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_3(memref<1024x64xf32>, memref<64x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_4(memref<19456x896xf32>, memref<896x64xf32>, memref<19456x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_5(memref<1024x48xf32>, memref<48x27xf32>, memref<1024x27xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_6(memref<1024x76xf32>, memref<76x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_7(memref<1024x64xf32>, memref<64x4xf32>, memref<1024x4xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_8(memref<18432x64xf32>, memref<64x4xf32>, memref<18432x4xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_9(memref<1024x128xf32>, memref<128x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_10(memref<1024x128xf32>, memref<128x128xf32>, memref<1024x128xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_11(memref<1024x172xf32>, memref<172x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_12(memref<1024x43x344xf32>, memref<43x344x172xf32>, memref<43x1024x172xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_13(memref<1024x43x172xf32>, memref<43x172x344xf32>, memref<43x1024x344xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_14(memref<44032x64xf32>, memref<64x172xf32>, memref<44032x172xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_15(memref<1024x768xf32>, memref<768x96xf32>, memref<1024x96xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_16(memref<1024x256xf32>, memref<256x256xf32>, memref<1024x256xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_17(memref<262144x32xf32>, memref<32x3xf32>, memref<262144x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 262144)>>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_18(memref<262144x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 262144)>>, memref<3x32xf32>, memref<262144x32xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_19(memref<1024x768xf32>, memref<768x256xf32>, memref<1024x256xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_20(memref<1024x512xf32>, memref<512x256xf32>, memref<1024x256xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_21(memref<1024x768xf32>, memref<768x512xf32>, memref<1024x512xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_22(memref<1024x64xf32>, memref<64x768xf32>, memref<1024x768xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_23(memref<1024x768xf32>, memref<768x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_24(memref<1024x128xf32>, memref<128x256xf32>, memref<1024x256xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_25(memref<1024x256xf32>, memref<256x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_26(memref<1024x384xf32>, memref<384x128xf32>, memref<1024x128xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_27(memref<1024x512xf32>, memref<512x128xf32>, memref<1024x128xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_28(memref<1024x128xf32>, memref<128x512xf32>, memref<1024x512xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_29(memref<1024x384xf32>, memref<384x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_30(memref<1024x64xf32>, memref<64x128xf32>, memref<1024x128xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_31(memref<1024x384xf32>, memref<384x512xf32>, memref<1024x512xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_32(memref<1024x64xf32>, memref<64x384xf32>, memref<1024x384xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_33(memref<3072x128xf32>, memref<128x128xf32>, memref<3072x128xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_34(memref<131072x128xf32>, memref<128x3xf32>, memref<131072x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 131072)>>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.gemm_35(memref<131072x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 131072)>>, memref<3x128xf32>, memref<131072x128xf32>) attributes {rt.custom_call = "xla.gpu.gemm"}
  func.func private @xla.gpu.func.launch(i32, i32, i32, i32, i32, i32, i32, memref<32xf32>, memref<32xf32>, memref<1024x96xf32>, memref<1024x32xf32>, memref<1024xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<48xf32>, memref<1024xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<48xf32>, memref<1024x48xf32>, memref<1024x48xf32>, memref<1024xi64>, memref<1024x27xf32>, memref<27xf32>, memref<1024x27xf32>, memref<27xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024x32xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_0(i32, i32, i32, i32, i32, i32, i32, memref<1024x48xf32>, memref<48xf32>, memref<f32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_1(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xi64>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_2(i32, i32, i32, i32, i32, i32, i32, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_3(i32, i32, i32, i32, i32, i32, i32, memref<1024x64xf32>, memref<64xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_4(i32, i32, i32, i32, i32, i32, i32, memref<8192xf32>, memref<8192xf32>, memref<160xf32>, memref<160xf32>, memref<8192xf32>, memref<8192xf32>, memref<1024x8x160xf32>, memref<160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_5(i32, i32, i32, i32, i32, i32, i32, memref<8192xf32>, memref<8192xf32>, memref<1024x8x160xf32>, memref<160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<1024x160xf32>, memref<160xf32>, memref<8192xf32>, memref<8192xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_6(i32, i32, i32, i32, i32, i32, i32, memref<1024x160xf32>, memref<160xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_7(i32, i32, i32, i32, i32, i32, i32, memref<8192xf32>, memref<8192xf32>, memref<1024x8x160xf32>, memref<160xf32>, memref<160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>, memref<1024x1x160xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_8(i32, i32, i32, i32, i32, i32, i32, memref<1024x8x160xf32>, memref<8192xf32>, memref<8192xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_9(i32, i32, i32, i32, i32, i32, i32, memref<19456x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x8x160xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_10(i32, i32, i32, i32, i32, i32, i32, memref<1024x1152xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_11(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x48xf32>, memref<48xf32>, memref<1024x48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_12(i32, i32, i32, i32, i32, i32, i32, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xi64>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<48xf32>, memref<1024xf32>, memref<1024x48xf32>, memref<48xf32>, memref<48xf32>, memref<1024x48xf32>, memref<1024x48xf32>, memref<1024x48xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_13(i32, i32, i32, i32, i32, i32, i32, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x96xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xi64>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_14(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x96xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_15(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<1024x96xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_16(i32, i32, i32, i32, i32, i32, i32, memref<76xf32>, memref<76xf32>, memref<1024xf32>, memref<18432x4xf32>, memref<4xf32>, memref<1024x4xf32>, memref<4xf32>, memref<1024x76xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_17(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<18432x4xf32>, memref<4xf32>, memref<1024x4xf32>, memref<4xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_18(i32, i32, i32, i32, i32, i32, i32, memref<1024x20xf32>, memref<1024x20x256xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_19(i32, i32, i32, i32, i32, i32, i32, memref<1024x20x256xf32>, memref<1024x1152xf32>, memref<1024x20xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_20(i32, i32, i32, i32, i32, i32, i32, memref<1024x64xf32>, memref<1024x1152xf32>, memref<1024x18x64xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_21(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_22(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_23(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_24(i32, i32, i32, i32, i32, i32, i32, memref<1024x258xf32>, memref<1024x128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_25(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_26(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_27(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x64xf32>, memref<64xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_28(i32, i32, i32, i32, i32, i32, i32, memref<43x1024x172xf32>, memref<43x172xf32>, memref<1024x172xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_29(i32, i32, i32, i32, i32, i32, i32, memref<43x1024x344xf32>, memref<43x344xf32>, memref<1024x43x344xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_30(i32, i32, i32, i32, i32, i32, i32, memref<44032xf32>, memref<44032xf32>, memref<172xf32>, memref<172xf32>, memref<1024x43x172xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_31(i32, i32, i32, i32, i32, i32, i32, memref<1024x43x172xf32>, memref<44032xf32>, memref<44032xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_32(i32, i32, i32, i32, i32, i32, i32, memref<43x1024x172xf32>, memref<43x172xf32>, memref<1024x43x172xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_33(i32, i32, i32, i32, i32, i32, i32, memref<44032x172xf32>, memref<172xf32>, memref<1024x43x172xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_34(i32, i32, i32, i32, i32, i32, i32, memref<1024x2304xf32>, memref<1024x448xf32>, memref<1024x2752xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_35(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<768xf32>, memref<768xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x768xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_36(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_37(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x256xf32>, memref<256xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_38(i32, i32, i32, i32, i32, i32, i32, memref<3x1024x256xf32>, memref<3x262144xf32>, memref<1024x256xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_39(i32, i32, i32, i32, i32, i32, i32, memref<262144x32xf32>, memref<32xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_40(i32, i32, i32, i32, i32, i32, i32, memref<1024x768xf32>, memref<1024xf32>, memref<768xf32>, memref<768xf32>, memref<1024x768xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_41(i32, i32, i32, i32, i32, i32, i32, memref<1024x768xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_42(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<512xf32>, memref<512xf32>, memref<1024xf32>, memref<1024x512xf32>, memref<512xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_43(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x512xf32>, memref<512xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_44(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x512xf32>, memref<512xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_45(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<768xf32>, memref<768xf32>, memref<1024x768xf32>, memref<1024xf32>, memref<1024x768xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_46(i32, i32, i32, i32, i32, i32, i32, memref<1024x768xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_47(i32, i32, i32, i32, i32, i32, i32, memref<1024x768xf32>, memref<768xf32>, memref<3x1024x256xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x768xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_48(i32, i32, i32, i32, i32, i32, i32, memref<1024x256xf32>, memref<f32>, memref<1024x256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_49(i32, i32, i32, i32, i32, i32, i32, memref<3x1024x256xf32>, memref<1024x3x256xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_50(i32, i32, i32, i32, i32, i32, i32, memref<1024x256xf32>, memref<1024x256xf32>, memref<1024x256xf32>, memref<256xf32>, memref<3x1024x256xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_51(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<256xf32>, memref<256xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x256xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_52(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_53(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_54(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_55(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x384xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_56(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_57(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_58(i32, i32, i32, i32, i32, i32, i32, memref<1024x128xf32>, memref<128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_59(i32, i32, i32, i32, i32, i32, i32, memref<1024x384xf32>, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<1024x384xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_60(i32, i32, i32, i32, i32, i32, i32, memref<1024x384xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_61(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<1024x384xf32>, memref<1024xf32>, memref<1024x384xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_62(i32, i32, i32, i32, i32, i32, i32, memref<1024x384xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_63(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<3x1024x128xf32>, memref<1024xf32>, memref<1024x3x128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_64(i32, i32, i32, i32, i32, i32, i32, memref<3x1024x128xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_65(i32, i32, i32, i32, i32, i32, i32, memref<3072x128xf32>, memref<128xf32>, memref<3x1024x128xf32>, memref<3xf32>, memref<3x131072xf32>, memref<1024x384xf32>, memref<384xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024x384xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_66(i32, i32, i32, i32, i32, i32, i32, memref<1024x128xf32>, memref<f32>, memref<128xf32>, memref<1024x256xf32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_67(i32, i32, i32, i32, i32, i32, i32, memref<3x1024x128xf32>, memref<1024x3x128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_68(i32, i32, i32, i32, i32, i32, i32, memref<3072x128xf32>, memref<128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_69(i32, i32, i32, i32, i32, i32, i32, memref<3x1024x128xf32>, memref<3xf32>, memref<3x131072xf32>, memref<1024x3x128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_70(i32, i32, i32, i32, i32, i32, i32, memref<131072x128xf32>, memref<128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_71(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<384xf32>, memref<384xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<128xf32>, memref<1024x256xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x130xf32>, memref<1024xf32>, memref<1024x258xf32>, memref<128xf32>, memref<1024x128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024x384xf32>, memref<3x1024x128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_72(i32, i32, i32, i32, i32, i32, i32, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, memref<128xf32>, memref<1024x256xf32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x130xf32>, memref<1024xf32>, memref<1024x258xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_73(i32, i32, i32, i32, i32, i32, i32, memref<1024x258xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_74(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<128xf32>, memref<1024x256xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_75(i32, i32, i32, i32, i32, i32, i32, memref<128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024xf32>, memref<128xf32>, memref<128xf32>, memref<1024x128xf32>, memref<1024x128xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_76(i32, i32, i32, i32, i32, i32, i32, memref<1024x128xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_77(i32, i32, i32, i32, i32, i32, i32, memref<f32>, memref<1024x128xf32>, memref<128xf32>, memref<1024xf32>, memref<1024x128xf32>, memref<1024xf32>, memref<1024x130xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_78(i32, i32, i32, i32, i32, i32, i32, memref<1024x128xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_79(i32, i32, i32, i32, i32, i32, i32, memref<1024x130xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_80(i32, i32, i32, i32, i32, i32, i32, memref<64xf32>, memref<1024xf32>, memref<1024x64xf32>, memref<1024xf32>, memref<64xf32>, memref<64xf32>, memref<1024xf32>, memref<1024x64xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
  func.func private @xla.gpu.func.launch_81(i32, i32, i32, i32, i32, i32, i32, memref<1024x64xf32>, memref<f32>, memref<1024xf32>, memref<1024xf32>) attributes {rt.custom_call = "xla.gpu.func.launch"}
}