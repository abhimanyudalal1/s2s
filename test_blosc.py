import numcodecs

with open('/Users/abhimanyu/Desktop/s2s/IMD_rainfall_0p25.zarr/time/1', 'rb') as f:
    data = f.read()

try:
    compressor = numcodecs.Blosc()
    decompressed = compressor.decode(data)
    print("Time chunk 1 decompressed size:", len(decompressed))
except Exception as e:
    print("Not blosc:", e)
