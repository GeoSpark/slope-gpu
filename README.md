# Slope GPU

Calculates and outputs the slope of a given DEM file using the GPU.

It can be used as a starting point for other geospatial processing tools that would benefit from
the speed of a GPU.

```
Usage: main.py [OPTIONS] INPUT_FILE_PATH OUTPUT_FILE_PATH

Options:
  --input-band INTEGER          Band to read from input file. Default: 1
  --overwrite / --no-overwrite  Overwrite output file if it already exists.
                                Default: --no-overwrite
  --help                        Show this message and exit.
```

If running on a multiple-GPU machine, you may need to specify which GPU to use as environment variables.
This will probably be different for different setups, but on an Intel/NVIDIA system something like this
should work:
```
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```
