# AMGX library setup

This is a compiled version of the [AMGX](https://github.com/NVIDIA/AMGX) library. System is Windows 10, visual studio 2022. 

See [AMGX API reference](https://github.com/NVIDIA/AMGX/blob/main/doc/AMGX_Reference.pdf) for API usage.

## Directory structure
- `include` contains the AMGX header files
- `lib` contains the AMGX library files
  - configs: contains the json configuration files
  - amgx.lib: static library
  - amgxsh.lib and amgxsh.dll: shared library 
  - matrix.mtx: a sample matrix file
- `amgx_capi.c` and `amgx_capi.h`: an example case

Note: Because **amgx.lib** and **amgxsh.dll** is too big(>50MB), which cannot be uploaded to github, you can download it from the [release page](https://github.com/chunleili/mgxpbd/releases/tag/AMGX_lib_files) of this repository. Just download the lib.zip  and extract it to the `extern/amgx/lib` directory.


## Sample Usage

Turn the **USE_AMGX** option on in your **CMakeCache.txt**.

### amgx_capi
Full case to **solve Ax=b**.

Compile the amgx_capi target.

Run the sample case:

In `C:\Dev\mgxpbd\build\extern\amgx\Release`

```
.\amgx_capi.exe -m .\matrix.mtx -c .\configs\CLASSICAL_CG_CYCLE.json
```

This will print the result information of the solver and output a file named `output.system.mtx` in the same directory. `output.system.mtx` lists the A, b and x.

### convert
Convert **matrix market format** to **binary** for faster IO.

In `C:\Dev\mgxpbd\build\extern\amgx\Release`

```
.\convert.exe matrix.mtx
```

This will output a file named `matrix.mtx.bin` in the same directory.
