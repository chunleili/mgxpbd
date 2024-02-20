AMGX library setup

This is a compiled version of the AMGX library. 

Directory structure:
- `include` contains the AMGX header files
- `lib` contains the AMGX library files
  - configs: contains the json configuration files
  - amgx.lib: static library
  - amgxsh.lib and amgxsh.dll: shared library 
  - matrix.mtx: a sample matrix file
- `amgx_capi.c` and `amgx_capi.h`: an example case

Note: Because **amgx.lib** and **amgxsh.dll** is too big(>50MB), which cannot be uploaded to github, you can download it from the [release page](https://github.com/chunleili/mgxpbd/releases/tag/AMGX_lib_files) of this repository. Just download the lib.zip  and extract it to the `extern/amgx/lib` directory.


Sample Usage:

Turn the AMGX option on of your CMAKE, and compile the amgx_capi target.

Run the sample case:

In `C:\Dev\mgxpbd\build\extern\amgx\Release`

```
.\amgx_capi.exe -m .\matrix.mtx -c .\configs\CLASSICAL_CG_CYCLE.json
```