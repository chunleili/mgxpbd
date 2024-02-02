# multigrid xpbd

## Build
System: Windows 10

Requirements: 
- CMake >=3.24
- visual studio 2022
- vcpkg

Build steps:
1. Change the vcpkg path in CMakePresets.json "configurePresets" to your own path.
   For example, if your vcpkg is in `C:/Dev/vcpkg`
   `"toolchainFile": "C:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake"`. Or if your vcpkg is in `E:/codes/vcpkg`
   `"toolchainFile": "E:/codes/vcpkg/scripts/buildsystems/vcpkg.cmake"`

2. Config
    ```
    cmake --preset=vs2022
    ```
    The dependencies will be installed automatically as listed in vcpkg.json. They are installed in `.\build\vcpkg_installed\x64-windows`.

3. Build
    ```
    cmake --build --preset=vs2022-Rel
    ```

## Run
```
./build/Release/main.exe
```

Output files locate in results folder.


## Extra

Variables, functions, folder names, and file names should all be written in **lowercase_with_underscores**, while classes should be named using PascalCase.

For larger models(>100MB) that cannot be uploaded to GitHub, please place them under the "large_models" directory to avoid polluting the Git repository.

Run `python auto.py` can directly compile and run the program and generate the results.

Copy `extern\eigen\debug\msvc\eigen.natvis` to `C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\Packages\Debugger\Visualizers`(or `%USERPROFILE%\Documents\Visual Studio 2022\Visualizers`) can make you inspect Eigen matrices in Visual Studio debugger(VSCode is not available yet).