# multigrid xpbd

## Build
System: Windows 10

Requirements: 
- CMake >=3.24
- visual studio 2022
- vcpkg

Build steps:
1. Change the vcpkg path in CMakePresets.json to your own path.
   ```
   "toolchainFile": "C:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake"
   ```
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

为了保持代码风格与libigl一致，一律使用小写下划线命名变量、函数、文件夹名和文件名，用大驼峰命名类。

较大模型github无法上传, 请放在large_models下面，不要污染了git仓库。

proj_dir_path 是全局变量，获取到的是项目的根目录，它依赖于main.cpp的位置，因此不要随意移动main.cpp的位置。
