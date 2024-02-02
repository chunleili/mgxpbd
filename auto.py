from subprocess import call
call(["cmake", "--preset", "vs2022"])
call(["cmake", "--build", "--preset", "vs2022-Rel"])
call(["./build/Release/main.exe", ])