@echo off

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64
set path=w:\graphs;"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin";%path%
doskey ci=git commit -a -m

