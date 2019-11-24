# 217project

#How to run
nvcc parboil.o file.o args.o main.cu -o zclgpu

./zclgpu -i datasets/small/input/32_32_32_dataset.bin -o zcloutput.bin

gcc -o compare.o compareFiles.cc

./compare.o zcloutput.bin datasets/small/output/32_32_32_dataset.out

