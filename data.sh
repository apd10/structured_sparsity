for w in 128 256 512 1024 2048 4096 8192 16384;
do
	python3 script1.py --M $w --N $w --K $w &> simple.$w.log
done


for K in 10240;
do
	for M in 2048 4096 8192 10240 16384 20480
	do
	python3 script1.py --M $M --N $M --K $K &> M_N.$M.$K.log
done
done
