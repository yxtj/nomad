1:

sh ./gen.sh 40000 2000 100

./bin/release/nomad_double -l 0.003 -d 0.04 -r 0.1 --nthreads=2 --path=Data/t2/ --pause=0 --timeouts 5 10

TIMEOUT=""
for i in {1..10}; do
	TIMEOUT+=" $(($i*180))"
done
mpirun -np 16 -hostfile ~/mpihosts nomad_double2 -l 0.003 -d 0.04 -r 0.1 --nthreads=1  --pause=0 --path=Data/t2/ --timeout $TIMEOUT > log


