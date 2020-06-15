#! /bin/sh

pre=$1
echo $pre
if [ -z $pre ] ; then
	echo 'no folder is given'
	return
fi
nrow=$2
ncol=$3
if [ $# -lt 3 ] || [ -z $nrow ] || [ -z $ncol ]; then
	echo 'size (nrow and ncol) of data is not given'
	return
fi
dim=20
if [ $# -ge 4 ]; then
	dim=$4
fi
seed=12345
if [ $# -ge 5 ]; then
	seed=$5
fi

echo 'generating internal files'
mkdir -p $pre
#echo $nrow > $pre/nrow
#echo $ncol > $pre/ncol
./bin/release/synthgen_conf --rowfile $pre/rowdegree.txt --colfile $pre/coldegree.txt --nrow $nrow --ncol $ncol --seed $seed
./bin/release/synthgen --rowfile $pre/rowdegree.txt --colfile $pre/coldegree.txt --output $pre/ --nrow $nrow --ncol $ncol --dim $dim --seed $seed

rm $pre/rowdegree.txt
rm $pre/coldegree.txt

nlist="header size ind val"
type="train test"

cnt_fail=0
for t in $type; do
	rm $pre/$t.dat -f
	echo "making $t.dat"
	for n in $nlist; do
		cat "$pre/$t""_$n.dat" >> $pre/$t.dat
		cnt_fail=$((cnt_fail + $?))
		# $? is the return of last command, 0 is success
	done
	if [ $cnt_fail -eq 0 ]; then
		echo '  success. remove internal data files.'
		for n in $nlist; do
			echo "removing $t""_$n.dat"
			rm "$pre/$t""_$n.dat"
		done
	else
		echo 'data generation failed.'
		return
	fi
done

echo 'success'
