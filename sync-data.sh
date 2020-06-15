
if [ $# -lt 1 ]; then
	echo 'no folder is given'
	return 1
fi
FOLDER=$1
END=17

 for i in $(eval echo {03..$END}); do
	if [ $i -lt 10 ]; then
		remote=amazon0$i
	else
		remote=amazon$i
	fi
	ssh $remote "mkdir Code/nomad/$FOLDER"
	scp $FOLDER/* $remote:Code/nomad/$FOLDER/
done

