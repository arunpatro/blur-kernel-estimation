mkdir brodatz
cd brodatz	
for i in `seq 1 112`;
do
	wget -b http://www.ux.uis.no/~tranden/brodatz/D$i.gif
done
rm wget-log*