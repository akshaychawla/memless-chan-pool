
# Depth 
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=40 --num_triplets=1 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160  
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=40 --num_triplets=2 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=40 --num_triplets=4 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=40 --num_triplets=5 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 

# Width 
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=20 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=60 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=80 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=120 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160

# Vary Growth rate 
python lenet.py --batchsize=64 --G=4 --CMP=4 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160  
python lenet.py --batchsize=64 --G=8 --CMP=4 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 
python lenet.py --batchsize=64 --G=16 --CMP=4 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 
python lenet.py --batchsize=64 --G=24 --CMP=4 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 

# CMP 
python lenet.py --batchsize=64 --G=12 --CMP=1 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
python lenet.py --batchsize=64 --G=12 --CMP=2 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
python lenet.py --batchsize=64 --G=12 --CMP=3 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160  
python lenet.py --batchsize=64 --G=12 --CMP=4 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160 
# python lenet.py --batchsize=64 --G=12 --CMP=5 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
# python lenet.py --batchsize=64 --G=12 --CMP=6 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
# python lenet.py --batchsize=64 --G=12 --CMP=8 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
# python lenet.py --batchsize=64 --G=12 --CMP=9 --num_hidden=40 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
# python lenet.py --batchsize=64 --G=12 --CMP=10 --num_hidden=60 --num_triplets=3 --datapoints=100 --outfile="timingresults.txt" --conv_channels=160
