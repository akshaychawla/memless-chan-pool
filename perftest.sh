
## Performance testing
# PRCNv1

echo "TESTING FOR PRCNv1"
# height/width variation
python perftest.py --module=PRCNv1 --batch_size=16 --ip_chans=16 --op_chans=64 --h=16 --w=16 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv1 --batch_size=16 --ip_chans=16 --op_chans=64 --h=32 --w=32 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv1 --batch_size=16 --ip_chans=16 --op_chans=64 --h=64 --w=64 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv1 --batch_size=16 --ip_chans=16 --op_chans=64 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 
# ip/op channels variations 
python perftest.py --module=PRCNv1 --batch_size=16 --ip_chans=16 --op_chans=32 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv1 --batch_size=16 --ip_chans=16 --op_chans=64 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv1 --batch_size=16 --ip_chans=32 --op_chans=64 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv1 --batch_size=16 --ip_chans=32 --op_chans=128 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 


# PRCNv2

echo "TESTING FOR PRCNv2"
# height/width variation
python perftest.py --module=PRCNv2 --batch_size=16 --ip_chans=16 --op_chans=64 --h=16 --w=16 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv2 --batch_size=16 --ip_chans=16 --op_chans=64 --h=32 --w=32 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv2 --batch_size=16 --ip_chans=16 --op_chans=64 --h=64 --w=64 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv2 --batch_size=16 --ip_chans=16 --op_chans=64 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 
# ip/op channels variations 
python perftest.py --module=PRCNv2 --batch_size=16 --ip_chans=16 --op_chans=32 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv2 --batch_size=16 --ip_chans=16 --op_chans=64 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv2 --batch_size=16 --ip_chans=32 --op_chans=64 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 
python perftest.py --module=PRCNv2 --batch_size=16 --ip_chans=32 --op_chans=128 --h=128 --w=128 --exp=3 --G=4 --numsamples=50 


