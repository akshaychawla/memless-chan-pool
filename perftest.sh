
## Performance testing
# PRCNPTN

echo "TESTING FOR PRCNPTN"
# height/width variation
python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=24 --op_chans=48 --h=16 --w=16 --CMP=4 --G=12 --numsamples=50 --xlsx="./performance.xlsx"
python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=24 --op_chans=48 --h=32 --w=32 --CMP=4 --G=12 --numsamples=50 --xlsx="./performance.xlsx"
python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=24 --op_chans=48 --h=64 --w=64 --CMP=4 --G=12 --numsamples=50 --xlsx="./performance.xlsx"
python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=24 --op_chans=48 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 --xlsx="./performance.xlsx"
# python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=16 --op_chans=64 --h=32 --w=32 --CMP=4 --G=12 --numsamples=50 
# python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=16 --op_chans=64 --h=64 --w=64 --CMP=4 --G=12 --numsamples=50 
# python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=16 --op_chans=64 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 
# # ip/op channels variations 
# python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=16 --op_chans=32 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 
# python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=16 --op_chans=64 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 
# python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=32 --op_chans=64 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 
# python perftest.py --module=PRCNPTN --batch_size=16 --ip_chans=32 --op_chans=128 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 
# 
# 
# # FastPRCNPTN
# 
# height/width variation
echo "TESTING FOR FastPRCNPTN"
python perftest.py --module=FastPRCNPTN --batch_size=16 --ip_chans=24 --op_chans=48 --h=16 --w=16 --CMP=4 --G=12 --numsamples=50 --xlsx="./performance.xlsx"
python perftest.py --module=FastPRCNPTN --batch_size=16 --ip_chans=24 --op_chans=48 --h=32 --w=32 --CMP=4 --G=12 --numsamples=50 --xlsx="./performance.xlsx"
python perftest.py --module=FastPRCNPTN --batch_size=16 --ip_chans=24 --op_chans=48 --h=64 --w=64 --CMP=4 --G=12 --numsamples=50 --xlsx="./performance.xlsx"
python perftest.py --module=FastPRCNPTN --batch_size=16 --ip_chans=24 --op_chans=48 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 --xlsx="./performance.xlsx"
# # ip/op channels variations 
# python perftest.py --module=FastPRCNPTN --batch_size=16 --ip_chans=16 --op_chans=32 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 
# python perftest.py --module=FastPRCNPTN --batch_size=16 --ip_chans=16 --op_chans=64 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 
# python perftest.py --module=FastPRCNPTN --batch_size=16 --ip_chans=32 --op_chans=64 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 
# python perftest.py --module=FastPRCNPTN --batch_size=16 --ip_chans=32 --op_chans=128 --h=128 --w=128 --CMP=4 --G=12 --numsamples=50 


