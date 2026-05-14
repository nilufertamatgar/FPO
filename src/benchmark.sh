


export PYTHONPATH=$PWD:$PYTHONPATH

python /dapustor/nilufer/ADFLIP/test/benchmark.py --ckpt_path ADFLIP_v1.pt --pdb_type pMT --sample_scheme adaptive --thershold 0.9 --argmax_final 1 --temp 0.1 --noise 0.1 --num_step 8                                             
                                                                                                                                                                                                            
# python /dapustor/nilufer/ADFLIP/test/benchmark.py --ckpt_path ADFLIP_v1.pt --pdb_type nucleotide --sample_scheme adaptive --thershold 0.9 --argmax_final 1 --temp 0.1 --noise 0.1 --num_step 8

# python /dapustor/nilufer/ADFLIP/test/benchmark.py --ckpt_path ADFLIP_v1.pt --pdb_type molecule --sample_scheme adaptive --thershold 0.9 --argmax_final 1 --temp 0.1 --noise 0.1 --num_step 8
    