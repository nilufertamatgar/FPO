


export PYTHONPATH=$PWD:$PYTHONPATH

python /dapustor/nilufer/ADFLIP/INFERENCE/benchmark_1.py --ckpt_path beta0.2alpha1_lr0.001_ADFLIP_yyh_best.pt --pdb_type test_set_109 --sample_scheme adaptive --thershold 0.9 --argmax_final 1 --temp 0.1 --noise 0.1 --num_step 8                                             
                                                                                                                                                                                                            
# python /dapustor/nilufer/ADFLIP/INFERENCE/benchmark.py --ckpt_path best_ce.pt --pdb_type nucleotide --sample_scheme adaptive --thershold 0.9 --argmax_final 1 --temp 0.1 --noise 0.1 --num_step 8

# python /dapustor/nilufer/ADFLIP/INFERENCE/benchmark.py --ckpt_path best_ce.pt --pdb_type molecule --sample_scheme adaptive --thershold 0.9 --argmax_final 1 --temp 0.1 --noise 0.1 --num_step 8
    