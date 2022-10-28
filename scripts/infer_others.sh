export valid_can_path="/Data/candidate/candidate_combine_val_test/combine_valid_can_autosf_concat-complex_shallow-distmult_shallow-smore-autosf-complex_e-distmult_g-transe_l2-rule.npy"
export test_can_path="/Data/candidate/candidate_combine_val_test/combine_test_can_autosf_concat-complex_shallow-distmult_shallow-smore-autosf-complex_e-distmult_g-transe_l2-rule.npy"
export rst_prename="autosf_concat-complex_shallow-distmult_shallow-smore-autosf-complex_e-distmult_g-transe_l2-rule"
export root_save_path="/workspace/Data/embeddingVec"

echo "scripts/autosf_concat.sh"
python python/dglke/eval.py  --load_json \
autosf_concat/AutoSF_wikikg90m-v2_concat_d_768_g_50.0_lr_0.15_seed_0_0_mrr_0.20063574115435281_step_724999/config.json \
--valid_can_path ${valid_can_path} \
--test_can_path ${test_can_path} \
--rst_prename ${rst_prename} \
--root_save_path ${root_save_path}

echo "scripts/complex_shallow.sh"
python python/dglke/eval.py  --load_json \
complex_shallow/ComplEx_wikikg90m-v2_shallow_d_512_g_10.0_lr_0.1_seed_77_0_mrr_0.17148838564753532_step_849999/config.json \
--valid_can_path ${valid_can_path} \
--test_can_path ${test_can_path} \
--rst_prename ${rst_prename} \
--root_save_path ${root_save_path}

echo "scripts/distmult_shallow.sh"
python python/dglke/eval.py  --load_json \
distmult_shallow/DistMult_wikikg90m-v2_shallow_d_512_g_10.0_lr_0.1_seed_13_1_mrr_0.17049912363290787_step_2599999/config.json \
--valid_can_path ${valid_can_path} \
--test_can_path ${test_can_path} \
--rst_prename ${rst_prename} \
--root_save_path ${root_save_path}


echo "scripts/rotate_concat.sh"
python python/dglke/eval.py  --load_json \
rotate_concat/RotatE_wikikg90m-v2_concat_d_100_g_8.0_lr_0.1_seed_0_0_mrr_0.17069180309772491_step_1599999/config.json \
--valid_can_path ${valid_can_path} \
--test_can_path ${test_can_path} \
--rst_prename ${rst_prename} \
--root_save_path ${root_save_path}


echo "scripts/rotate_shallow.sh"
python python/dglke/eval.py  --load_json \
rotate_shallow/RotatE_wikikg90m-v2_shallow_d_256_g_12.0_lr_0.1_seed_0_3_mrr_0.2024185173213482_step_7849999/config.json \
--valid_can_path ${valid_can_path} \
--test_can_path ${test_can_path} \
--rst_prename ${rst_prename} \
--root_save_path ${root_save_path}

echo "infer other over"
