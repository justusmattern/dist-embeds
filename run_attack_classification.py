import os

# for wordLSTM target
command = 'python attack_classification.py --dataset_path data/mr/test.txt ' \
           '--target_model wordLSTM --batch_size 128 ' \
           '--target_model_path model_nondist.pt ' \
           '--word_embeddings_path data/glove.6B.200d.txt ' \
           '--counter_fitting_embeddings_path data/counter-fitted-vectors.txt ' \
           '--counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy ' \
           '--USE_cache_path tf_cache'

# for BERT target
#command = 'python attack_classification.py --dataset_path data/mr ' \
#          '--target_model lstm ' \
#          '--target_model model_nondist.pt' \
#          '--max_seq_length 256 --batch_size 32 ' \
#          '--counter_fitting_embeddings_path data/counter-fitted-vectors.txt ' \
#          '--counter_fitting_cos_sim_path data/cos_sim_counter_fitting.npy ' \
#          '--USE_cache_path tf_cache'

os.system(command)
