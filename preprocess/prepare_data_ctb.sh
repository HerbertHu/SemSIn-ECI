#python read_document_ctb.py
#python generate_sample_ctb.py

#python get_amr_data.py --dataset_name=Causal-TimeBank

python get_amr_dict.py --dataset_name=Causal-TimeBank
python get_amr_triple.py --dataset_name=Causal-TimeBank
python get_align_data.py --dataset_name=Causal-TimeBank

python split_doc_cv_ctb.py
python build_graph_cv_ctb.py
