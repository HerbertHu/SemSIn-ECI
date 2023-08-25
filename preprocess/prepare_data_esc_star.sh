#python read_document.py --dataset_name=EventStoryLine_star
#python generate_sample_esc_star.py

#python get_amr_data.py --dataset_name=EventStoryLine_star

python get_amr_dict.py --dataset_name=EventStoryLine_star
python get_amr_triple.py --dataset_name=EventStoryLine_star
python get_align_data.py --dataset_name=EventStoryLine_star
python split_data_star.py
python build_graph.py --dataset_name=EventStoryLine_star

python split_topic_cv_star.py
python build_graph_cv.py --dataset_name=EventStoryLine_star
