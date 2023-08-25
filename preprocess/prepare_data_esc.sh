#python read_document.py --dataset_name=EventStoryLine
#python generate_sample_esc.py

#python get_amr_data.py --dataset_name=EventStoryLine

python get_amr_dict.py --dataset_name=EventStoryLine
python get_amr_triple.py --dataset_name=EventStoryLine
python get_align_data.py --dataset_name=EventStoryLine
python split_data.py --dataset_name=EventStoryLine
python build_graph.py --dataset_name=EventStoryLine

python split_topic_cv.py --dataset_name=EventStoryLine
python build_graph_cv.py --dataset_name=EventStoryLine
