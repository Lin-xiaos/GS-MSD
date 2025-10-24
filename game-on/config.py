# Base data folder
root_dir = "/home/zxl/MultiTask classification/"

######## Memotion ########

# Train, Dev and Test csvs
me_train_csv_name = "data/train2.csv"
me_test_csv_name = "data/test2.csv"
me_dev_csv_name = "data/dev2.csv"

# Image graph data (Node Embeddings)
me_image_vec_dir = "proposed/test2/large-model/memotion/imagefeature/"

# Text graph data (Node Embeddings)
me_text_vec_dir = "proposed/test2/large-model/memotion/Textfeature/"


######## Mustard ########

# Train, Dev and Test csvs
mu_train_csv_name = "M2Seq2Seq-master/mustard-dataset-train.csv"
mu_test_csv_name = "M2Seq2Seq-master/mustard-dataset-test.csv"
mu_dev_csv_name = "M2Seq2Seq-master/mustard-dataset-dev.csv"

# Image graph data (Node Embeddings)
mu_image_vec_dir = "proposed/test2/large-model/mustard/imagefeature/"

# Text graph data (Node Embeddings)
mu_text_vec_dir = "proposed/test2/large-model/mustard/Textfeature/"

# Audio graph data (Node Embeddings)
mu_audio_vec_dir = "proposed/test2/AudioFeature/"


######## Parameters ########
batch_size = 32
epochs = 30
lr = 2e-4
gradient_accumulation_steps = 2
