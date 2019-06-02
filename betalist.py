from textgenrnn import textgenrnn
textgen = textgenrnn(weights_path='betalist_model_weights.hdf5',
                       vocab_path='betalist_model_vocab.json',
                       config_path='betalist_model_config.json')

textgen.generate_samples(max_gen_length=1000)
textgen.generate_to_file('textgenrnn_texts.txt', max_gen_length=1000)