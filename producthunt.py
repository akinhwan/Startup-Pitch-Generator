from textgenrnn import textgenrnn

def main(event, context):
    textgen = textgenrnn(weights_path='producthunt_model_weights.hdf5',
                        vocab_path='producthunt_model_vocab.json',
                        config_path='producthunt_model_config.json')

    textgen.generate_samples(max_gen_length=1000)
    textgen.generate_to_file('producthunt_output.txt', max_gen_length=1000)

if __name__ == "__main___":
    main('','')