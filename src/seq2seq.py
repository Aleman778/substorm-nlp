import spacy
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm

class Seq2Seq(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.enc_cell = nn.RNN(input_size=embedding_size,
                               hidden_size=hidden_size, # connects to dec_cell and the next time step (aka. token)
                               num_layers=2,
                               dropout=0.5) # dropout means only training certain neurons at a time.
        self.dec_cell = nn.RNN(input_size=embedding_size, 
                               hidden_size=hidden_size, 
                               num_layers=2, 
                               dropout=0.5)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, enc_inputs, enc_hidden, dec_inputs):
        enc_inputs = enc_inputs.transpose(0, 1)
        dec_inputs = dec_inputs.transpose(0, 1)

        _, enc_state = self.enc_cell(enc_inputs, enc_hidden)
        dec_outputs, _ = self.dec_cell(dec_inputs, enc_state)
        return self.fc(dec_outputs)

def extract_word_embeddings(nlp, keywords, text):
    result = []
    for entry in text:
        doc = nlp(entry)
        vectors = []
        for i, token in enumerate(doc):
            if token.text == "<" and doc[i + 2].text == ">": # HACK(alexander): must be a better way to do this
                if ("<" + doc[i + 1].text + ">") in keywords:
                    vectors.append(keywords["<" + doc[i + 1].text + ">"])
                    continue
            vectors.append(doc.vector)
        result.append(vectors)
    return result
                
def correct_vector_length(vectors, target_length, pad_vector):
    result = []
    for v in vectors:
        while len(v) > target_length: # Split into two entries
            sub = v[:len(v)]
            v = v[len(v) + 1:]
            result.append(v)

        if len(v) == target_length: # Perfect this one is done
            result.append(v)
        else: # Add padding to reach the targe length
            for i in range(target_length-len(v)):
                v.append(pad_vector)
            result.append(v)
    return result

if __name__ == "__main__":
    # Hyper-parameters (tunable parameters to improve training)
    lr_rate = 0.001 # factor of how much the network weights should change per training batch.
    num_epochs = 5000 # number of times to train (i.e. change weights) the model.
    num_steps = 20 # the number of words that can appear in sequence.
    embedding_size = 300 # the size of vectors, spaCy uses 300 dimensions for their embeddings.
    hidden_size = 128 # size of the hidden state in RNN, chosen arbitrarily.

    # Find device, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU");
    
    # Seting up the dataset
    nlp = spacy.load("en_core_web_md")
    dataset_inputs  = ["<START> Email alemen-6@student.ltu.se saying \"Hello world!\". <END>",
                       "<START> Send email to my friend, I'm about 15 minutes late for school. <END>"
                       "<START> I'm about 15 minutes late for work, send email to my colleagues <END>"]
    dataset_outputs = ["<COMMAND> email <TO> alemen-6@student.ltu.se <BODY> Hello world! <END>",
                       "<COMMAND> email <TO> my friend <BODY> I'm about 15 minutes late for school. <END>",
                       "<COMMAND> email <TO> my colleagues <BODY> I'm about 15 minutes late for work. <END>"]

    # Add custom vectors for tagging stuff
    keywords = {"<START>": np.random.uniform(-1, 1, (300,)),
                "<COMMAND>": np.random.uniform(-1, 1, (300,)),
                "<END>": np.random.uniform(-1, 1, (300,)),
                "<TO>": np.random.uniform(-1, 1, (300,)),
                "<BODY>": np.random.uniform(-1, 1, (300,)),
                "<PAD>": np.random.uniform(-1, 1, (300,))}
    for word, vector in keywords.items():
        nlp.vocab.set_vector(word, vector)
    # TODO(alexander): These needs to be stored so we can interpret future uses of this model

    # Convert sentences into vectors of words
    input_vectors = extract_word_embeddings(nlp, keywords, dataset_inputs)
    target_vectors = extract_word_embeddings(nlp, keywords, dataset_targets)

    # Make sure that each sentence have num_steps vectors, add padding if needed
    # NOTE: the neural network requires sequences to have the same dimensions, strict requirement.
    input_vectors = correct_vector_length(input_vectors, num_steps, keywords["<PAD>"])
    target_vectors = correct_vector_length(target_vectors, num_steps, keywords["<PAD>"])

    # Output is the target shifted by one
    output_vectors = []
    # TODO(alexander): implement this somehow :)
    # for i, v in enumerate(target_vectors):
        # output_vectors.append(v[1:])
        # if i < len(target_vectors) - 1:
            # output_vectors[i].append(output_

    # Setting up the model (criterion/optimizare are also hyper-parameters)
    model = Seq2Seq(embedding_size, hidden_size)
    criterion = nn.MSELoss() # TODO(alexander); don't know if this is good loss to use
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    filename = "seq2seq_checkpoint.pt" # used for saving/loading trained models
    if True:
        print("Training started")
        lowest_loss = 1000
        model = model.to(device)
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            # Pick a random sample from the training set
            idx = random.randint(0, len(input_vectors) - 1)
            inputs = torch.FloatTensor([input_vectors[idx]]).to(device)
            hidden = torch.FloatTensor(torch.zeros(2, 1, hidden_size)).to(device)
            outputs = torch.FloatTensor([output_vector[idx]])
            targets = torch.FloatTensor([target_vectors[idx]]).to(device)

            optimizer.zero_grad() # resets the gradiens from previous epoch
            outputs = model(inputs, hidden, targets)
            outputs = outputs.transpose(0, 1)
            loss = criterion(outputs, targets)
            loss.backward();
            optimizer.step();
            epoch_loss = loss.item()

            if (epoch % 20) == 0:
                pbar.set_description("Training model - loss: %.6f" % epoch_loss)

            if (epoch % 100) == 0:
                if epoch_loss < lowest_loss:
                    torch.save(model.state_dict(), filename)
                    lowest_loss = epoch_loss
                    print("Reached lower loss of %.6f at epoch %d saving to %s" % (lowest_loss, epoch + 1, filename))
        print("Training ended")
    else:
        print("Loading previously trained model...")
        model.load_state_dict(torch.load(filename))
        print("Loaded `" + filename + "` model successfully.")

    # Testing the model
    if True:
        print("Testing the model")
        model = model.to(device)
        model.eval(); # Ignore dropout
        with torch.no_grad(): # Ignore gradient calculations, lower memory footprint
            # test = input("> ").split(' ')
            # test_text = "Send an email with information about my project to my group members"
            # test_text = "Email alemen-6@student.ltu.se saying \"Hello world!\"."
            test_text = "Send email to my friend, I'm about 15 minutes late for school."
            print("input:", test_text)

            test_text = "<START> " + test_text + " <END>"
            # The output needs to be fed into the network, just give it a bunch of padding tokens.
            # HACK(alexander): kind of works but should rely on spacys nlp instead.
            pad_text = "<PAD> "*(len(test_text.split(' ')))
            pad_text = "<START> " + pad_text + " <END>"

            # Preprocess
            test_vectors = extract_word_embeddings(nlp, keywords, [test_text])
            pad_vectors = extract_word_embeddings(nlp, keywords, [pad_text])

            test_vectors = correct_vector_length(test_vectors, num_steps, keywords["<PAD>"])
            pad_vectors = correct_vector_length(pad_vectors, num_steps, keywords["<PAD>"])

            # Create tensors for pytorch
            inputs = torch.FloatTensor([test_vectors[0]]).to(device)
            hidden = torch.FloatTensor(torch.zeros(2, 1, hidden_size)).to(device)
            padding = torch.FloatTensor([input_vectors[0]]).to(device)

            outputs = model(inputs, hidden, padding).cpu()

            result = ""
            for v in outputs:
                outputs = np.reshape(v.numpy(), (1, 300))
                keys, _, _ = nlp.vocab.vectors.most_similar(v)
                for k in keys:
                    result += nlp.vocab[k[0]].text + " "
            print("output:", result)
    print("Exiting the program")
