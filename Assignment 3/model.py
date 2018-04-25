import os
import time
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Reshape, Dense, Activation, Dropout, Add
from utils.feature_extraction import load_datasets, DataParams, ModelParams, punc_pos, pos_prefix


class Parser(object):
    def __init__(self, config, word_embeddings, pos_embeddings, dep_embeddings):
        self.word_embeddings = word_embeddings
        self.pos_embeddings = pos_embeddings
        self.dep_embeddings = dep_embeddings
        self.config = config
        self.add_inputs()
        self.add_layers()
        self.model = Model(inputs=[self.word_input, self.pos_input, self.dep_input],
                           outputs=self.predictions)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.summary()

    def add_inputs(self):
        self.word_input = Input(shape=(self.config.word_features_types,), dtype='int32', name="batch_word_indices")
        self.pos_input = Input(shape=(self.config.pos_features_types,), dtype='int32', name="batch_pos_indices")
        self.dep_input = Input(shape=(self.config.dep_features_types,), dtype='int32', name="batch_dep_indices")

    def add_emb(self):
        word_embedding_input = Embedding(input_dim=self.word_embeddings.shape[0],
                                         output_dim=self.config.embedding_dim,
                                         weights=[self.word_embeddings],
                                         input_length=self.config.word_features_types,
                                         trainable=True)(self.word_input)
        pos_embedding_input = Embedding(input_dim=self.pos_embeddings.shape[0],
                                        output_dim=self.config.embedding_dim,
                                        weights=[self.pos_embeddings],
                                        input_length=self.config.pos_features_types,
                                        trainable=True)(self.pos_input)
        dep_embedding_input = Embedding(input_dim=self.dep_embeddings.shape[0],
                                        output_dim=self.config.embedding_dim,
                                        weights=[self.dep_embeddings],
                                        input_length=self.config.dep_features_types,
                                        trainable=True)(self.dep_input)

        word_embedding_input = Reshape((self.config.word_features_types * self.config.embedding_dim,))(
            word_embedding_input)
        pos_embedding_input = Reshape((self.config.pos_features_types * self.config.embedding_dim,))(
            pos_embedding_input)
        dep_embedding_input = Reshape((self.config.dep_features_types * self.config.embedding_dim,))(
            dep_embedding_input)

        return word_embedding_input, pos_embedding_input, dep_embedding_input

    def add_layers(self):

        print('Adding layers')
        # Layer 1
        word_embeddings, pos_embeddings, dep_embeddings = self.add_emb()
        layer_1_1 = Dense(units=self.config.l1_hidden_size,
                          activation=None,
                          kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=42))(
            word_embeddings)
        layer_1_2 = Dense(units=self.config.l1_hidden_size,
                          activation=None,
                          kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=42))(
            pos_embeddings)
        layer_1_3 = Dense(units=self.config.l1_hidden_size,
                          activation=None,
                          kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=42))(
            dep_embeddings)

        layer_1 = Add()([layer_1_1, layer_1_2, layer_1_3])
        layer_1 = Activation('sigmoid')(layer_1)
        h1 = Dropout(rate=0.2, seed=42)(layer_1)

        # Layer 2
        h2 = Dense(units=self.config.l2_hidden_size,
                   activation='relu',
                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=42))(h1)

        # Layer 3
        self.predictions = Dense(units=self.config.num_classes,
                                 activation='softmax',
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01,
                                                                                     seed=42))(h2)

    def compute_dependencies(self, data, dataset):
        sentences = data
        rem_sentences = [sentence for sentence in sentences]
        [sentence.clear_prediction_dependencies() for sentence in sentences]
        [sentence.clear_children_info() for sentence in sentences]

        while len(rem_sentences) != 0:
            curr_batch_size = min(dataset.model_config.batch_size, len(rem_sentences))
            batch_sentences = rem_sentences[:curr_batch_size]

            enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                               batch_sentences]
            enable_count = np.count_nonzero(enable_features)

            while enable_count > 0:
                curr_sentences = [sentence for i, sentence in enumerate(batch_sentences) if enable_features[i] == 1]

                # get feature for each sentence
                # call predictions -> argmax
                # store dependency and left/right child
                # update state
                # repeat

                curr_inputs = [
                    dataset.feature_extractor.extract_for_current_state(sentence, dataset.word2idx, dataset.pos2idx,
                                                                        dataset.dep2idx) for sentence in curr_sentences]
                word_inputs_batch = np.array([curr_inputs[i][0] for i in range(len(curr_inputs))])
                pos_inputs_batch = np.array([curr_inputs[i][1] for i in range(len(curr_inputs))])
                dep_inputs_batch = np.array([curr_inputs[i][2] for i in range(len(curr_inputs))])

                predictions = self.model.predict([word_inputs_batch, pos_inputs_batch, dep_inputs_batch])


                legal_labels = np.asarray([sentence.get_legal_labels() for sentence in curr_sentences],
                                          dtype=np.float32)
                legal_transitions = np.argmax(predictions + 1000 * legal_labels, axis=1)

                # update left/right children so can be used for next feature vector
                [sentence.update_child_dependencies(transition) for (sentence, transition) in
                 zip(curr_sentences, legal_transitions) if transition != 2]

                # update state
                [sentence.update_state_by_transition(legal_transition, gold=False) for (sentence, legal_transition) in
                 zip(curr_sentences, legal_transitions)]

                enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                                   batch_sentences]
                enable_count = np.count_nonzero(enable_features)

            # Reset stack and buffer
            [sentence.reset_to_initial_state() for sentence in batch_sentences]
            rem_sentences = rem_sentences[curr_batch_size:]

    def get_score(self, data):
        correct_tokens = 0
        all_tokens = 0
        punc_token_pos = [pos_prefix + each for each in punc_pos]
        for sentence in data:
            # reset each predicted head before evaluation
            [token.reset_predicted_head_id() for token in sentence.tokens]

            head = [-2] * len(sentence.tokens)
            # assert len(sentence.dependencies) == len(sentence.predicted_dependencies)
            for h, t, in sentence.predicted_dependencies:
                head[t.token_id] = h.token_id

            non_punc_tokens = [token for token in sentence.tokens if token.pos not in punc_token_pos]
            correct_tokens += sum([1 if token.head_id == head[token.token_id] else 0 for (_, token) in enumerate(
                non_punc_tokens)])

            # all_tokens += len(sentence.tokens)
            all_tokens += len(non_punc_tokens)

        UAS = correct_tokens / float(all_tokens)
        return UAS

    def run_valid_epoch(self, dataset):
        print("Evaluating on dev set")
        self.compute_dependencies(dataset.valid_data, dataset)
        valid_UAS = self.get_score(dataset.valid_data)
        print("- dev SCORE: {:.2f}".format(valid_UAS * 100.0))
        return valid_UAS

    def fit(self, dataset):

        best_valid_UAS = 0
        for epoch in range(self.config.n_epochs):

            self.model.fit(x=[dataset.train_inputs[0], dataset.train_inputs[1], dataset.train_inputs[2]],
                           y=dataset.train_targets,
                           batch_size=self.config.batch_size,
                           epochs=1)

            if (epoch + 1) % dataset.model_config.run_valid_after_epochs == 0:
                valid_UAS = self.run_valid_epoch(dataset)
                if valid_UAS > best_valid_UAS:
                    best_valid_UAS = valid_UAS
                    print("New best dev SCORE! Saving model..")
                    self.model.save(os.path.join(DataParams.data_dir_path, DataParams.model_dir,
                                                 DataParams.model_name))

    def restore(self, dataset):

        self.model = load_model(os.path.join(DataParams.data_dir_path, DataParams.model_dir, DataParams.model_name))
        self.compute_dependencies(dataset.test_data, dataset)
        test_UAS = self.get_score(dataset.test_data)
        print("test SCORE: {}".format(test_UAS * 100))


def highlight_string(temp):
    print(80 * "=")
    print(temp)
    print(80 * "=")


def main(flag, load_existing_dump=False):
    highlight_string("INITIALIZING")
    print("loading data..")

    dataset = load_datasets(load_existing_dump)
    np.save('dataset.npy', dataset)
    dataset = np.load('dataset.npy').item()
    config = dataset.model_config

    print("word vocab Size: {}".format(len(dataset.word2idx)))
    print("pos vocab Size: {}".format(len(dataset.pos2idx)))
    print("dep vocab Size: {}".format(len(dataset.dep2idx)))
    print("Training Size: {}".format(len(dataset.train_inputs[0])))
    print("valid data Size: {}".format(len(dataset.valid_data)))
    print("test data Size: {}".format(len(dataset.test_data)))

    if not os.path.exists(os.path.join(DataParams.data_dir_path, DataParams.model_dir)):
        os.makedirs(os.path.join(DataParams.data_dir_path, DataParams.model_dir))

    print("Building network...")
    start = time.time()
    model = Parser(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix, dataset.dep_embedding_matrix)
    print("took {:.2f} seconds\n".format(time.time() - start))

    if flag == ModelParams.TRAIN:
        highlight_string("TRAINING")

        model.fit(dataset)

        # Testing
        highlight_string("Testing")
        print("Restoring best found parameters on dev set")
        model.restore(dataset)


if __name__ == '__main__':
    main(ModelParams.TRAIN, load_existing_dump=False)
