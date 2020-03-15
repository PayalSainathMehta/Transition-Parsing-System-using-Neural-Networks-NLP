# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers


# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """

    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        return tf.math.pow(vector, 3)
        # raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        embedding_dim : ``str``
        Parameters
        ----------
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        #Initializing the values passed
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.num_transitions = num_transitions
        self.trainable_embeddings = trainable_embeddings

        #Initializing the embeddings
        self.embeddings = tf.Variable(tf.random.truncated_normal(shape=(vocab_size, embedding_dim), stddev=1. / math.sqrt(self.embedding_dim)
                                                                 ), trainable = trainable_embeddings)
        #Initializing weights for the first layer
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=(embedding_dim * num_tokens, hidden_dim), stddev=0.005))
        #Initializing weights for the second layer
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=(hidden_dim, num_transitions), stddev=0.005))
        #Initializing the bias
        self.b1 = tf.Variable(tf.zeros(shape=(1, hidden_dim)))
        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        #After initializations, we now lookup the respective values
        embedding_lookup = tf.nn.embedding_lookup(self.embeddings, inputs)
        embedding_2D = tf.reshape(embedding_lookup, [-1, self.num_tokens * self.embedding_dim])

        #The vector matrix is defined as h = (W1.xw + W1.xt + W1.xl + b1) where h is the vector_matrix
        vector_mat = tf.matmul(embedding_2D, self.W1)
        #We then add the bias to it
        vector_mat = tf.add(vector_mat, self.b1)
        #The cube of this is returned by the call function of the cubic activation class
        activated = self._activation(vector_mat)
        #Logits is defined as the softmax of W2.h
        logits = tf.matmul(activated, self.W2)
        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        #Now we write the loss function including regularization lambda
        r1 = self._regularization_lambda / 2.0
        r2 = tf.reduce_sum(self.W1 ** 2)
        r3 = tf.reduce_sum(self.b1 ** 2)
        r4 = tf.reduce_sum(self.W2 ** 2)
        if self.trainable_embeddings:
            r5 = tf.reduce_sum(self.embeddings ** 2)
        else:
            r5 = 0
        regularization = r1 * (r2 + r3 + r4 + r5)

        second_exp = tf.keras.backend.max(logits, axis=1, keepdims=True)
        logits_exp = tf.exp(logits - second_exp)

        #Taking a mask of only labels labeled 1s
        numerator = tf.cast(labels > 0, tf.float32)
        num = tf.reduce_sum(logits_exp * numerator, axis=1)

        #Taking a mask of labels labeled 1 and 0
        denominator = tf.cast(labels >= 0, tf.float32)
        denom = tf.reduce_sum(logits_exp * denominator, axis=1)

        loss = -tf.reduce_mean(tf.math.log((num / denom) + 1e-10))
        # TODO(Students) End
        return loss + regularization
