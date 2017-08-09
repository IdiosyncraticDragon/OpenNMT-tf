"""Sequence tagger."""

import tensorflow as tf

from opennmt.models.model import Model
from opennmt.utils.misc import count_lines
from opennmt.utils.losses import masked_sequence_loss


class SequenceTagger(Model):

  def __init__(self,
               embedder,
               encoder,
               labels_vocabulary_file,
               crf_decoding=False,
               name="seqtagger"):
    super(SequenceTagger, self).__init__(name)

    self.encoder = encoder
    self.embedder = embedder
    self.labels_vocabulary_file = labels_vocabulary_file
    self.num_labels = count_lines(labels_vocabulary_file)
    self.crf_decoding = crf_decoding

    self.id_to_label = []
    with open(labels_vocabulary_file) as labels_vocabulary:
      for label in labels_vocabulary:
        self.id_to_label.append(label.strip())

  def set_filters(self, maximum_length):
    self.maximum_length = maximum_length

  def _get_size(self, features, labels):
    return self.embedder.get_data_field(features, "length")

  def _get_maximum_size(self):
    return getattr(self, "maximum_length", None)

  def _filter_example(self, features, labels):
    """Filters examples with invalid length."""
    cond = tf.greater(self.embedder.get_data_field(features, "length"), 0)

    if hasattr(self, "maximum_length"):
      cond = tf.logical_and(
        cond,
        tf.less_equal(self.embedder.get_data_field(features, "length"),
                      self.maximum_length))

    return cond

  def _build_dataset(self, mode, batch_size, features_file, labels_file=None):
    features_dataset = tf.contrib.data.TextLineDataset(features_file)

    self.embedder.init()
    features_dataset = features_dataset.map(lambda x: self.embedder.process(x))

    if labels_file is None:
      dataset = features_dataset
      padded_shapes = self.embedder.padded_shapes
    else:
      labels_dataset = tf.contrib.data.TextLineDataset(labels_file)

      labels_vocabulary = tf.contrib.lookup.index_table_from_file(
        self.labels_vocabulary_file,
        vocab_size=self.num_labels)

      labels_dataset = labels_dataset.map(lambda x: tf.string_split([x]).values)
      labels_dataset = labels_dataset.map(lambda x: labels_vocabulary.lookup(x))

      dataset = tf.contrib.data.Dataset.zip((features_dataset, labels_dataset))
      padded_shapes = (self.embedder.padded_shapes, [None])

    return dataset, padded_shapes

  def _build(self, features, labels, params, mode):
    with tf.variable_scope("encoder"):
      inputs = self.embedder.embed_from_data(features, mode)
      self.embedder.visualize(params["log_dir"])

      encoder_outputs, encoder_states, encoder_sequence_length = self.encoder.encode(
        inputs,
        sequence_length=self.embedder.get_data_field(features, "length"),
        mode=mode)

    with tf.variable_scope("generator"):
      logits = tf.layers.dense(
        encoder_outputs,
        self.num_labels)

    if mode != tf.estimator.ModeKeys.PREDICT:
      if self.crf_decoding:
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
          logits,
          tf.cast(labels, tf.int32),
          self.embedder.get_data_field(features, "length"))
        loss = tf.reduce_mean(-log_likelihood)
      else:
        loss = masked_sequence_loss(
          logits,
          labels,
          self.embedder.get_data_field(features, "length"))

      return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=self._build_train_op(loss, params))
    else:
      predictions = {}
      predictions["length"] = encoder_sequence_length

      if self.crf_decoding:
        transition_params = tf.get_variable("transitions", shape=[self.num_labels, self.num_labels])

        # predictions must contain tensors with the same batch size
        # so replicate the transition matrix accordingly.
        transition_params = tf.convert_to_tensor(transition_params)
        transition_params = tf.expand_dims(transition_params, axis=0)
        transition_params = tf.tile(transition_params, [tf.shape(logits)[0], 1, 1])

        predictions["logits"] = logits
        predictions["transition_params"] = transition_params
      else:
        probs = tf.nn.softmax(logits)
        predictions["argmax"] = tf.argmax(probs, axis=2)

      return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions)

  def format_prediction(self, prediction, params=None):
    if self.crf_decoding:
      sequence, _ = tf.contrib.crf.viterbi_decode(
        prediction["logits"],
        prediction["transition_params"])
    else:
      sequence = prediction["argmax"]

    # Convert ids to labels.
    for i in range(prediction["length"]):
      sequence[i] = self.id_to_label[sequence[i]]

    labels = sequence[:prediction["length"]]
    sent = b' '.join(labels)
    sent = sent.decode('utf-8')
    return sent