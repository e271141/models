from abc import abstractmethod
from functools import partial
import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import box_predictor
from object_detection.core import losses
from object_detection.core import model
from object_detection.core import post_processing
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import variables_helper

slim = tf.contrib.slim



Faster-Rcnn full process:

A: prediction_dict = detection_model.predict()
	a: Train RPN (first stage)
	  0. preprocessed inputs
	  1. self._extract_rpn_feature_maps()
	    I: rpn_features_to_crop = self._feature_extractor.extract_proposal_features()
	    II: anchors = self._first_stage_anchor_generator.generate()
          -->GridAnchorGenerator._generate() -->GridAnchorGenerator.tile_anchors()
	    III: rpn_box_predictor_features = slim.conv2d()
	  2. self._predict_rpn_proposals()
	    I: box_predictions = self._first_stage_box_predictor.predict()
	    II: objectness_predictions_with_background = 
	                      box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND()
	  3. _remove_invalid_anchors_and_predictions
	    I: pruned_anchors_boxlist, keep_indices = box_list_ops.prune_outside_window()
	    II: _batch_gather_kept_indices(box_encodings)
	        _batch_gather_kept_indices(objectness_predictions_with_background)
	        'Extremely hard to get through!!!!!!'

	b: Classification (second stage)
	  1. _predict_second_stage
	  	I: flattened_proposal_feature_maps = self._postprocess_rpn()
      'Very complicate function!!!!!!'
        i: self._format_groundtruth_data(): 
        ii: decoded_boxes = self._box_coder.decode(rpn_box_encodings, box_list.BoxList(anchors))
                            --> faster_rcnn_box_coder.FasterRcnnBoxCoder._decode()
            objectness_scores = tf.nn.softmax(rpn_objectness_predictions_with_background)
        iii:proposal_boxlist = post_processing.multiclass_non_max_suppression()
        iv: padded_proposals = box_list_ops.pad_or_clip_box_list()
	  	II: self._compute_second_stage_input_feature_maps()
	  	III: box_classifier_features = self._feature_extractor.extract_box_classifier_features()
	  	IV: box_predictions = self._mask_rcnn_box_predictor.predict()
	  	V: absolute_proposal_boxes = ops.normalized_to_image_coordinates()

B: losses_dict = detection_model.loss
	a. _loss_rpn
	  1. target_assigner.batch_assign_targets()
      I: target_assigner.assign()
        i: match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes,anchors)
                -->sim_calc.IouSimilarity()-->box_list_ops.iou()
        ii: match = self._matcher.match(match_quality_matrix, **params)
                -->argmax_matcher.ArgMaxMatcher._match()
        iii: reg_targets = self._create_regression_targets(anchors,groundtruth_boxes,match)
        iv: cls_targets = self._create_classification_targets(groundtruth_labels,match)
        v: reg_weights = self._create_regression_weights(match)
        vi: cls_weights = self._create_classification_weights(
                        match, self._positive_class_weight, self._negative_class_weight)
    2. localization_losses = self._first_stage_localization_loss
          -->losses.WeightedSmoothL1LocalizationLoss._compute_loss()
	  3. objectness_losses = self._first_stage_objectness_loss
          -->losses.WeightedSoftmaxClassificationLoss._compute_loss()
	b. _loss_box_classifier
	  1. paddings_indicator = self._padded_batched_proposals_indicator
	  2. second_stage_loc_losses = self._second_stage_localization_loss
          -->losses.WeightedSmoothL1LocalizationLoss._compute_loss()
    3. second_stage_cls_losses = self._second_stage_classification_loss
          -->losses.WeightedSoftmaxClassificationLoss._compute_loss()
    4. second_stage_loc_loss, second_stage_cls_loss = self._unpad_proposals_and_apply_hard_mining
      I: _hard_example_miner()
          -->losses_builder.build_hard_example_miner()
              -->losses.HardExampleMiner()


  def predict(self, preprocessed_inputs):
    """Predicts unpostprocessed tensors from input tensor.

    This function takes an input batch of images and runs it through the
    forward pass of the network to yield "raw" un-postprocessed predictions.
    If `first_stage_only` is True, this function only returns first stage
    RPN predictions (un-postprocessed).  Otherwise it returns both
    first stage RPN predictions as well as second stage box classifier
    predictions.

    Other remarks:
    + Anchor pruning vs. clipping: following the recommendation of the Faster
    R-CNN paper, we prune anchors that venture outside the image window at
    training time and clip anchors to the image window at inference time.
    + Proposal padding: as described at the top of the file, proposals are
    padded to self._max_num_proposals and flattened so that proposals from all
    images within the input batch are arranged along the same batch dimension.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) rpn_box_predictor_features: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] to be used for predicting proposal
          boxes and corresponding objectness scores.
        2) rpn_features_to_crop: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] representing image features to crop
          using the proposal boxes predicted by the RPN.
        3) image_shape: a 1-D tensor of shape [4] representing the input
          image shape.
        4) rpn_box_encodings:  3-D float tensor of shape
          [batch_size, num_anchors, self._box_coder.code_size] containing
          predicted boxes.
        5) rpn_objectness_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, 2] containing class
          predictions (logits) for each of the anchors.  Note that this
          tensor *includes* background class predictions (at class index 0).
        6) anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
          for the first stage RPN (in absolute coordinates).  Note that
          `num_anchors` can differ depending on whether the model is created in
          training or inference mode.

        (and if first_stage_only=False):
        7) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, 4] representing predicted
          (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals
        8) class_predictions_with_background: a 3-D tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        9) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
        10) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes (in absolute coordinates).
        11) mask_predictions: (optional) a 4-D tensor with shape
          [total_num_padded_proposals, num_classes, mask_height, mask_width]
          containing instance mask predictions.
    """
    (rpn_box_predictor_features, rpn_features_to_crop, anchors_boxlist,
     image_shape) = self._extract_rpn_feature_maps(preprocessed_inputs)
    (rpn_box_encodings, rpn_objectness_predictions_with_background
    ) = self._predict_rpn_proposals(rpn_box_predictor_features)

    # The Faster R-CNN paper recommends pruning anchors that venture outside
    # the image window at training time and clipping at inference time.
    clip_window = tf.to_float(tf.stack([0, 0, image_shape[1], image_shape[2]]))
    if self._is_training:
      (rpn_box_encodings, rpn_objectness_predictions_with_background,
       anchors_boxlist) = self._remove_invalid_anchors_and_predictions(
           rpn_box_encodings, rpn_objectness_predictions_with_background,
           anchors_boxlist, clip_window)
    else:
      anchors_boxlist = box_list_ops.clip_to_window(
          anchors_boxlist, clip_window)

    anchors = anchors_boxlist.get()
    prediction_dict = {
        'rpn_box_predictor_features': rpn_box_predictor_features,
        'rpn_features_to_crop': rpn_features_to_crop,
        'image_shape': image_shape,
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'anchors': anchors
    }

    if not self._first_stage_only:
      prediction_dict.update(self._predict_second_stage(
          rpn_box_encodings,
          rpn_objectness_predictions_with_background,
          rpn_features_to_crop,
          anchors, image_shape))
    return prediction_dict