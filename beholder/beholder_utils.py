import os
import csv
import cv2
import json
import torch
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from beholder.output_utils import (load_detections,
                                   build_tracks_df,
                                   draw_track_boxes)
from ngmot.tracker.visualize_tracks import compute_centroid, draw_track_curves
import beholder.transforms as T
from DistanceRegressor.distresnet import DistResNet18
from DistanceRegressor.mlpnet import DistanceRegressor


logger = logging.getLogger('fmv_tracker.visualizer')


def generate_video_stream(frames, stream, tracks_df, dists, output_file,
                          dets=None, tracks_src_size=None,
                          draw_track_history=False, smoothing=True,
                          codec='libx264', frame_rate=None,
                          attributes_max=3, attributes_thresh=0.5):
    """
    Generates a video file for a set of tracks.
    Args:
        frames (list): list of livestream frames
        tracks_df (dataframe): a dataframe of the tracks, one line per
            detection
        output_file (str): output video file name
        dets (dict): detections to be visualized if provided
        tracks_src_size (tuple): reference frame size for track roi's, if not
            the same as video.src_size
        draw_track_history (bool): if true, track history will be visualized
        smoothing (bool): whether to smooth the track history curve
        codec (str): The video codec to use; make sure filename extension
            matches
        frame_rate (int/float): Frame rate for output video. If not provided,
            video.fps will be used
        attributes_max: maximum number of track attributes to be displayed per
            track
        attributes_thresh: minimum score threshold for an attribute to be
            displayed
    """
    final_frames = []
    frame_start = 0
    frame_rate = stream.fps if frame_rate is None else frame_rate
    print('output file is ', output_file, ' fps ', frame_rate)
    # outvideo = get_ffmpeg_handle(
    #     output_file, frame_size, frame_rate=frame_rate, vcodec=codec)
    # print('out is ', outvideo)
    track_centroids = {}

    # group the tracklets by frame number
    gk = tracks_df.groupby('frame_num')

    num_frames = len(frames)
    print('there are ', num_frames)
    if num_frames is None:
        arr = np.array(gk['frame_num'])
        num_frames = arr.max() - arr.min()

    logger.info('Building video...')
    frame_idx = frame_start

    with tqdm(total=num_frames) as pbar:
        for frame in frames:
            if frame is None:
                logger.info('Error grabbing frame {} from source video. Video '
                            'finished'.format(frame_idx))
                break

            try:
                tracked_objects = gk.get_group(frame_idx)
                this_dist = dists[frame_idx]
            except Exception:
                tracked_objects = []

            if len(tracked_objects):
                # draw bounding boxes and track IDs for each tracked object
                tracks = tracked_objects.to_dict('records')
                # draw raw detection boxes
                frame_dets = None
                if dets is not None:
                    try:
                        frame_dets = dets[frame_idx]
                    except KeyError:
                        pass  # no detections for this frame
                frame = draw_track_boxes(frame, tracks, this_dist,
                                         dets=frame_dets,
                                         src_size=tracks_src_size,
                                         attributes_max=attributes_max,
                                         attributes_thresh=attributes_thresh)

                # update list of track centroids for each track in the current
                # frame and draw track history curves
                if draw_track_history:
                    ids = []
                    for _, row in tracked_objects.iterrows():
                        if not int(row['track_id']) in track_centroids:
                            track_centroids[int(row['track_id'])] = []
                        track_centroids[int(row['track_id'])].append(
                            compute_centroid(row, tracks_src_size, frame.size))
                        ids.append(int(row['track_id']))
                    # remove ids of any track that is no longer in current
                    # frame
                    missing_tracks = []
                    for track_id in track_centroids:
                        if track_id not in ids:
                            missing_tracks.append(track_id)
                    for track_id in missing_tracks:
                        del track_centroids[track_id]
                    frame = draw_track_curves(
                        frame, ids, track_centroids, smoothing)
            # ret = np.float32(np.array(frame)[:,:,::-1])
            # cv2.imwrite('outputs/test_new_dist/'+str(frame_idx)+'.jpg',ret)
            final_frames.append(frame)
            # write out the current annotated frame
            # write_frame_to_ffmpeg(outvideo, frame)
            frame_idx += 1
            pbar.update(1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, stream.src_size)

    for frame in final_frames:
        ret = np.uint8(np.array(frame)[:, :, ::-1])
        out.write(ret)
    out.release()
    # close_ffmpeg_handle(outvideo)


def visualize_stream(frames, stream, output_file, tracks, dists,
                     tracks_csv=None, labels_file=None, dets=None,
                     tracks_src_size=None, frame_rate=None, codec='libx264',
                     draw_track_history=False, smoothing=True,
                     attributes_max=3, attributes_thresh=0.5):
    """
    Main method for generating videos from output that can now be called
    from other files.
       Args:
           video (VideoReader): VideoReader object from which frames will be
            obtained for track visualization
           output_file (str): name for output video file
           tracks_df (pd.DataFrame): DataFrame containing final tracks to be
            visualized
           tracks_csv (str): path to .csv file containing track information if
            tracks_df is not provided
           labels_file (str): if using tracks_csv, path to file mapping track
            ids to track labels
           dets (dict or str): dictionary or filepath for detection information
            to be visualized
           tracks_src_size (tuple): reference frame size for track roi's, if
            not the same as video.src_size
           frame_rate (float): if provided, frame rate for output video. else
            the video framerate will be used
           codec (str): codec to be used by ffmpeg
           draw_track_history/smoothing (booleans): video creation options for
            visualizing track history via curve
           attributes_max: maximum number of track attributes to be displayed
            per track
           attributes_thresh: minimum score threshold for an attribute to be
            displayed
    """
    tracks_df = build_tracks_df(tracks)
    if tracks_df is None and tracks_csv is None:
        raise ValueError(
            'Please provide dataframe or .csv file containing track outputs.')
    if tracks_df is not None and tracks_csv is not None:
        raise ValueError(
            'Please provide dataframe or .csv file containing track outputs, '
            'but not both.')

    # load the tracks output
    if tracks_df is not None:
        if isinstance(tracks_df, str):
            with open(tracks_df, "rb") as f:
                tracks_df = pickle.load(f)
    else:
        with open(tracks_csv, 'r') as csvfile:
            sniffer = csv.Sniffer()
            line1 = csvfile.readline()
            line2 = csvfile.readline()
            has_header = sniffer.has_header(line1 + line2)
            ncols = len(line2.split(','))
        if has_header:
            tracks_df = pd.read_csv(tracks_csv)
        else:
            if ncols == 10:
                tracks_df = pd.read_csv(tracks_csv, header=None,
                                        names=['frame', 'track_id',
                                               'x', 'y', 'w', 'h', 'conf',
                                               1, 2, 3])
            else:
                tracks_df = pd.read_csv(tracks_csv, header=None,
                                        names=['frame', 'tracklet_id',
                                               'track_id', 'x', 'y', 'w', 'h',
                                               'conf', 1, 2, 3])
        rois = []
        for x, y, w, h in zip(tracks_df['x'], tracks_df['y'], tracks_df['w'],
                              tracks_df['h']):
            rois.append([x, y, x + w, y + h])
        tracks_df['roi'] = rois
        tracks_df = tracks_df.rename(columns={"frame": "frame_num"})

        # optionally load track labels
        labels_map = {}
        if labels_file:
            labels_df = pd.read_csv(labels_file)
            for i, row in labels_df.iterrows():
                if isinstance(row['label_name'], str):
                    if len(row['label_name']) > 0:
                        labels_map[int(row['tracklet_id'])] = row['label_name']
                else:
                    labels_map[int(row['tracklet_id'])] = 'class {}'.format(
                        int(row['label']))
        label_names = []
        for tracklet_id in tracks_df['tracklet_id']:
            label_names.append(labels_map.get(int(tracklet_id), 1))
        tracks_df['label_name'] = label_names

    # optionally load detections
    if isinstance(dets, str):
        # load detections from file
        if not os.path.isfile(dets):
            logger.error("Unable to draw detections because dets file not "
                         "found. Make sure that %s exists" % dets)
        else:
            dets = load_detections(dets)

    # generate the video
    logger.info('Building video at {}.'.format(output_file))
    generate_video_stream(frames, stream, tracks_df, dists, output_file, dets,
                          tracks_src_size,
                          draw_track_history=draw_track_history,
                          smoothing=smoothing,
                          codec=codec, frame_rate=frame_rate,
                          attributes_max=attributes_max,
                          attributes_thresh=attributes_thresh)
    logger.info('Done building video! Saved at {}'.format(output_file))


def save_dists(args, dists):
    path = os.path.join(args.output_dir, 'final_dists.json')
    with open(path, 'w') as f:
        json.dump(dists, f)


def get_dists(model, img, tracks, transforms, dist_min=0.914273901974966,
              dist_max=311.4842790022132, half_precision=False,
              model_name='resnet'):
    """
    Return distances of objects dected in tracks

    :param model: the model to predict with
    :param img: img as numpy array to inference
    :param tracks: list of json track objects
    :param transforms: image transform
    :param dist_min: minimum distance from dataset in meters
    :param dist_max: maximum distance from dataset in meters
    :param half_precision: sets the model to half precision when True. Defaults
        False
    :returns: list of tuples: [(track_id, object label, [bounding box],
                                predicted distance)]
    """

    # TODO: iterate through tracks and remove all tracks with label that is not
    # 0 or 1
    if len(tracks) == 0:
        return []

    ret = []
    boxes = []
    ret_tup = []
    resize = T.Resize((1024, 1024))
    device = torch.device('cuda:0')

    for track in tracks:
        label = track['label']

        # if not a person or a car move on
        if label != 0 and label != 1:
            continue

        ret.append((track['track_id'], label, track['roi']))
        boxes.append(track['roi'])

    # complete same transforms as when we trained the model
    im, boxes = resize(np.array(img), np.array(boxes))
    boxes = [torch.tensor(boxes).type(torch.FloatTensor)]
    if half_precision:
        boxes = [b.to(device).half() for b in boxes]
    else:
        boxes = [b.to(device) for b in boxes]

    if model_name == 'mlp':
        # scale bboxes from 0 to 1
        boxes = boxes / 1024.0

        # one hot encode out of six columns to either 0 (Pedestrian, column 0)
        # or 2 (Car, column 2)
        class_encodings = np.zeros((len(boxes), 6))
        inputs = np.concatenate((boxes, class_encodings), axis=1)

        for i in range(len(boxes)):
            if ret[i][1] == 0:
                # label is pedestrian, set column 0
                inputs[i][4] = 1.0
            elif ret[i][1] == 1:
                # label is car, set column 2
                inputs[i][6] = 1.0

        inputs = torch.tensor(inputs).type(torch.FloatTensor).to(device)
        if half_precision:
            inputs = inputs.half()

        distances = model(inputs)
    else:
        # torch.cuda.synchronize()
        # print(f'{time.time() - t1:.10f} resize boxes')

        # althought there is one image, this is necessary to
        # have the proper dimensions for the model
        im = transforms(im)
        im = torch.unsqueeze(im, dim=0)
        im = im.to(device)
        if half_precision:
            im = im.half()

        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f'{t2 - t1:.10f} image transform')

        _, distances = model(im, boxes)

    distances = (distances + 1) * (dist_max-dist_min) / 2.0 + dist_min
    distances = distances.to('cpu')

    for tup, dist in zip(ret, distances.tolist()):
        ret_tup.append([*tup, dist[0]])

    return ret_tup


def build_distnet(half_precision=False):
    """
    Builds disnet model and loads in weigths specified in config file

    :param half_precision: sets the model to half precision when True. Defaults
        False
    :returns: DistResNeXt50 model
    """
    model = DistResNet18(8, image_size=1024, pretrained=False, keypoints=False)
    model_pth = "weights/resnet_weights.pth"
    model.load_state_dict(torch.load(model_pth)[
                          'models_state_dict'][0], strict=False)

    # model half precision
    if half_precision:
        model.half()
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.float()

    # put model in evaluate mode
    model.to(torch.device('cuda:0'))
    model.eval()
    return model


def build_mlp(half_precision=False):
    """Builds mlp model and loads in weights specified in config file

    Args:
        half_precision (bool, optional): sets the model to half precision when
            True. Defaults to False.

    Returns:
        DistanceRegressor model
    """

    # nuscenes mapped to kitti (4 bbox, 6 classes)
    model = DistanceRegressor(n_features=10)
    model_pth = "ngmot/weights/mlp_weights.pth"
    model.load_state_dict(torch.load(model_pth)[
                          'models_state_dict'][0], strict=True)

    if half_precision:
        model.half()
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.float()

    model.to(torch.device('cuda:0'))
    model.eval()
    return model
